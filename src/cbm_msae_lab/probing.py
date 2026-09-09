"""Linear-probe "downstream task accuracy" evaluation.

Follows the CFM paper's own protocol (App. B.5): an AdamW linear head, batch
1024, learning rate and an L1 weight-sparsity penalty swept, the best
checkpoint selected by held-out validation accuracy, test accuracy reported
with that selected configuration averaged over a few seeds. This answers
"how much of CUB's 200-way classification signal survives the SAE's concept
bottleneck, compared to the raw feature it was trained to reconstruct?" --
run once per finished checkpoint by `scripts/evaluate_task_accuracy.py`,
never during training itself.

Ported and trimmed from `cfm_finegrained/src/cfm_fg/probes.py` (there,
validated against the CFM paper's own released ImageNet numbers). The only
material simplification: `cfm_finegrained` also supported a host-resident
float16 streaming path for full-ImageNet-scale representations (a single
`a_meanmax` row is 84 GB there); CUB's largest representation here
(dict_size=8192 over ~11.8k images, ~390 MB) comfortably fits on one GPU, so
everything stays resident and that branch is dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import Tensor

BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ProbeConfig:
    """The CFM paper's own probe recipe (App. B.5).

    `standardize=False` is deliberate, not an oversight: z-scoring an
    almost-always-zero sparse SAE code divides near-constant-zero dimensions
    by a tiny standard deviation and hurts its accuracy badly (`cfm_finegrained`
    measured ~16pp on ImageNet for exactly this reason); the dense 512-d F+
    baseline is roughly unaffected either way, so leaving standardization off
    keeps one recipe fair across every representation being compared.
    `epochs=8000` is CUB-scale: with ~9.5k training images and batch_size=1024,
    that is only ~10 optimizer steps per epoch, so many epochs are needed for
    the sparse (`z`) representation in particular to converge.

    `lr_grid` adds 1e-2 to CFM's own (1e-4, 1e-3). CFM only ever probed features
    on one fixed scale; here the representations being compared can differ in
    magnitude by several times, and with `standardize=False` a too-small learning
    rate simply underfits within `epochs` rather than converging to a worse
    optimum. Selection is still by held-out validation accuracy, so the wider
    grid cannot make any representation look better than it is -- it only stops
    a scale mismatch from being reported as a representation-quality difference.
    """

    lr: float = 1e-4
    batch_size: int = 1024
    epochs: int = 8000
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    bias: bool = False
    standardize: bool = False
    lambda_grid: tuple[float, ...] = field(default_factory=lambda: (0.0, 0.1, 1.0))
    lr_grid: tuple[float, ...] = field(default_factory=lambda: (1e-4, 1e-3, 1e-2))


@dataclass(frozen=True)
class SplitTensors:
    """One pooled representation, split into train/val/test, resident on `device`."""

    x_train: Tensor  # [N_train, dim]
    y_train: Tensor  # [N_train]
    x_val: Tensor  # [N_val, dim]
    y_val: Tensor  # [N_val]
    x_test: Tensor  # [N_test, dim]
    y_test: Tensor  # [N_test]
    n_classes: int
    device: torch.device

    @property
    def dim(self) -> int:
        return int(self.x_train.shape[1])


@dataclass(frozen=True)
class SeedResult:
    seed: int
    lam: float
    val_top1: float
    test_top1: float
    test_top5: float
    best_epoch: int


@dataclass(frozen=True)
class ProbeResult:
    """Aggregated over seeds; this is what gets written to the report JSON."""

    name: str
    dim: int
    test_top1_mean: float
    test_top1_std: float
    test_top5_mean: float
    val_top1_mean: float
    selected_lambda: float
    selected_lr: float
    chance: float
    per_seed: tuple[SeedResult, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["per_seed"] = [asdict(s) for s in self.per_seed]
        return d


def prepare_splits(
    x: Tensor,
    labels: IntArray,
    train: BoolArray,
    val: BoolArray,
    test: BoolArray,
    n_classes: int,
    device: torch.device,
    standardize: bool,
) -> SplitTensors:
    """x: [N, dim] (one pooled representation for every image, across all
    splits, in the same row order as `labels`/`train`/`val`/`test`) -> GPU-resident
    per-split tensors, optionally z-scored using train-split statistics only."""
    xt = x.to(device=device, dtype=torch.float32)  # [N, dim]
    y = torch.from_numpy(labels).to(device)  # [N]
    m_tr = torch.from_numpy(train).to(device)
    m_va = torch.from_numpy(val).to(device)
    m_te = torch.from_numpy(test).to(device)

    if standardize:
        mean = xt[m_tr].mean(0, keepdim=True)  # [1, dim]
        std = xt[m_tr].std(0, keepdim=True).clamp_min(1e-6)  # [1, dim]
        xt = (xt - mean) / std

    return SplitTensors(
        x_train=xt[m_tr],
        y_train=y[m_tr],
        x_val=xt[m_va],
        y_val=y[m_va],
        x_test=xt[m_te],
        y_test=y[m_te],
        n_classes=n_classes,
        device=device,
    )


def _accuracy(logits: Tensor, y: Tensor, k: int = 1) -> float:
    """logits: [N, n_classes], y: [N] -> top-k accuracy as a plain float."""
    if k == 1:
        return float((logits.argmax(1) == y).float().mean())
    return float((logits.topk(k, dim=1).indices == y[:, None]).any(1).float().mean())


def fit_head(data: SplitTensors, cfg: ProbeConfig, lam: float, seed: int) -> tuple[torch.nn.Linear, float, int]:
    """Trains one linear probe head; returns it (loaded with its best-validation
    weights) together with that best validation top-1 accuracy and the epoch
    it was reached at."""
    torch.manual_seed(seed)
    device = data.device
    head = torch.nn.Linear(data.dim, data.n_classes, bias=cfg.bias).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)

    n = int(data.x_train.shape[0])
    gen = torch.Generator(device=device).manual_seed(seed)
    best_val = -1.0
    best_state: dict[str, Tensor] = {}
    best_epoch = -1
    eval_every = max(1, cfg.epochs // 200)  # cap how often we pay for a val pass

    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device, generator=gen)  # [N_train]
        for s in range(0, n, cfg.batch_size):
            idx = perm[s : s + cfg.batch_size]
            logits = head(data.x_train[idx])  # [b, n_classes]
            loss = F.cross_entropy(logits, data.y_train[idx])
            if lam > 0.0:
                loss = loss + lam * head.weight.abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if epoch % eval_every == 0 or epoch == cfg.epochs - 1:
            with torch.no_grad():
                va = _accuracy(head(data.x_val), data.y_val)
            if va > best_val:
                best_val = va
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                best_epoch = epoch

    head.load_state_dict(best_state)
    return head, best_val, best_epoch


def train_once(data: SplitTensors, cfg: ProbeConfig, lam: float, seed: int) -> SeedResult:
    """Trains one probe and reports its validation-selected test accuracy."""
    head, best_val, best_epoch = fit_head(data, cfg, lam, seed)
    with torch.no_grad():
        logits = head(data.x_test)  # [N_test, n_classes]
        top1 = _accuracy(logits, data.y_test, k=1)
        top5 = _accuracy(logits, data.y_test, k=min(5, data.n_classes))
    return SeedResult(seed=seed, lam=lam, val_top1=best_val, test_top1=top1, test_top5=top5, best_epoch=best_epoch)


def run_probe(
    name: str,
    x: Tensor,
    labels: IntArray,
    train: BoolArray,
    val: BoolArray,
    test: BoolArray,
    n_classes: int,
    cfg: ProbeConfig,
    device: torch.device,
    seeds: Sequence[int],
) -> ProbeResult:
    """Full protocol for one representation: sweep lr and the L1 lambda on a
    single seed, selected by validation accuracy; then average test accuracy
    over every seed at that selected configuration."""
    data = prepare_splits(x, labels, train, val, test, n_classes, device, cfg.standardize)

    probe_seed = int(seeds[0])
    best: SeedResult | None = None
    best_lr = cfg.lr
    lam = 0.0
    for lr in cfg.lr_grid:
        trial_cfg = replace(cfg, lr=lr)
        for candidate_lam in cfg.lambda_grid:
            trial = train_once(data, trial_cfg, candidate_lam, probe_seed)
            if best is None or trial.val_top1 > best.val_top1:
                best, best_lr, lam = trial, lr, candidate_lam
    assert best is not None
    chosen = replace(cfg, lr=best_lr)

    results: list[SeedResult] = [best]
    for extra_seed in seeds[1:]:
        results.append(train_once(data, chosen, lam, int(extra_seed)))

    top1 = np.asarray([r.test_top1 for r in results], dtype=np.float64)
    top5 = np.asarray([r.test_top5 for r in results], dtype=np.float64)
    vals = np.asarray([r.val_top1 for r in results], dtype=np.float64)

    return ProbeResult(
        name=name,
        dim=data.dim,
        test_top1_mean=float(top1.mean()),
        test_top1_std=float(top1.std(ddof=0)),
        test_top5_mean=float(top5.mean()),
        val_top1_mean=float(vals.mean()),
        selected_lambda=lam,
        selected_lr=best_lr,
        chance=1.0 / n_classes,
        per_seed=tuple(results),
    )
