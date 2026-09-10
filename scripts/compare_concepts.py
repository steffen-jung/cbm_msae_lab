"""Side-by-side concept visualization for two checkpoints on the same images.

Loads two checkpoints, runs both over the *same* val split (guaranteed only
when both were trained on the same dataset config -- checked at startup), and
for a handful of concepts per model renders the top-K activating patch crops
as one composite figure.

Concept indices between the two models are meaningless to compare directly
(independently initialized/trained dictionaries), so concepts are selected
*within* each model: a few frequently-active ones for a general impression,
plus the concept with the lowest and the highest region participation ratio
(strongest vs. weakest spatial specialization) -- the property a
participation-ratio loss or Group-TopK run is meant to move.

Headless by design (writes a PNG, no interactive cells) so it can run under
sbatch; the streaming top-K buffer and patch-crop logic are ported from
`notebooks/concept_monosemanticity.ipynb`, generalized from one checkpoint to
two run side by side over one shared loader pass.

Example:
    uv run scripts/compare_concepts.py \\
        --checkpoint outputs/checkpoints/cub_cfm_vanilla/best.pt cfm_vanilla \\
        --checkpoint outputs/checkpoints/cub_pr_feature/best.pt pr_feature \\
        --out outputs/concept_comparison/cfm_vanilla_vs_pr_feature.png
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cbm_msae_lab.checkpointing import load_for_reproduction
from cbm_msae_lab.metrics.sparsity import ActivationFrequencyMetric
from cbm_msae_lab.participation_ratio import participation_ratio

log = logging.getLogger(__name__)


@dataclass
class ModelRun:
    name: str
    checkpoint_path: str
    pipeline: object
    dict_size: int
    top_values: Tensor  # [dict_size, top_k]
    top_image_idx: Tensor
    top_row: Tensor
    top_col: Tensor
    frequency: Tensor | None = None  # [dict_size], filled in after the pass
    region_pr: Tensor | None = None  # [dict_size], nanmean over images where active


def build_model_run(checkpoint_path: str, name: str, device: str, top_k: int) -> tuple[ModelRun, object]:
    """Returns the run plus the (cfg, dataset) it was trained with, so the caller can
    verify all runs share the same dataset config before assuming a shared image index."""
    cfg, pipeline = load_for_reproduction(checkpoint_path, device=device)
    pipeline.eval()
    dict_size = pipeline.sae.dict_size
    run = ModelRun(
        name=name,
        checkpoint_path=checkpoint_path,
        pipeline=pipeline,
        dict_size=dict_size,
        top_values=torch.full((dict_size, top_k), -float("inf")),
        top_image_idx=torch.full((dict_size, top_k), -1, dtype=torch.long),
        top_row=torch.full((dict_size, top_k), -1, dtype=torch.long),
        top_col=torch.full((dict_size, top_k), -1, dtype=torch.long),
    )
    return run, cfg


@torch.no_grad()
def update_top_k(run: ModelRun, concept_values: Tensor, image_idx: Tensor, row: Tensor, col: Tensor) -> None:
    """concept_values/image_idx/row/col: [dict_size, B*H*W]. Merges this batch's
    candidates into the running per-concept top-k buffer (ported verbatim from
    the notebook's `update_top_k`)."""
    top_k = run.top_values.shape[1]
    cand_values = torch.cat([run.top_values, concept_values], dim=1)
    cand_image_idx = torch.cat([run.top_image_idx, image_idx], dim=1)
    cand_row = torch.cat([run.top_row, row], dim=1)
    cand_col = torch.cat([run.top_col, col], dim=1)

    keep = cand_values.topk(top_k, dim=1).indices  # [dict_size, top_k]
    run.top_values.copy_(cand_values.gather(1, keep))
    run.top_image_idx.copy_(cand_image_idx.gather(1, keep))
    run.top_row.copy_(cand_row.gather(1, keep))
    run.top_col.copy_(cand_col.gather(1, keep))


def crop_patch(image: Tensor, row: int, col: int, grid_h: int, grid_w: int, margin: int):
    """image: [3, image_size, image_size] in [0,1] -> HWC numpy crop around patch (row, col)."""
    image_size = image.shape[-1]
    stride_h = image_size // grid_h
    stride_w = image_size // grid_w
    y0 = max(0, row * stride_h - margin)
    y1 = min(image_size, (row + 1) * stride_h + margin)
    x0 = max(0, col * stride_w - margin)
    x1 = min(image_size, (col + 1) * stride_w + margin)
    return image[:, y0:y1, x0:x1].permute(1, 2, 0).numpy()


def select_concepts(run: ModelRun, n_frequent: int, region_tile_size: int) -> list[tuple[int, str]]:
    """(concept_id, label) pairs: the `n_frequent` most-active non-dead concepts,
    plus the non-dead concept with the lowest and the highest region_pr."""
    dead = run.frequency == 0
    live = (~dead).nonzero(as_tuple=True)[0]
    if live.numel() == 0:
        return []

    by_freq = live[run.frequency[live].argsort(descending=True)]
    frequent = by_freq[:n_frequent].tolist()

    region_pr = run.region_pr[live]
    valid = torch.isfinite(region_pr)
    picks = [(c, f"freq #{i + 1}") for i, c in enumerate(frequent)]
    if valid.any():
        finite_live = live[valid]
        finite_pr = region_pr[valid]
        most_local = int(finite_live[finite_pr.argmin()])
        most_spread = int(finite_live[finite_pr.argmax()])
        if most_local not in frequent:
            picks.append((most_local, f"lowest region_pr (tile={region_tile_size})"))
        if most_spread not in frequent and most_spread != most_local:
            picks.append((most_spread, f"highest region_pr (tile={region_tile_size})"))
    return picks


def render_comparison(
    runs: list[ModelRun],
    dataset: Dataset,
    selections: list[list[tuple[int, str]]],
    grid_h: int,
    grid_w: int,
    top_k: int,
    crop_margin: int,
    n_cols: int,
    out_path: Path,
) -> None:
    rows = [(run, concept, label) for run, picks in zip(runs, selections, strict=True) for concept, label in picks]
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows), squeeze=False)

    for r, (run, concept, label) in enumerate(rows):
        freq = float(run.frequency[concept])
        pr = float(run.region_pr[concept])
        axes[r, 0].set_ylabel(
            f"{run.name}\nconcept {concept}\n{label}\nfreq={freq:.4f}\nregion_pr={pr:.2f}",
            fontsize=8,
            rotation=0,
            ha="right",
            va="center",
            labelpad=60,
        )
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= top_k:
                ax.axis("off")
                continue
            img_idx = int(run.top_image_idx[concept, c])
            row = int(run.top_row[concept, c])
            col = int(run.top_col[concept, c])
            value = float(run.top_values[concept, c])
            image, _label, _idx = dataset[img_idx]
            crop = crop_patch(image, row, col, grid_h, grid_w, crop_margin)
            ax.imshow(crop)
            ax.set_title(f"idx={img_idx}\nact={value:.2f}", fontsize=7)

    fig.suptitle("Concept comparison: top activating patches per concept", fontsize=12)
    fig.tight_layout(rect=(0.12, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    log.info(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        nargs=2,
        metavar=("PATH", "NAME"),
        action="append",
        required=True,
        help="a checkpoint to include, with a short display name; pass twice for two models",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--top-k", type=int, default=8, help="top activating patches shown per concept")
    parser.add_argument("--n-frequent", type=int, default=3, help="most-active concepts to show per model")
    parser.add_argument("--region-tile-size", type=int, default=2, help="tile size for the region_pr selection")
    parser.add_argument("--crop-margin", type=int, default=8, help="pixels of context around each patch crop")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="outputs/concept_comparison/comparison.png")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    runs: list[ModelRun] = []
    cfgs = []
    dataset = None
    for checkpoint_path, name in args.checkpoint:
        run, cfg = build_model_run(checkpoint_path, name, args.device, args.top_k)
        runs.append(run)
        cfgs.append(cfg)

    base_dataset_cfg = cfgs[0].dataset
    for cfg, (checkpoint_path, name) in zip(cfgs[1:], args.checkpoint[1:], strict=True):
        if cfg.dataset != base_dataset_cfg:
            raise ValueError(
                f"{name!r} ({checkpoint_path}) was trained on a different dataset config than "
                f"{args.checkpoint[0][1]!r} -- the val split (and hence image indices) would not "
                f"line up between models. Compare checkpoints trained on the same dataset config only."
            )

    import hydra.utils

    dataset = hydra.utils.instantiate(base_dataset_cfg, split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    grid_h = grid_w = None
    region_labels = None  # CPU, built once the grid is known
    for run in runs:
        frequency_metric = ActivationFrequencyMetric(run.dict_size)  # CPU: post-processing never touches the GPU
        pr_sum = pr_active = None  # per-concept region_pr accumulators, sized once the grid is known

        global_offset = 0
        for images, _labels, _idx in tqdm(loader, desc=run.name):
            images = images.to(args.device)
            with torch.no_grad():
                out = run.pipeline.forward(images, use_threshold=True)
            # Off the GPU immediately: two encoder+SAE pipelines share one GPU here, and
            # everything below (top-k merge, region_pr, frequency) is index bookkeeping
            # that gains nothing from CUDA but, expanded to [dict_size, B*P], is exactly
            # what OOM'd this script the first time it ran.
            latents = out.latents.detach().cpu()  # [B, dict_size, H, W]
            B, dict_size, H, W = latents.shape
            grid_h, grid_w = H, W

            if region_labels is None or region_labels.shape[0] != H * W:
                from cbm_msae_lab.losses.tiling import tile_group_labels

                region_labels = tile_group_labels(H, W, args.region_tile_size)
            if pr_sum is None:
                pr_sum = torch.zeros(dict_size)
                pr_active = torch.zeros(dict_size)

            latents_img = latents.permute(0, 2, 3, 1).reshape(B, H * W, dict_size)  # [B, P, dict_size]
            frequency_metric.update(latents_img.reshape(B * H * W, dict_size))

            n_regions = int(region_labels.max()) + 1
            result = participation_ratio(latents_img, region_labels.unsqueeze(0).expand(B, -1), n_regions)
            pr_sum += (result.pr * result.active).sum(dim=0)
            pr_active += result.active.sum(dim=0)

            rows_grid = torch.arange(H).repeat_interleave(W)  # [P]
            cols_grid = torch.arange(W).repeat(H)  # [P]
            image_idx_grid = global_offset + torch.arange(B)  # [B]

            concept_values = latents_img.permute(2, 0, 1).reshape(dict_size, B * H * W)
            image_idx_bp = image_idx_grid.view(B, 1).expand(B, H * W).reshape(1, B * H * W).expand(dict_size, -1)
            row_bp = rows_grid.view(1, H * W).expand(B, H * W).reshape(1, B * H * W).expand(dict_size, -1)
            col_bp = cols_grid.view(1, H * W).expand(B, H * W).reshape(1, B * H * W).expand(dict_size, -1)

            update_top_k(run, concept_values, image_idx_bp, row_bp, col_bp)
            global_offset += B

        run.frequency = frequency_metric.compute()
        run.region_pr = torch.where(
            pr_active > 0, pr_sum / pr_active.clamp_min(1), torch.full_like(pr_sum, float("nan"))
        )

    selections = [select_concepts(run, args.n_frequent, args.region_tile_size) for run in runs]
    for run, picks in zip(runs, selections, strict=True):
        log.info(f"{run.name}: selected concepts " + ", ".join(f"{c} ({label})" for c, label in picks))

    n_cols = max(args.top_k, 1)
    render_comparison(runs, dataset, selections, grid_h, grid_w, args.top_k, args.crop_margin, n_cols, Path(args.out))


if __name__ == "__main__":
    main()
