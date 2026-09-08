"""Tversky Monosemanticity Score (TMS): Filus & Pokuciński, arXiv:2607.17770, Eq. 1.

Unlike `MonosemanticityScore`, TMS needs no external encoder and no labels --
it only looks at the SAE's own latent activation patterns, trading semantic
grounding for being fully self-contained. For each latent k:

  1. Binarize every stimulus's activation of k against k's own mean activation
     across the evaluation set: `B[n, k] = 1[Z[n, k] > mean_k]`.
  2. Let `A_k = {n : B[n, k] = 1}` be k's "active set".
  3. For random pairs (n_i, n_j) drawn from `A_k`, compute the Tversky index
     between their full binarized activation vectors `B[n_i, :]`, `B[n_j, :]`
     (treated as sets of "which latents are on"):

        TI_{a,b}(B_i, B_j) = |B_i n B_j| / (|B_i n B_j| + a|B_i \\ B_j| + b|B_j \\ B_i|)

     The paper's parameter-free default `a = b = 1` reduces this to the
     Jaccard index.
  4. `s_k = mean` of that index over the sampled pairs; the model-level TMS is
     the mean of `s_k` over every latent with a large enough active set.

Ported from `cfm_finegrained/src/cfm_fg/polysemy.py` (`pack_bitset`,
`bitset_column`, `tversky_index`, `tversky_ms`), which bit-packs each
stimulus's binarized activation vector into `uint64` words so that
intersection/difference become a single `AND`/`ANDNOT` + popcount instead of
an O(dict_size) Python-level comparison per pair -- necessary since this
metric compares whole activation vectors across potentially many sampled
pairs per latent, for every latent in a (typically large) dictionary. Runs on
CPU via NumPy since it's only ever invoked on the periodic "expensive eval"
cadence, not every training step.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torchmetrics import Metric


def _pack_bitset(binary: NDArray[np.bool_]) -> NDArray[np.uint64]:
    """[N, K] booleans -> [N, ceil(K/64)] uint64, bit j of word w = latent 64*w+j."""
    n, k = binary.shape
    words = (k + 63) // 64
    padded = np.zeros((n, words * 64), dtype=np.bool_)
    padded[:, :k] = binary
    bits = padded.reshape(n, words, 64)
    weights = np.uint64(1) << np.arange(64, dtype=np.uint64)
    return (bits.astype(np.uint64) * weights).sum(axis=2, dtype=np.uint64)


def _bitset_column(packed: NDArray[np.uint64], k: int) -> NDArray[np.int64]:
    """Rows (stimuli) in which latent k is active: this is A_k."""
    word, bit = divmod(k, 64)
    mask = np.uint64(1) << np.uint64(bit)
    return np.flatnonzero(packed[:, word] & mask).astype(np.int64)


def _tversky_index(
    packed: NDArray[np.uint64],
    i: NDArray[np.int64],
    j: NDArray[np.int64],
    alpha: float,
    beta: float,
) -> NDArray[np.float64]:
    a = packed[i]
    b = packed[j]
    inter = np.bitwise_count(a & b).sum(axis=1).astype(np.float64)
    only_a = np.bitwise_count(a & ~b).sum(axis=1).astype(np.float64)
    only_b = np.bitwise_count(b & ~a).sum(axis=1).astype(np.float64)
    denom = inter + alpha * only_a + beta * only_b
    return np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)


def _tversky_ms_per_latent(
    packed: NDArray[np.uint64],
    n_latents: int,
    max_pairs: int,
    alpha: float,
    beta: float,
    seed: int,
    min_active: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Returns (per-latent TMS [n_latents], active-set size |A_k| [n_latents])."""
    rng = np.random.default_rng(seed)
    scores = np.zeros(n_latents, dtype=np.float32)
    sizes = np.zeros(n_latents, dtype=np.int64)
    for k in range(n_latents):
        rows = _bitset_column(packed, k)
        sizes[k] = rows.size
        if rows.size < min_active:
            continue
        i = rng.integers(0, rows.size, size=max_pairs)
        j = rng.integers(0, rows.size, size=max_pairs)
        keep = i != j
        if not keep.any():
            continue
        scores[k] = float(_tversky_index(packed, rows[i[keep]], rows[j[keep]], alpha, beta).mean())
    return scores, sizes


class TverskyMS(Metric):
    """Accumulates raw per-stimulus latent activations across a validation pass;
    `compute()` returns a single scalar: the mean per-latent TMS over latents
    with a large enough active set."""

    full_state_update = False

    def __init__(
        self,
        dict_size: int,
        max_pairs: int = 1000,
        alpha: float = 1.0,
        beta: float = 1.0,
        seed: int = 0,
        min_active: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dict_size = dict_size
        self.max_pairs = max_pairs
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.min_active = min_active
        self.add_state("activations", default=[], dist_reduce_fx="cat")

    def update(self, activations: Tensor) -> None:
        """activations: [B, dict_size], any real-valued per-stimulus latent activations."""
        self.activations.append(activations.detach())

    def compute(self) -> Tensor:
        acts = torch.cat(self.activations, dim=0).cpu().numpy()  # [N, dict_size]
        mean = acts.mean(axis=0, keepdims=True)
        binary = acts > mean  # [N, dict_size]
        packed = _pack_bitset(binary)

        scores, sizes = _tversky_ms_per_latent(
            packed,
            acts.shape[1],
            self.max_pairs,
            self.alpha,
            self.beta,
            self.seed,
            self.min_active,
        )
        valid = sizes >= self.min_active
        if not valid.any():
            return torch.zeros(())
        return torch.tensor(float(scores[valid].mean()))
