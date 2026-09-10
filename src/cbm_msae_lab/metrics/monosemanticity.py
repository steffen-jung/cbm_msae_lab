"""Monosemanticity Score (MS): Pach et al., NeurIPS 2025, arXiv:2504.02821, Eq. 7-9.

For each concept (dictionary feature) k, MS is an activation-weighted mean
pairwise similarity of the stimuli it fires on, measured in the space of an
*external* encoder (not the SAE itself): a monosemantic concept should fire
strongly on images that are also visually/semantically similar to each other.

    tilde_a^k_n = minmax(a^k_n)                          (Eq. 7)
    r^k_{nm}    = tilde_a^k_n * tilde_a^k_m               (activation-weight of the pair)
    s_nm        = cos(E(x_n), E(x_m))                     (external-encoder similarity)
    MS^k        = sum_{n<m} r^k_nm * s_nm / sum_{n<m} r^k_nm    (Eq. 9)

Implemented as a closed-form O(N*D) computation (never forming the N x N
similarity matrix), following the identity
    sum_{n!=m} w_n w_m s_nm = || sum_n w_n e_n ||^2 - sum_n w_n^2
for L2-normalized embedding rows `e_n`, which is exact and is what makes this
tractable for a full validation set. Ported from
`cfm_finegrained/src/cfm_fg/polysemy.py::monosemanticity_score`.

`PachMonosemanticityScore` below is a second metric, not a variant of this one: it is a literal
transcription of the *official* reference implementation the authors published alongside the
paper (`ExplainableML/sae-for-vlm/metric.py::main()`), which does not match the Eq. 9 text above.
That code accumulates `weighted_cosine_similarity_sum` and `weight_sum = sum_{n!=m} r^k_nm`
per neuron and divides one by the other -- i.e. it reports the *weighted average* pairwise
similarity, not a fixed-denominator sum. See `PachMonosemanticityScore`'s own docstring for what
that changes.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric


class MonosemanticityScore(Metric):
    """Accumulates (per-image concept activations, external-encoder embeddings) across
    a validation pass; `compute()` returns one MS value per dictionary feature: [dict_size]."""

    full_state_update = False

    def __init__(self, dict_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dict_size = dict_size
        self.add_state("activations", default=[], dist_reduce_fx="cat")
        self.add_state("embeddings", default=[], dist_reduce_fx="cat")

    def update(self, activations: Tensor, embeddings: Tensor) -> None:
        """activations: [B, dict_size] (one activation strength per image per concept, e.g.
        spatially max-pooled SAE latents). embeddings: [B, embed_dim], L2-normalized rows, from
        a genuinely external encoder -- NOT the one the SAE is trained to reconstruct (see
        `external_encoder.py`, which pairs CLIP-DINOiser <-> DINOv3)."""
        self.activations.append(activations.detach())
        self.embeddings.append(embeddings.detach())

    def compute(self) -> Tensor:
        acts = torch.cat(self.activations, dim=0)  # [N, dict_size]
        emb = torch.cat(self.embeddings, dim=0)  # [N, embed_dim]
        N, K = acts.shape
        if N < 2:
            return torch.zeros(K, device=acts.device)

        lo = acts.min(dim=0, keepdim=True).values  # [1, K]
        hi = acts.max(dim=0, keepdim=True).values  # [1, K]
        span = hi - lo
        w = torch.where(span > 0, (acts - lo) / span.clamp_min(1e-12), torch.ones_like(acts))  # [N, K]

        pooled = w.t() @ emb  # [K, embed_dim]
        total = (pooled * pooled).sum(dim=1) - (w * w).sum(dim=0)  # [K]
        return total / (N * (N - 1))


class PachMonosemanticityScore(Metric):
    """Literal port of `ExplainableML/sae-for-vlm/metric.py::main()`'s monosemanticity
    computation, the code the paper's authors actually ran -- not a transcription of the Eq. 9
    text (see `MonosemanticityScore` for that one and for why it's the metric used elsewhere in
    this codebase). Same Eq. 7/8 weights and per-pair similarity as `MonosemanticityScore`, but
    `compute()` divides by `sum_{n!=m} r^k_nm` instead of by `N(N-1)`: the weighted-average
    pairwise similarity a concept's active stimuli have with each other, undiluted by how many
    of the N stimuli it actually fires on. That makes a concept active on 3 tightly-clustered
    stimuli score the same as one active on 300 equally coherent stimuli, where
    `MonosemanticityScore` would score the sparser concept far lower purely for covering fewer
    of the N(N-1) pairs -- neither notion is wrong, they answer different questions ("how
    coherent are this concept's stimuli" vs. "how much of the evaluation set does this concept
    explain coherently"). Accumulates identically to `MonosemanticityScore`; `compute()` returns
    one value per dictionary feature: [dict_size], `nan` where the official code's
    `weight_sum == 0` guard would fire (a constant concept, or one active on a single stimulus).
    Aggregate with `.nanmean()`, never `.mean()`."""

    full_state_update = False

    def __init__(self, dict_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dict_size = dict_size
        self.add_state("activations", default=[], dist_reduce_fx="cat")
        self.add_state("embeddings", default=[], dist_reduce_fx="cat")

    def update(self, activations: Tensor, embeddings: Tensor) -> None:
        """See `MonosemanticityScore.update` -- identical inputs and semantics."""
        self.activations.append(activations.detach())
        self.embeddings.append(embeddings.detach())

    def dead_fraction(self) -> float:
        """Share of concepts `compute()` returns `nan` for. See `MonosemanticityScore.dead_fraction`;
        differs only in also counting single-active-stimulus concepts (`weight_sum == 0` there too)."""
        acts = torch.cat(self.activations, dim=0)  # [N, dict_size]
        span = acts.max(dim=0).values - acts.min(dim=0).values  # [dict_size]
        return float((span <= 0).float().mean())

    def compute(self) -> Tensor:
        acts = torch.cat(self.activations, dim=0)  # [N, dict_size]
        emb = torch.cat(self.embeddings, dim=0)  # [N, embed_dim]
        N, K = acts.shape
        if N < 2:
            return torch.zeros(K, device=acts.device)

        lo = acts.min(dim=0, keepdim=True).values  # [1, K]
        hi = acts.max(dim=0, keepdim=True).values  # [1, K]
        span = hi - lo  # [1, K]
        w = (acts - lo) / span.clamp_min(1e-12)  # [N, K]; the constant columns are all-zero here

        pooled = w.t() @ emb  # [K, embed_dim]
        w_sum = w.sum(dim=0)  # [K]
        w_sq_sum = (w * w).sum(dim=0)  # [K]
        total = (pooled * pooled).sum(dim=1) - w_sq_sum  # sum_{n!=m} w_n w_m s_nm
        weight_sum = w_sum * w_sum - w_sq_sum  # sum_{n!=m} w_n w_m

        ms = total / weight_sum.clamp_min(1e-12)
        dead = (span.squeeze(0) <= 0) | (weight_sum <= 0)
        return ms.masked_fill(dead, float("nan"))
