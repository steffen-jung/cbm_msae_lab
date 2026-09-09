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
