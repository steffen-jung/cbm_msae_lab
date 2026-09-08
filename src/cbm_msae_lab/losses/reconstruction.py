"""CFM's original nested per-Matryoshka-group L2 reconstruction loss.

Ported from ``dictionary_learning.trainers.matryoshka_batch_top_k.MatryoshkaBatchTopKTrainer.loss``.
Reconstructs progressively from each Matryoshka prefix group and averages the
per-prefix L2 losses, so the *first* group alone must already reconstruct
reasonably well, the first two groups together better, and so on -- this is
what gives the dictionary "nested"/coarse-to-fine structure.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cbm_msae_lab.losses.base import Loss, LossContext


class ReconstructionLoss(Loss):
    def compute(self, ctx: LossContext) -> Tensor:
        sae = ctx.sae
        x = ctx.x  # [N, activation_dim]
        f = ctx.f  # [N, dict_size]

        num_groups = sae.active_groups
        group_weight = 1.0 / num_groups  # uniform, matching dictionary_learning's default

        x_reconstruct = torch.zeros_like(x) + sae.b_dec  # [N, activation_dim]
        W_dec_chunks = torch.split(sae.W_dec, sae.group_sizes.tolist(), dim=0)
        f_chunks = torch.split(f, sae.group_sizes.tolist(), dim=1)

        per_group_losses = []
        for i in range(num_groups):
            x_reconstruct = x_reconstruct + f_chunks[i] @ W_dec_chunks[i]
            l2 = (x - x_reconstruct).pow(2).sum(dim=-1).mean() * group_weight
            per_group_losses.append(l2)

        # NOTE: matches dictionary_learning's MatryoshkaBatchTopKTrainer.loss()
        # exactly, including its double 1/num_groups weighting: each per-group
        # term already carries one factor of `group_weight` above, and this
        # final `.mean()` (not `.sum()`) applies a second one. Kept as-is
        # rather than "fixed", since it's the reference behaviour the
        # published CFM checkpoint was trained under.
        return torch.stack(per_group_losses).mean()
