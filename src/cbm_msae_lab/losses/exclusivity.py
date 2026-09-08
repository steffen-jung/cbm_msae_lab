"""S2AE exclusive sparsity loss (arXiv:2607.08605, Eq. 10).

    L_es = (1/N) * sum_j ( sum_g |s^g_j| )^2

For each latent dimension ``j``, sum its group-level activation profile
(`s^g_j`, see `group_sparsity.py`) across all groups `g`, then square. This
penalizes a single dictionary feature from being active across *many*
different patch groups -- i.e. it pushes each feature toward specializing in
one group (spatial region) rather than firing everywhere.

See `GroupSparsityLoss` for what `grouping="tile"` vs `"attention"` selects.
"""

from __future__ import annotations

from torch import Tensor

from cbm_msae_lab.attention_grouping import onehot_l2_aggregate
from cbm_msae_lab.losses.base import Loss, LossContext
from cbm_msae_lab.losses.tiling import tile_l2_norm


class ExclusivityLoss(Loss):
    def __init__(self, weight: float, grouping: str = "tile", tile_size: int = 2, n_clusters: int = 20) -> None:
        super().__init__(weight)
        if grouping not in ("tile", "attention"):
            raise ValueError(f"unknown grouping {grouping!r}; expected 'tile' or 'attention'")
        self.grouping = grouping
        self.tile_size = tile_size
        self.n_clusters = n_clusters

    def compute(self, ctx: LossContext) -> Tensor:
        if self.grouping == "tile":
            grid_h, grid_w = ctx.grid
            s_g = tile_l2_norm(ctx.f_img, grid_h, grid_w, self.tile_size)  # [B, G, dict_size]
        else:
            if ctx.group_labels is None:
                raise RuntimeError(
                    "exclusivity.grouping='attention' but no group_labels were supplied to this training "
                    "step -- build a cache with scripts/extract_attention_groups.py first."
                )
            s_g = onehot_l2_aggregate(ctx.f_img, ctx.group_labels, self.n_clusters)  # [B, n_clusters, dict_size]

        per_latent = s_g.abs().sum(dim=1)  # sum_g |s^g_j| -> [B, dict_size]
        return per_latent.pow(2).mean()  # mean over both latents and batch
