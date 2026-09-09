"""S2AE exclusive sparsity loss (arXiv:2607.08605, Eq. 10).

    L_es = (1/N) * sum_j ( sum_g |s^g_j| )^2

For each latent dimension ``j``, sum its group-level activation profile
(`s^g_j`, see `group_sparsity.py`) across all groups `g`, then square. This
penalizes a single dictionary feature from being active across *many*
different patch groups -- i.e. it pushes each feature toward specializing in
one group (spatial region) rather than firing everywhere.

Like `GroupSparsityLoss`, `s^g` is binarized through a straight-through
estimator (`ste.py::ste_binarize`, paper Sec. 4.2) before this sum: `sum_g` is
"in how many groups is latent `j` active at all", not a magnitude-weighted
sum -- see `group_sparsity.py`'s docstring for why (the shrinkage-bias
failure mode this avoids).

See `GroupSparsityLoss` for what `grouping="tile"` vs `"attention"` vs
`"feature"` selects.
"""

from __future__ import annotations

from torch import Tensor

from cbm_msae_lab.attention_grouping import onehot_l2_aggregate
from cbm_msae_lab.losses.base import Loss, LossContext
from cbm_msae_lab.losses.ste import ste_binarize
from cbm_msae_lab.losses.tiling import tile_l2_norm


class ExclusivityLoss(Loss):
    def __init__(self, weight: float, grouping: str = "tile", tile_size: int = 2, n_clusters: int = 20) -> None:
        super().__init__(weight)
        if grouping not in ("tile", "attention", "feature"):
            raise ValueError(f"unknown grouping {grouping!r}; expected 'tile', 'attention', or 'feature'")
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
                    f"exclusivity.grouping={self.grouping!r} but no group_labels were supplied to this "
                    f"training step -- build a cache with scripts/extract_attention_groups.py --method "
                    f"{self.grouping} first."
                )
            s_g = onehot_l2_aggregate(ctx.f_img, ctx.group_labels, self.n_clusters)  # [B, n_clusters, dict_size]

        s_g = ste_binarize(s_g)  # forward: {0, 1} gate; backward: saturating (tanh) gradient into s_g
        per_latent = s_g.sum(dim=1)  # sum_g |s^g_j| -> [B, dict_size] -- already >= 0, no .abs() needed
        return per_latent.pow(2).mean()  # mean over both latents and batch
