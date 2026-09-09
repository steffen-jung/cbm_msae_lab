"""S2AE group sparsity loss (arXiv:2607.08605, "When Structured Sparse Autoencoders
Learn Consistent Concepts Across Modalities", Eq. 9).

    L_gs = (1/G) * sum_g || s^g ||_1

where ``s^g in R^N`` (N = dict_size) is the group-level activation profile
for patch group ``g``: for every latent dimension, the L2 norm of that
latent's activations over the patches belonging to group ``g``. Intuitively,
this pushes each *group* of patches (not each individual patch) toward using
few dictionary features in total.

Three ways to form the groups, selected by ``grouping``:
- ``"tile"`` (default): simple s-by-s spatial tiles of the regular ViT patch
  grid (`tiling.py::tile_l2_norm`), identical for every image.
- ``"attention"``: S2AE's own approach -- patches are clustered per-image by
  combining the encoder's self-attention with spatial proximity
  (`attention_grouping.py::cluster_patches`). Needs `ctx.group_labels` to be
  populated (i.e. a group-label cache built via
  `scripts/extract_attention_groups.py --method attention`).
- ``"feature"``: like ``"attention"``, but clusters by the encoder's raw
  per-patch feature similarity instead of attention
  (`attention_grouping.py::cluster_patches_by_features`). Same cache
  mechanism, built with `--method feature`.
"""

from __future__ import annotations

from torch import Tensor

from cbm_msae_lab.attention_grouping import onehot_l2_aggregate
from cbm_msae_lab.losses.base import Loss, LossContext
from cbm_msae_lab.losses.tiling import tile_l2_norm


class GroupSparsityLoss(Loss):
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
                    f"group_sparsity.grouping={self.grouping!r} but no group_labels were supplied to this "
                    f"training step -- build a cache with scripts/extract_attention_groups.py --method "
                    f"{self.grouping} first."
                )
            s_g = onehot_l2_aggregate(ctx.f_img, ctx.group_labels, self.n_clusters)  # [B, n_clusters, dict_size]

        l1_per_group = s_g.abs().sum(dim=-1)  # ||s^g||_1 per (image, group) -> [B, G]
        return l1_per_group.mean()  # mean over both groups and batch
