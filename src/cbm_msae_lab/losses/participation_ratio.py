"""Participation-ratio loss: push each dictionary feature toward firing in few regions.

    L_PR^(l) = mean over active features j in block B_l of PR_j
    L_PR     = sum_l gamma_l * L_PR^(l)

PR_j is the effective number of groups feature j spreads its activation mass
over (see `participation_ratio.py` for the definition and for why it is written
without an epsilon). Minimising it concentrates each feature's mass into fewer
regions. The floor is 1 -- a feature living entirely in one region -- so the loss
is bounded below by 1 whenever anything is active, not by 0.

What distinguishes this from `exclusivity.py`, which chases the same goal: PR is
homogeneous of degree 0, so it produces no shrinkage gradient and needs no
straight-through binarisation to avoid one. The S2AE exclusivity loss reaches for
`ste_binarize` precisely because its raw L2 form would otherwise pay for
structure by making every activation smaller.

Computed per Matryoshka level, since the levels are nested dictionaries with
different capacities and a shared mean would let the large last block dominate.
"""

from __future__ import annotations

from torch import Tensor

from cbm_msae_lab.grouping import resolve_group_labels, validate_grouping
from cbm_msae_lab.losses.base import Loss, LossContext
from cbm_msae_lab.participation_ratio import mean_over_active, participation_ratio


class ParticipationRatioLoss(Loss):
    def __init__(
        self,
        weight: float,
        grouping: str = "tile",
        tile_size: int = 2,
        n_clusters: int = 20,
        level_weights: list[float] | None = None,
    ) -> None:
        super().__init__(weight)
        self.grouping = validate_grouping(grouping, "participation_ratio")
        self.tile_size = tile_size
        self.n_clusters = n_clusters
        self.level_weights = level_weights

    def _gammas(self, n_levels: int) -> list[float]:
        if self.level_weights is None:
            return [1.0 / n_levels] * n_levels
        if len(self.level_weights) != n_levels:
            raise ValueError(
                f"participation_ratio.level_weights has {len(self.level_weights)} entries but the SAE has "
                f"{n_levels} active Matryoshka levels"
            )
        return list(self.level_weights)

    def compute(self, ctx: LossContext) -> Tensor:
        z_img = ctx.f_img  # [B, P, dict_size] -- post-BatchTopK (and post-Group-TopK, if enabled)
        labels, n_groups = resolve_group_labels(
            self.grouping,
            grid=ctx.grid,
            tile_size=self.tile_size,
            batch_size=z_img.shape[0],
            device=z_img.device,
            group_labels=ctx.group_labels,
            n_clusters=self.n_clusters,
            consumer="participation_ratio",
        )

        blocks = ctx.sae.matryoshka_blocks
        total = z_img.new_zeros(())
        for gamma, (start, end) in zip(self._gammas(len(blocks)), blocks, strict=True):
            result = participation_ratio(z_img[:, :, start:end], labels, n_groups)
            total = total + gamma * mean_over_active(result.pr, result.active)
        return total
