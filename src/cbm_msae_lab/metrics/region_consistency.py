"""Region consistency: how spatially specialised are the learned features?

Reports the participation ratio (`participation_ratio.py`) as a *measurement*
rather than an objective, so the quantity the PR loss optimises can also be read
off runs where that loss is switched off -- which is the point of the ablation:
Group-TopK is expected to move this number too, and the baseline needs a value
on the same scale to be compared against.

Two scalars:

    region_pr             mean effective number of regions per active feature.
                          Lower = more spatially specialised. Floor is 1.
    region_dominant_share mean fraction of a feature's activation mass that sits
                          in its single largest region. Upper bound 1. Reported
                          alongside PR because the two disagree in an informative
                          way: a feature with 0.9 in one region and a thin tail
                          over ten others has a high dominant share but a PR well
                          above 1.

Both are averaged per image first (regions are per-image, especially under the
attention/feature clusterings) and then over images.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric

from cbm_msae_lab.participation_ratio import mean_over_active, participation_ratio


class RegionConsistencyMetric(Metric):
    """Accumulates per-image (latents, group labels) and reduces to the two scalars above."""

    full_state_update = False
    higher_is_better = False

    def __init__(self, n_groups: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_groups = n_groups
        self.add_state("pr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("share_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("images", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, latents_img: Tensor, group_labels: Tensor) -> None:
        """latents_img: [B, P, dict_size] >= 0, group_labels: [B, P] int64 in [0, n_groups)."""
        result = participation_ratio(latents_img, group_labels, self.n_groups)
        # Per-image means so that an image where few features fire does not
        # outweigh one where many do.
        for b in range(latents_img.shape[0]):
            active = result.active[b : b + 1]
            if not active.any():
                continue
            self.pr_sum += mean_over_active(result.pr[b : b + 1], active)
            self.share_sum += mean_over_active(result.dominant_share[b : b + 1], active)
            self.images += 1

    def compute(self) -> dict[str, Tensor]:
        if self.images == 0:
            zero = torch.zeros((), device=self.pr_sum.device)
            return {"region_pr": zero, "region_dominant_share": zero}
        return {
            "region_pr": self.pr_sum / self.images,
            "region_dominant_share": self.share_sum / self.images,
        }
