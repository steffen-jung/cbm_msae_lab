"""A cheap, non-learned upsampling baseline (no image guidance)."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from cbm_msae_lab.upsampling.base import Upsampler


class BilinearUpsampler(Upsampler):
    def __init__(self, target_resolution: int = 64, stage: str = "features") -> None:
        super().__init__()
        self.target_resolution = target_resolution
        self.stage = stage

    def forward(self, images: Tensor, features: Tensor) -> Tensor:
        """features: [B, C, h, w] -> [B, C, target_resolution, target_resolution]. `images` is unused."""
        return F.interpolate(
            features,
            size=(self.target_resolution, self.target_resolution),
            mode="bilinear",
            align_corners=False,
        )
