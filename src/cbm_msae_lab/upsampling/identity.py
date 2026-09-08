"""No-op upsampler: keeps the encoder's native spatial resolution."""

from __future__ import annotations

from torch import Tensor

from cbm_msae_lab.upsampling.base import Upsampler


class IdentityUpsampler(Upsampler):
    def __init__(self) -> None:
        super().__init__()
        self.stage = "none"

    def forward(self, images: Tensor, features: Tensor) -> Tensor:
        """features: [B, C, h, w] -> [B, C, h, w] (unchanged)."""
        return features
