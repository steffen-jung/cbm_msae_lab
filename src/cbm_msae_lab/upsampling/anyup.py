"""Image-guided universal feature upsampler.

Wraps AnyUp (Wimmer et al., "AnyUp: Universal Feature Upsampling", arXiv:2510.12764,
ICLR 2026), loaded via ``torch.hub`` from https://github.com/wimmerth/anyup.
Unlike earlier learned upsamplers (e.g. FeatUp), AnyUp is trained once and
generalizes at inference time to features from any backbone/channel-width and
any target resolution -- exactly what's needed here, since the same
upsampler must work whether the SAE is fed CLIP-DINOiser or DINOv3 features.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cbm_msae_lab.upsampling.base import Upsampler

# AnyUp's guidance image must be ImageNet-normalized (see the AnyUp README /
# the same convention already used in CFM's own AnyUp call site).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class AnyUpUpsampler(Upsampler):
    def __init__(
        self,
        target_resolution: int = 64,
        stage: str = "features",
        torch_hub_repo: str = "wimmerth/anyup",
        torch_hub_model: str = "anyup_multi_backbone",
        use_natten: bool = False,
    ) -> None:
        super().__init__()
        self.target_resolution = target_resolution
        self.stage = stage
        self.model = torch.hub.load(torch_hub_repo, torch_hub_model, use_natten=use_natten)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def _normalize(self, images: Tensor) -> Tensor:
        mean = images.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        return (images - mean) / std

    def forward(self, images: Tensor, features: Tensor) -> Tensor:
        """images: [B, 3, H_img, W_img] in [0, 1], features: [B, C, h, w] -> [B, C, target_resolution, target_resolution]."""
        guidance = self._normalize(images)
        with torch.no_grad():
            return self.model(
                guidance,
                features,
                output_size=(self.target_resolution, self.target_resolution),
            )
