"""Abstract interface for spatial upsamplers.

An ``Upsampler`` can sit at one of two points in ``ConceptPipeline`` (picked
by ``stage``): ``"features"`` upsamples the projected encoder features before
the SAE ever sees them (so the SAE itself operates at the higher resolution,
and the reconstruction loss trains against that higher-resolution target);
``"latents"`` instead upsamples the SAE's already-encoded concept activations
purely for producing higher-resolution concept maps for inspection -- it
never affects the reconstruction loss. ``stage="none"`` (``IdentityUpsampler``)
disables upsampling entirely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Upsampler(nn.Module, ABC):
    #: "none" | "features" | "latents" -- which point in the pipeline this instance targets.
    stage: str

    @abstractmethod
    def forward(self, images: Tensor, features: Tensor) -> Tensor:
        """images: [B, 3, H_img, W_img] (guidance), features: [B, C, h, w] -> [B, C, H_t, W_t]."""
        raise NotImplementedError
