"""The trainable head between a frozen encoder and the SAE.

This is what the user-facing "1x1 vs 3x3 kernel" choice actually configures.
It is deliberately a *separate*, newly-initialized module -- never part of a
frozen pretrained checkpoint -- so its kernel size is a free experimental
choice, trained jointly with the SAE (both start randomly initialized on top
of a frozen encoder). This is unlike CLIP-DINOiser's own internal `obj_proj`
conv (see `encoders/clip_dinoiser_backend/clip_dinoiser.py`), whose kernel
size is fixed by its pretrained checkpoint and is not user-configurable.
"""

from __future__ import annotations

from torch import Tensor, nn


class EncoderProjection(nn.Module):
    """A single Conv2d mapping encoder features to the SAE's input dimension.

    With ``kernel_size=1``, this is mathematically identical to applying an
    ``nn.Linear(in_channels, out_channels)`` independently to every spatial
    position (a 1x1 convolution IS a per-pixel linear map: the same weight
    matrix multiplies every spatial location, with no neighbour mixing).
    ``kernel_size=3`` (or any larger odd size) instead lets every output
    position see its spatial neighbourhood before the SAE encodes it.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd so 'same' padding is exact, got {kernel_size}")
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="replicate",
        )

    def forward(self, features: Tensor) -> Tensor:
        """features: [B, C_backbone, H, W] -> [B, C_sae, H, W] (H, W unchanged: 'same' padding)."""
        return self.conv(features)
