"""Abstract interface every encoder backbone implements.

An ``Encoder`` is always frozen (no gradients ever flow into it): it maps raw
images to a dense spatial feature grid, which the SAE consumes directly.
Everything downstream (the optional upsampler, the SAE) only depends on this
interface, so swapping CLIP-DINOiser for DINOv3 never requires touching any
other module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class EncoderOutput:
    """The result of one encoder forward pass.

    features: [B, C_backbone, H0, W0] -- dense per-patch features, channel-first
        so they compose directly with `nn.Conv2d`-based downstream modules.
    attention: [B, P, P] or None -- patch-to-patch self-attention (P = H0*W0),
        averaged over every attention head and every transformer block, with
        CLS/register tokens already dropped. Only populated when the encoder
        was constructed with `extract_attention=True` (default False, since
        this needs extra hooks and adds a little compute even when unused
        downstream) -- see `attention_grouping.py` for what it's for.
    """

    features: Tensor
    attention: Tensor | None = None


class Encoder(nn.Module, ABC):
    """Base class for frozen vision backbones used as the SAE's input features."""

    #: Native channel dimension of `features` (set by subclasses); read by
    #: `SAEConfig.activation_dim`'s `${encoder.output_dim}` interpolation.
    output_dim: int

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, images: Tensor) -> EncoderOutput:
        """images: [B, 3, H, W], values in [0, 1] (encoder-specific normalization is applied internally)."""
        raise NotImplementedError

    def freeze(self) -> None:
        """Disables gradients on every parameter and switches to eval mode.

        Every concrete `Encoder` calls this at the end of its own `__init__`,
        since none of them are ever meant to be fine-tuned in this codebase.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True) -> Encoder:
        # A frozen encoder must never leave eval mode (e.g. BatchNorm/Dropout
        # inside a backbone would otherwise start updating running stats the
        # moment the surrounding ConceptPipeline.train() is called).
        return super().train(False)
