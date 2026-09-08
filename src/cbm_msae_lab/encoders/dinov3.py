"""``Encoder`` wrapper around DINOv3, loaded through ``timm``.

Uses ``timm.create_model(..., pretrained=True)`` (matching the pattern already
used in ``master-thesis/train_dinov3.py``) rather than Meta's official
``torch.hub`` path, since the latter requires manual approval and downloading
gated checkpoint URLs, while ``timm``/HuggingFace serves the same weights
without any gating.
"""

from __future__ import annotations

from typing import Any

import timm
import torch
import torchvision.transforms as T
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from cbm_msae_lab.encoders.base import Encoder, EncoderOutput

# DINOv3 (like DINOv2) was trained with standard ImageNet normalization.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv3Encoder(Encoder):
    """Frozen DINOv3 ViT; returns the dense patch-token grid (register/CLS tokens dropped)."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        output_dim: int = 768,
        image_size: int = 224,
        patch_size: int = 16,
        extract_attention: bool = False,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.extract_attention = extract_attention
        self.normalize = T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        actual_dim = self.model.embed_dim
        if actual_dim != output_dim:
            raise ValueError(
                f"'{model_name}' has embed_dim={actual_dim}, but output_dim={output_dim} was "
                "configured -- fix DINOv3EncoderConfig.output_dim to match the chosen model_name."
            )
        # 1 CLS token + N register tokens are prepended before the patch tokens;
        # timm exposes their combined count so we know how many to drop below.
        self.num_prefix_tokens: int = self.model.num_prefix_tokens

        if self.extract_attention:
            # `EvaAttention.forward` (timm/models/eva.py) only computes an
            # explicit, hookable attention matrix on its *eager* path -- the
            # default `fused_attn=True` path calls `F.scaled_dot_product_attention`,
            # a fused kernel that never materializes the attention weights.
            # Forcing every block onto the eager path is a one-time, harmless
            # change (same math, just not fused) that makes the attention
            # matrix observable via a hook on `attn_drop` (a plain `nn.Dropout`
            # that receives the post-softmax weights as its input).
            for block in self.model.blocks:
                block.attn.fused_attn = False

        self.freeze()

    def _attention_hooks(self, captured: list[Tensor]) -> list[RemovableHandle]:
        def make_hook():
            def hook(module: Any, args: Any, output: Tensor) -> None:
                captured.append(args[0].detach())  # [B, num_heads, N, N], N = num_prefix_tokens + h*w

            return hook

        return [block.attn.attn_drop.register_forward_hook(make_hook()) for block in self.model.blocks]

    @torch.no_grad()
    def forward(self, images: Tensor) -> EncoderOutput:
        """images: [B, 3, H, W] in [0, 1] -> features: [B, output_dim, H/patch_size, W/patch_size]."""
        B, _, H, W = images.shape
        x = self.normalize(images)

        captured_attn: list[Tensor] = []
        handles = self._attention_hooks(captured_attn) if self.extract_attention else []
        try:
            tokens = self.model.forward_features(x)  # [B, num_prefix_tokens + h*w, output_dim]
        finally:
            for handle in handles:
                handle.remove()

        patch_tokens = tokens[:, self.num_prefix_tokens :, :]  # [B, h*w, output_dim]
        h, w = H // self.patch_size, W // self.patch_size
        features = patch_tokens.permute(0, 2, 1).reshape(B, self.output_dim, h, w)  # [B, output_dim, h, w]

        attention = None
        if self.extract_attention:
            # One captured tensor per block: [B, num_heads, N, N]. Average over
            # blocks (dim 0 of the stack) and heads, then drop prefix tokens.
            attention = torch.stack(captured_attn, dim=0).mean(dim=0).mean(dim=1)  # [B, N, N]
            attention = attention[:, self.num_prefix_tokens :, self.num_prefix_tokens :]  # [B, h*w, h*w]

        return EncoderOutput(features=features, attention=attention)
