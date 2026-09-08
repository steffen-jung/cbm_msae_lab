"""Thin ``Encoder`` wrapper around the vendored, frozen CLIP-DINOiser backend."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from cbm_msae_lab.encoders.base import Encoder, EncoderOutput
from cbm_msae_lab.encoders.clip_dinoiser_backend.builder import build_model
from cbm_msae_lab.encoders.clip_dinoiser_backend.clip_dinoiser import CLIP_DINOiser

# CFM itself never uses the segmentation-head class predictions when just
# extracting dense features -- it passes a single placeholder class name so
# `MaskClipHead`'s constructor (which always builds a class-text embedding,
# see `maskclip.py`) has something valid to embed. We do the same.
_PLACEHOLDER_CLASS_NAMES = ["dummy"]


class ClipDinoiserEncoder(Encoder):
    """ "DINOised CLIP": CLIP-DINOiser (Wysoczanska & Simeoni et al., ECCV 2024).

    Loads the pretrained checkpoint (default: the laion2b-CLIP variant shipped
    with CFM, copied into this repo by ``scripts/download_checkpoints.py``)
    and freezes every parameter. ``forward`` returns the pooled, DINOised
    dense CLIP features at the backbone's native patch-grid resolution.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/clip_dinoiser/laion2b.pt",
        clip_backbone: str = "maskclip",
        vit_arch: str = "vit_base",
        vit_patch_size: int = 16,
        feats_idx: int = -3,
        enc_type_feats: str = "v",
        gamma: float = 0.2,
        delta: float = 0.99,
        output_dim: int = 512,
        image_size: int = 224,
        extract_attention: bool = False,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size
        self.extract_attention = extract_attention

        maskclip_cfg = {
            "_target_": "cbm_msae_lab.encoders.clip_dinoiser_backend.clip_dinoiser.CLIP_DINOiser",
            "clip_backbone": clip_backbone,
            "vit_arch": vit_arch,
            "vit_patch_size": vit_patch_size,
            "enc_type_feats": enc_type_feats,
            "gamma": gamma,
            "delta": delta,
            "feats_idx": feats_idx,
            "in_dim": 256,  # internal obj_proj/correlation width; fixed by the pretrained checkpoint
        }
        self.model: CLIP_DINOiser = build_model(maskclip_cfg, class_names=_PLACEHOLDER_CLASS_NAMES)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # strict=False: `class_embeddings` (a buffer, not a learned weight) has
        # a shape depending on `_PLACEHOLDER_CLASS_NAMES`, which will not match
        # whatever class list the checkpoint's own MaskClipHead was built with.
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        self.freeze()

    def _attention_hooks(self, captured: list[Tensor]) -> list[RemovableHandle]:
        """Registers a forward-pre-hook on every CLIP ViT transformer block that
        recomputes just enough of that block's attention (q, k -- not v, not the
        rest of the block) to get its post-softmax attention matrix, without
        touching or duplicating any of the vendored `MaskClip`/`CLIP_DINOiser`
        code: it only reads already-public submodules
        (`resblocks[i]`, `.ln_1`, `.attn.in_proj_weight/in_proj_bias/num_heads`)
        that the vendored `extract_v`/`extract_feat` methods already rely on for
        the exact same parameter layout.

        A forward-pre-hook (not a forward hook) is used because we need the
        block's *input* (the residual-stream tensor `nn.MultiheadAttention`'s
        own q/k/v projections are applied to inside `attention()`), not its
        output.
        """
        resblocks = self.model.clip_backbone.backbone.visual.transformer.resblocks

        def make_hook(block: torch.nn.Module):
            def hook(module: torch.nn.Module, args: tuple) -> None:
                x = args[0]  # [B, L, D] -- residual-stream input to this block (L = 1 CLS + h*w patches)
                y = block.ln_1(x)  # matches ResidualAttentionBlock.attention()'s q_x=self.ln_1(q_x)
                d_model = y.shape[-1]
                attn_module = block.attn
                num_heads = attn_module.num_heads
                head_dim = d_model // num_heads

                # Only need q, k (not v) to get the attention matrix -- slice
                # in_proj_weight/bias to the first two thirds instead of
                # computing the unused v-projection.
                qk_weight = attn_module.in_proj_weight[: 2 * d_model]
                qk_bias = attn_module.in_proj_bias[: 2 * d_model] if attn_module.in_proj_bias is not None else None
                q, k = F.linear(y, qk_weight, qk_bias).chunk(2, dim=-1)  # each [B, L, d_model]

                B, L, _ = q.shape
                q = q.reshape(B, L, num_heads, head_dim).transpose(1, 2) * (
                    head_dim**-0.5
                )  # [B, num_heads, L, head_dim]
                k = k.reshape(B, L, num_heads, head_dim).transpose(1, 2)
                attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)  # [B, num_heads, L, L]
                captured.append(attn.mean(dim=1).detach())  # average over heads -> [B, L, L]

            return hook

        return [block.register_forward_pre_hook(make_hook(block)) for block in resblocks]

    def forward(self, images: Tensor) -> EncoderOutput:
        """images: [B, 3, H, W] in [0, 1] -> features: [B, output_dim, H/16, W/16]."""
        captured_attn: list[Tensor] = []
        handles = self._attention_hooks(captured_attn) if self.extract_attention else []
        try:
            features = self.model.get_pooled_feats(images)  # [B, output_dim, h, w]; also triggers our hooks above
        finally:
            for handle in handles:
                handle.remove()

        attention = None
        if self.extract_attention:
            # One captured tensor per transformer block: [B, L, L]. Average
            # over blocks, then drop the CLS token's row and column.
            attention = torch.stack(captured_attn, dim=0).mean(dim=0)  # [B, L, L]
            attention = attention[:, 1:, 1:]  # [B, h*w, h*w]

        return EncoderOutput(features=features, attention=attention)
