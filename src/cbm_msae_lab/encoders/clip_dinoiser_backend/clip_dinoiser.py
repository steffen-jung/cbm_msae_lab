"""CLIP-DINOiser ("DINOised CLIP"): dense CLIP features refined to be spatially crisp.

Adapted from CFM (cfm/clip_dinoiser_backbone/clip_dinoiser/clip_dinoiser.py),
which in turn adapts https://github.com/wysoczanska/clip_dinoiser (Wysoczanska
& Simeoni et al., ECCV 2024, arXiv:2312.12359). "DINOising" happens entirely
inside the pretrained ``obj_proj``/``bkg_decoder`` conv weights below: at
training time (not something this codebase repeats) those weights were fit so
that a *self*-correlation of the projected CLIP features approximates DINO's
patch-correlation structure. At inference we therefore never need a live DINO
ViT forward pass -- only these two frozen conv layers plus the frozen MaskCLIP
backbone.

Compared to CFM's original ``clip_dinoiser.py``, this vendored version keeps
only the code path that ``ClipDinoiserEncoder.get_pooled_feats()`` (used by
this codebase) actually exercises: the base class's segmentation/decoder-head
forward pass, the FOUND background-refinement branch, and the (never-invoked
in CFM's own checkpoint-loading code) live-DINO correlation methods have been
removed as dead code for this use case. The retained methods are otherwise
unmodified, since ``obj_proj``/``bkg_decoder``'s pretrained weights are only
meaningful paired with this exact feature-extraction path.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor, nn

from .builder import build_model


class DinoCLIP(nn.Module):
    """Base class: holds the frozen MaskCLIP backbone and shared pooling utilities."""

    def __init__(
        self,
        clip_backbone,
        class_names,
        vit_arch="vit_base",
        vit_patch_size=16,
        enc_type_feats="k",
        gamma=0.2,
        delta=0.99,
        apply_found=False,
    ):
        super().__init__()
        self.vit_arch = vit_arch
        self.enc_type_feats = enc_type_feats
        self.gamma = gamma
        self.vit_patch_size = vit_patch_size
        self.apply_found = apply_found
        self.delta = delta

        config_dir = Path(__file__).resolve().parent / "configs"
        maskclip_cfg = OmegaConf.load(str(config_dir / f"{clip_backbone}.yaml"))
        self.clip_backbone = build_model(maskclip_cfg["model"], class_names=class_names)
        for param in self.clip_backbone.parameters():
            param.requires_grad = False

    def make_input_divisible(self, x: Tensor) -> Tensor:
        """Zero-pads H/W up to the next multiple of the ViT patch size. x: [B, 3, H, W]."""
        B, _, H_0, W_0 = x.shape
        pad_w = (self.vit_patch_size - W_0 % self.vit_patch_size) % self.vit_patch_size
        pad_h = (self.vit_patch_size - H_0 % self.vit_patch_size) % self.vit_patch_size
        return F.pad(x, (0, pad_w, 0, pad_h), value=0)

    def get_clip_features(self, x: Tensor):
        """x: [B, 3, H, W] raw image -> dense CLIP features [B, text_channels, h, w], class map [B, num_classes, h, w]."""
        x = self.make_input_divisible(x)
        maskclip_map, feat = self.clip_backbone(x, return_feat=True)
        return feat, maskclip_map

    @staticmethod
    def compute_weighted_pool(maskclip_feats: Tensor, corrs: Tensor) -> Tensor:
        """Pools ``maskclip_feats`` using ``corrs`` as (thresholded) similarity weights.

        maskclip_feats: [B, C, h_m, w_m], corrs: [B, h_w*w_w, h_w, w_w] -> [B, C, h_w, w_w]
        """
        B = maskclip_feats.shape[0]
        h_m, w_m = maskclip_feats.shape[-2:]
        h_w, w_w = corrs.shape[-2:]

        if (h_m != h_w) or (w_m != w_w):
            maskclip_feats = F.interpolate(maskclip_feats, size=(h_w, w_w), mode="bilinear", align_corners=False)
            h_m, w_m = h_w, w_w

        maskclip_feats_ref = torch.einsum("bnij, bcij -> bcn", corrs, maskclip_feats)  # [B, C, h_w*w_w]
        norm_factor = corrs.flatten(-2, -1).sum(dim=-1)[:, None]  # [B, 1, h_w*w_w]
        maskclip_feats_ref = maskclip_feats_ref / (norm_factor + 1e-6)
        return maskclip_feats_ref.reshape(B, -1, h_m, w_m)  # [B, C, h_w, w_w]


class CLIP_DINOiser(DinoCLIP):
    """Adds the pretrained ``obj_proj``/``bkg_decoder`` heads that encode "DINO-ness"."""

    def __init__(
        self,
        clip_backbone,
        class_names,
        vit_arch="vit_base",
        vit_patch_size=16,
        enc_type_feats="v",
        feats_idx=-3,
        gamma=0.2,
        delta=0.99,
        in_dim=256,
        conv_kernel=3,
    ):
        super().__init__(clip_backbone, class_names, vit_arch, vit_patch_size, enc_type_feats, gamma)
        if vit_patch_size == 16:
            in_size = 768 if feats_idx != "final" else 512
        elif vit_patch_size == 14:
            in_size = 1024 if feats_idx != "final" else 768
        else:
            raise ValueError(f"unsupported vit_patch_size={vit_patch_size}")
        self.gamma = gamma
        self.feats_idx = feats_idx
        self.delta = delta
        self.in_dim = in_dim
        # Both convs are part of the frozen pretrained checkpoint: their kernel
        # sizes are fixed by what the checkpoint was trained with, NOT a knob
        # this codebase exposes (that role is played by this repo's own,
        # separately-trained `EncoderProjection`, applied downstream).
        self.bkg_decoder = nn.Conv2d(in_size, 1, (1, 1))
        self.obj_proj = nn.Conv2d(
            in_size,
            in_dim,
            (conv_kernel, conv_kernel),
            padding=conv_kernel // 2,
            padding_mode="replicate",
        )
        self.is_patch_first = self.clip_backbone.is_patch_first

        # Hook the intermediate CLIP transformer block whose features obj_proj was trained on.
        if feats_idx != "final":
            train_feats: dict[str, Tensor] = {}

            def get_activation(name, is_patch_first=False):
                def hook(model, input, output):
                    if is_patch_first:
                        train_feats[name] = output.detach().permute(1, 0, 2)  # -> [B, P, D]
                    else:
                        train_feats[name] = output.detach()  # already [B, P, D]

                return hook

            self.clip_backbone.backbone.visual.transformer.resblocks[feats_idx].ln_2.register_forward_hook(
                get_activation("clip_inter", self.is_patch_first)
            )
            self.train_feats = train_feats

    @torch.no_grad()
    def get_pooled_feats(self, x: Tensor) -> Tensor:
        """The main entry point: raw image -> dense, DINOised CLIP features.

        x: [B, 3, H, W] -> [B, text_channels, h, w] (h = H // patch_size, w = W // patch_size)
        """
        x = self.make_input_divisible(x)
        clip_proj_feats = self.get_clip_features(x)[0]  # [B, text_channels, h, w]
        B, c_dim, h, w = clip_proj_feats.shape

        if self.feats_idx != "final":
            clip_feats = self.train_feats["clip_inter"]  # [B, 1+h*w, D_inter]
            B, N, c_dim = clip_feats.shape
            clip_feats = clip_feats[:, 1:, :].permute(0, 2, 1).reshape(B, c_dim, h, w)  # drop CLS -> [B, D_inter, h, w]
        else:
            clip_feats = clip_proj_feats

        proj_feats = self.obj_proj(clip_feats).reshape(B, self.in_dim, -1)  # [B, in_dim, h*w]
        proj_feats = proj_feats / proj_feats.norm(dim=1, keepdim=True)
        corrs = torch.matmul(proj_feats.permute(0, 2, 1), proj_feats).reshape(B, h * w, h, w)  # [B, h*w, h, w]

        output = clip_proj_feats  # [B, text_channels, h, w]
        if self.gamma:
            corrs[corrs < self.gamma] = 0.0

        return self.compute_weighted_pool(output, corrs)  # [B, text_channels, h, w]
