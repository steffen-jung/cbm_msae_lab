"""MaskCLIP: a frozen CLIP ViT wrapped to expose dense (per-patch) features.

Vendored, functionally unchanged, from CFM (cfm/clip_dinoiser_backbone/maskclip/maskclip.py),
which itself adapts https://github.com/chongzhou96/MaskCLIP. This module is
part of the frozen, pretrained "DINOised CLIP" backbone: its exact forward
logic (which transformer block is hooked, how the value-vectors are
re-derived from the attention weights, how the positional embedding is
resized) must stay byte-for-byte identical to CFM's version, because the
pretrained ``CLIP_DINOiser`` checkpoint (obj_proj/bkg_decoder in
``clip_dinoiser.py``) was trained against exactly this feature-extraction
path. Do not "clean up" this file independently of the checkpoint.

Only ``ClipDinoiserEncoder`` (in ``encoders/clip_dinoiser.py``) is meant to be
imported by the rest of this codebase.
"""

from importlib.metadata import version

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from open_clip import create_model_from_pretrained, get_tokenizer
from torch import Tensor, nn

from .prompt_templates import imagenet_templates

# CLIP's own (OpenAI) image normalization stats -- applied inside this module
# so callers can pass raw [0, 1] tensors without knowing CLIP's preprocessing.
OPENAI_NORMALIZE = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


class MaskClip(nn.Module):
    """Frozen CLIP ViT-B/16 (open_clip) + a MaskCLIP-style dense projection head."""

    def __init__(self, backbone, decode_head, clip_model, class_names):
        super().__init__()

        self.decode_head = MaskClipHead(clip_model, class_names, **decode_head)
        self.patch_size = backbone.get("patch_size")
        self.img_size = tuple([backbone.get("img_size", 224)] * 2)
        pretrained = decode_head.get("pretrained")
        self.is_openai_clip = pretrained == "openai"

        if self.is_openai_clip:
            import clip

            model, _ = clip.load(clip_model, device="cpu")
            self.is_patch_first = True
        else:
            installed_version = version("open-clip-torch")
            major_version = int(installed_version.split(".")[0])
            # open_clip >= 3.x made the transformer batch-first internally,
            # which changes the shape of whatever we hook below.
            self.is_patch_first = major_version < 3
            model, _ = create_model_from_pretrained(clip_model, pretrained=pretrained)
        model.eval()
        self.clip_T = OPENAI_NORMALIZE
        self.hook_features: dict[str, Tensor] = {}
        self.backbone = model

        def hook_fn_forward(module, input, output):
            # output: [P, B, D] (patch-first, older open_clip) or [B, P, D] (open_clip>=3, batch-first)
            if not self.is_patch_first:
                self.hook_features["v"] = output.permute(1, 0, 2)  # -> [P, B, D]
            else:
                self.hook_features["v"] = output

        self.backbone.visual.transformer.resblocks[-2].register_forward_hook(hook_fn_forward)
        self._positional_embd = nn.Parameter(self.backbone.visual.positional_embedding.data.clone())

    def extract_feat(self, inputs: Tensor) -> Tensor:
        """inputs: [B, 3, H, W] (already CLIP-normalized) -> dense features [B, C, h, w]."""
        pos_embed = self.backbone.visual.positional_embedding

        B, C, H, W = inputs.shape
        hw_shape = (H // self.patch_size, W // self.patch_size)
        x_len, pos_len = hw_shape[0] * hw_shape[1], pos_embed.shape[0]

        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(f"{x_len}, {pos_len}")

            self.backbone.visual.positional_embedding.data = self.resize_pos_embed(
                self._positional_embd[None], hw_shape, (pos_h, pos_w), "bicubic"
            )[0]

        if self.is_openai_clip:
            _ = self.backbone.encode_image(inputs)
        else:
            _ = self.backbone(inputs)

        v = self.hook_features["v"]  # [P+1, B, D]
        v = self.extract_v(v, self.backbone.visual.transformer.resblocks[-1]).permute(1, 0, 2)  # -> [B, P+1, D]
        v = self.backbone.visual.ln_post(v)
        v = v[:, 1:]  # drop CLS token -> [B, P, D]
        v = v.reshape(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2).contiguous()  # -> [B, D, h, w]

        self.backbone.visual.positional_embedding.data = self._positional_embd
        return v

    def extract_v(self, x: Tensor, block) -> Tensor:
        """Re-derives the attention block's value-vectors and applies its residual MLP.

        x: [P+1, B, D] -> v: [P+1, B, D]
        """
        y = block.ln_1(x)
        y = F.linear(y, block.attn.in_proj_weight, block.attn.in_proj_bias)
        B, N, C = y.shape
        y = y.view(B, N, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * B, N, C // 3)
        y = F.linear(y, block.attn.out_proj.weight, block.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v = v + x
        v = v + block.mlp(block.ln_2(v))
        return v

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Bicubic-resizes CLIP's positional embedding to a new patch-grid resolution.

        pos_embed: [1, 1+pos_h*pos_w, D] -> [1, 1+h*w, D]
        """
        assert pos_embed.ndim == 3, "shape of pos_embed must be [B, L, C]"
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w) :]
        pos_embed_weight = pos_embed_weight.reshape(1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward(self, inputs: Tensor, return_feat: bool = False):
        """inputs: [B, 3, H, W] raw (unnormalized) images."""
        inputs = self.clip_T(inputs)
        x = self.extract_feat(inputs)  # [B, D, h, w]
        if return_feat:
            seg_logits, feats = self.decode_head(x, return_feat)
            return seg_logits, feats
        return self.decode_head(x)


class MaskClipHead(nn.Module):
    """Projects dense CLIP features into CLIP's shared image-text embedding space.

    ``class_embeddings`` (built from ``class_names`` at construction time) is
    only needed to produce a segmentation-style class map via ``cls_seg`` --
    this codebase never reads that map, only the projected dense features
    (``feat`` from ``forward(..., return_feat=True)``), but the constructor
    still needs *some* class list to build a well-formed module. CFM itself
    hardcodes a single placeholder class ``["dummy"]`` for the same reason;
    ``ClipDinoiserEncoder`` does the same.
    """

    def __init__(
        self,
        clip_model,
        class_names,
        in_channels=3,
        text_channels=512,
        use_templates=False,
        pretrained=None,
        **kwargs,
    ):
        super().__init__()

        self.text_channels = text_channels
        self.clip_model = clip_model
        self.pretrained = pretrained
        self.class_names = class_names
        self.in_channels = in_channels
        self.use_templates = use_templates

        if pretrained == "openai":
            import clip

            model, _ = clip.load(clip_model, device="cpu")
            self.tokenizer = clip.tokenize
        else:
            self.tokenizer = get_tokenizer(clip_model)
            model, _ = create_model_from_pretrained(clip_model, pretrained=pretrained)

        model.eval()
        self.register_buffer(
            "class_embeddings", self._get_class_embeddings(model, class_names)
        )  # [num_classes, text_channels]
        self.proj = nn.Conv2d(self.in_channels, text_channels, 1, bias=False)
        self.proj.weight = nn.Parameter(model.visual.proj.t()[:, :, None, None])

    @torch.no_grad()
    def _embed_label(self, text_model: nn.Module, label: str) -> Tensor:
        """Encodes one label name (optionally prompt-ensembled) into a single vector: [text_channels]."""
        if self.use_templates:
            templates = imagenet_templates
        elif "laion" in self.pretrained:
            templates = ["a photo of a {}", "a photo of an {}"]
        else:
            templates = ["a {}"]

        if self.pretrained == "openai":
            prompts = [template.format(label) for template in templates]
            all_prompts = self.tokenizer(prompts)
        else:
            all_prompts = [self.tokenizer(template.format(label)) for template in templates]
            all_prompts = torch.cat(all_prompts)

        out = text_model.encode_text(all_prompts)  # [num_templates, text_channels]
        out /= out.norm(dim=-1, keepdim=True)
        out = out.mean(dim=0)  # [text_channels]
        return out

    def _get_class_embeddings(self, text_model: nn.Module, class_names: list[str]) -> Tensor:
        aug_embeddings = torch.stack(
            [self._embed_label(text_model, label) for label in class_names]
        )  # [num_classes, text_channels]
        aug_embeddings = aug_embeddings / aug_embeddings.norm(dim=-1, keepdim=True)
        return aug_embeddings.squeeze(1)

    def forward(self, inputs: Tensor, return_feat: bool = False):
        """inputs (dense CLIP features): [B, in_channels, h, w]."""
        feat = self.proj(inputs)  # [B, text_channels, h, w]
        output = self.cls_seg(feat)
        if return_feat:
            return output, feat
        return output

    def cls_seg(self, feat: Tensor) -> Tensor:
        """feat: [B, text_channels, h, w] -> per-class softmax map [B, num_classes, h, w]."""
        feat = feat / feat.norm(dim=1, keepdim=True)
        output = F.conv2d(feat, self.class_embeddings[:, :, None, None])
        output = F.softmax(output * 100, dim=1)
        return output
