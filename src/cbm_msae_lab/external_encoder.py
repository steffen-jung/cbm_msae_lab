"""The "external encoder" `MonosemanticityScore` needs (see
`metrics/monosemanticity.py`'s docstring: "measured in the space of an
*external* encoder, not the SAE itself"). Using the same encoder the SAE is
trained to reconstruct isn't actually external -- its embedding is highly
correlated with the reconstruction target by construction. Instead, this
always pairs the training encoder with the *other* one of the two encoders
this codebase supports: CLIP-DINOiser's external encoder is DINOv3, and
DINOv3's external encoder is CLIP-DINOiser.
"""

from __future__ import annotations

from pathlib import Path

import hydra.utils
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor

from cbm_msae_lab.encoders.base import Encoder

CONF_ENCODER_DIR = Path(__file__).parent.parent.parent / "conf" / "encoder"

EXTERNAL_ENCODER_NAME: dict[str, str] = {
    "cbm_msae_lab.encoders.clip_dinoiser.ClipDinoiserEncoder": "dinov3",
    "cbm_msae_lab.encoders.dinov3.DINOv3Encoder": "clip_dinoiser",
}


def build_external_encoder(training_encoder_target: str, device: str) -> Encoder:
    """training_encoder_target: `cfg.encoder._target_` of the encoder being
    trained against. Loads the *other* encoder's own default config directly
    from `conf/encoder/<name>.yaml` (every field is filled in there, so no
    Hydra defaults-list resolution is needed) -- frozen, eval mode.
    """
    other_name = EXTERNAL_ENCODER_NAME.get(training_encoder_target)
    if other_name is None:
        raise ValueError(
            f"no external-encoder pairing known for training encoder {training_encoder_target!r}; "
            f"expected one of {list(EXTERNAL_ENCODER_NAME)}"
        )
    other_cfg = OmegaConf.load(CONF_ENCODER_DIR / f"{other_name}.yaml")
    if "defaults" in other_cfg:
        del other_cfg["defaults"]  # Hydra-only key, not a constructor argument

    encoder = hydra.utils.instantiate(other_cfg).to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


@torch.no_grad()
def external_embed(encoder: Encoder, images: Tensor) -> Tensor:
    """images: [B, 3, H, W] -> L2-normalized, global-average-pooled embeddings [B, embed_dim]."""
    features = encoder(images).features  # [B, C, H, W]
    pooled = features.mean(dim=(-2, -1))  # [B, C]
    return F.normalize(pooled, dim=-1)
