"""Wires encoder -> projection -> optional feature-stage upsampler -> SAE.

Two entry points, matching how this module is used in two different places:

``extract_features(images)`` runs only the encoder/projection/upsampler part,
producing the ``[B, P, activation_dim]`` tensor that both the training loop
(``scripts/train.py``, feeding it to ``ComposableLossTrainer.loss``) and the
caching script (``scripts/extract_features.py``) need. It is the only part of
the pipeline that ever needs a gradient (through ``projection``'s Conv2d
weights) -- everything else here is frozen.

``forward(images)`` runs the full pipeline including the SAE, for inference /
analysis (e.g. producing concept activation maps to look at, or the optional
latent-stage upsampling for a higher-resolution concept map). Training itself
never calls ``forward`` -- it goes through ``ComposableLossTrainer`` directly
so its own bookkeeping (dead-feature counters, threshold EMA) stays correct.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from cbm_msae_lab.encoders.base import Encoder
from cbm_msae_lab.projection import EncoderProjection
from cbm_msae_lab.sae.model import ConfigurableActivationSAE
from cbm_msae_lab.upsampling.base import Upsampler


def flatten_spatial(features: Tensor) -> Tensor:
    """[B, C, H, W] -> [B, H*W, C]. Shared by `ConceptPipeline.extract_features` and
    `caching.CachedActivationDataset` consumers, so both produce the exact same
    [B, P, D] layout `ComposableLossTrainer.loss` expects."""
    B, C, H, W = features.shape
    return features.permute(0, 2, 3, 1).reshape(B, H * W, C)


@dataclass
class PipelineOutput:
    features: Tensor  # [B, C_sae, H, W] -- SAE input, after projection and optional feature-stage upsampling
    latents: Tensor  # [B, dict_size, H, W] -- SAE concept activations, spatially reshaped
    reconstruction: Tensor  # [B, C_sae, H, W] -- SAE.decode(latents), same resolution as `features`
    upsampled_latents: Tensor | None  # [B, dict_size, H_t, W_t] if the upsampler targets "latents", else None


class ConceptPipeline(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        projection: EncoderProjection,
        upsampler: Upsampler,
        sae: ConfigurableActivationSAE,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.upsampler = upsampler
        self.sae = sae

    def project(self, images: Tensor) -> Tensor:
        """images: [B, 3, H_img, W_img] -> [B, C_sae, H, W] (H, W = the SAE-input grid).

        Gradients flow only into `self.projection` (encoder is frozen; the
        upsampler, when present, is either parameter-free or itself frozen).
        """
        encoder_out = self.encoder(images)  # features: [B, C_backbone, H0, W0]
        features = self.projection(encoder_out.features)  # [B, C_sae, H0, W0]
        if self.upsampler.stage == "features":
            features = self.upsampler(images, features)  # [B, C_sae, Ht, Wt]
        return features

    def extract_features(self, images: Tensor) -> Tensor:
        """images: [B, 3, H_img, W_img] -> [B, P, activation_dim] (P = H*W of the SAE-input grid).

        This is what `ComposableLossTrainer.loss` and the caching script consume.
        """
        features = self.project(images)  # [B, C_sae, H, W]
        return flatten_spatial(features)  # [B, P, C_sae]

    def forward(self, images: Tensor, use_threshold: bool = True) -> PipelineOutput:
        """images: [B, 3, H_img, W_img]. `use_threshold` selects the SAE's inference-time
        sparsification (a learned per-feature threshold) instead of training's BatchTopK."""
        features = self.project(images)  # [B, C_sae, H, W]
        B, C, H, W = features.shape

        x_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, C_sae]
        latents_flat = self.sae.encode(x_flat, use_threshold=use_threshold)  # [B*H*W, dict_size]
        recon_flat = self.sae.decode(latents_flat)  # [B*H*W, C_sae]

        dict_size = latents_flat.shape[-1]
        latents = latents_flat.reshape(B, H, W, dict_size).permute(0, 3, 1, 2)  # [B, dict_size, H, W]
        reconstruction = recon_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C_sae, H, W]

        upsampled_latents = None
        if self.upsampler.stage == "latents":
            upsampled_latents = self.upsampler(images, latents)  # [B, dict_size, Ht, Wt]

        return PipelineOutput(
            features=features,
            latents=latents,
            reconstruction=reconstruction,
            upsampled_latents=upsampled_latents,
        )
