"""Shape smoke tests for the pipeline wiring.

The fast tests below use `DummyEncoder` (fixed random features, no downloads)
to check `EncoderProjection` + `Upsampler` + SAE composition across every
combination the plan called for. The real encoders (CLIP-DINOiser, DINOv3,
AnyUp) need a checkpoint file / network access, so they get their own,
separately-skippable integration tests at the bottom of this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cbm_msae_lab.encoders.base import Encoder, EncoderOutput
from cbm_msae_lab.pipeline import ConceptPipeline
from cbm_msae_lab.projection import EncoderProjection
from cbm_msae_lab.sae.model import ConfigurableActivationSAE
from cbm_msae_lab.upsampling.bilinear import BilinearUpsampler
from cbm_msae_lab.upsampling.identity import IdentityUpsampler

REPO_ROOT = Path(__file__).resolve().parent.parent


class DummyEncoder(Encoder):
    """A frozen "encoder" that ignores its input and returns fixed-shape random
    features -- stands in for CLIP-DINOiser/DINOv3 so these tests need no
    downloads and run in milliseconds."""

    def __init__(self, output_dim: int = 32, grid_size: int = 14, extract_attention: bool = False) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.extract_attention = extract_attention
        self.freeze()

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        B = images.shape[0]
        features = torch.randn(B, self.output_dim, self.grid_size, self.grid_size)
        attention = None
        if self.extract_attention:
            p = self.grid_size * self.grid_size
            raw = torch.rand(B, p, p)
            attention = raw / raw.sum(dim=-1, keepdim=True)  # fake but valid row-stochastic attention
        return EncoderOutput(features=features, attention=attention)


def build_sae(
    activation_dim: int, dict_size: int = 64, k: int = 4, activation: str = "relu"
) -> ConfigurableActivationSAE:
    group_sizes = [dict_size // 2, dict_size - dict_size // 2]
    return ConfigurableActivationSAE(activation_dim, dict_size, k, group_sizes, activation=activation)


@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("activation", ["relu", "sigmoid"])
def test_no_upsampling(kernel_size: int, activation: str) -> None:
    encoder = DummyEncoder(output_dim=32, grid_size=14)
    projection = EncoderProjection(in_channels=32, out_channels=16, kernel_size=kernel_size)
    upsampler = IdentityUpsampler()
    sae = build_sae(activation_dim=16, activation=activation)
    pipeline = ConceptPipeline(encoder, projection, upsampler, sae)

    images = torch.rand(2, 3, 224, 224)

    x_img = pipeline.extract_features(images)
    assert x_img.shape == (2, 14 * 14, 16)

    out = pipeline(images)
    assert out.features.shape == (2, 16, 14, 14)
    assert out.latents.shape == (2, 64, 14, 14)
    assert out.reconstruction.shape == (2, 16, 14, 14)
    assert out.upsampled_latents is None


def test_feature_stage_upsampling_changes_sae_resolution() -> None:
    encoder = DummyEncoder(output_dim=32, grid_size=14)
    projection = EncoderProjection(in_channels=32, out_channels=16, kernel_size=3)
    upsampler = BilinearUpsampler(target_resolution=28, stage="features")
    sae = build_sae(activation_dim=16)
    pipeline = ConceptPipeline(encoder, projection, upsampler, sae)

    images = torch.rand(2, 3, 224, 224)
    out = pipeline(images)

    # The SAE now operates at the upsampled resolution, not the encoder's native one.
    assert out.features.shape == (2, 16, 28, 28)
    assert out.latents.shape == (2, 64, 28, 28)
    assert out.reconstruction.shape == (2, 16, 28, 28)
    assert out.upsampled_latents is None


def test_latent_stage_upsampling_only_affects_latents() -> None:
    encoder = DummyEncoder(output_dim=32, grid_size=14)
    projection = EncoderProjection(in_channels=32, out_channels=16, kernel_size=1)
    upsampler = BilinearUpsampler(target_resolution=28, stage="latents")
    sae = build_sae(activation_dim=16)
    pipeline = ConceptPipeline(encoder, projection, upsampler, sae)

    images = torch.rand(2, 3, 224, 224)
    out = pipeline(images)

    # The SAE still runs at the native resolution; only its output gets upsampled.
    assert out.features.shape == (2, 16, 14, 14)
    assert out.latents.shape == (2, 64, 14, 14)
    assert out.reconstruction.shape == (2, 16, 14, 14)
    assert out.upsampled_latents is not None
    assert out.upsampled_latents.shape == (2, 64, 28, 28)


def test_kernel_size_one_matches_a_per_token_linear() -> None:
    """Documents/verifies the claim in projection.py's docstring."""
    torch.manual_seed(0)
    projection = EncoderProjection(in_channels=8, out_channels=4, kernel_size=1)
    features = torch.randn(2, 8, 5, 5)

    conv_out = projection(features)  # [2, 4, 5, 5]

    weight = projection.conv.weight.reshape(4, 8)  # [out, in] (1x1 kernel squeezed)
    bias = projection.conv.bias
    linear_out = torch.einsum("bchw,oc->bohw", features, weight) + bias.view(1, 4, 1, 1)

    assert torch.allclose(conv_out, linear_out, atol=1e-5)


class TrainableDummyEncoder(DummyEncoder):
    """Same as DummyEncoder, but returns features that depend differentiably on
    the input, so a gradient check can confirm `EncoderProjection`'s weights
    actually receive gradients while the encoder's own params stay untouched."""

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        pooled = images.mean(dim=(2, 3), keepdim=True)  # [B, 3, 1, 1]
        features = pooled.expand(-1, 3, self.grid_size, self.grid_size)
        features = features.repeat(1, self.output_dim // 3 + 1, 1, 1)[:, : self.output_dim]
        return EncoderOutput(features=features)


def test_gradients_flow_into_projection_but_not_encoder() -> None:
    encoder = TrainableDummyEncoder(output_dim=9, grid_size=4)
    projection = EncoderProjection(in_channels=9, out_channels=6, kernel_size=3)
    upsampler = IdentityUpsampler()
    sae = build_sae(activation_dim=6, dict_size=16, k=2)
    pipeline = ConceptPipeline(encoder, projection, upsampler, sae)

    images = torch.rand(2, 3, 32, 32, requires_grad=True)
    x_img = pipeline.extract_features(images)
    x_img.sum().backward()

    assert projection.conv.weight.grad is not None
    assert torch.any(projection.conv.weight.grad != 0)
    for param in encoder.parameters():
        assert not param.requires_grad
        assert param.grad is None


def test_dummy_encoder_extract_attention_toggle() -> None:
    encoder = DummyEncoder(output_dim=8, grid_size=4, extract_attention=True)
    images = torch.rand(2, 3, 32, 32)

    out = encoder(images)
    assert out.attention is not None
    assert out.attention.shape == (2, 16, 16)
    assert torch.allclose(out.attention.sum(dim=-1), torch.ones(2, 16), atol=1e-5)

    encoder_off = DummyEncoder(output_dim=8, grid_size=4, extract_attention=False)
    assert encoder_off(images).attention is None


def test_group_sparsity_and_exclusivity_losses_under_attention_grouping() -> None:
    from cbm_msae_lab.losses.base import LossContext
    from cbm_msae_lab.losses.exclusivity import ExclusivityLoss
    from cbm_msae_lab.losses.group_sparsity import GroupSparsityLoss

    torch.manual_seed(0)
    B, grid_h, grid_w, dict_size, n_clusters = 2, 4, 4, 10, 3
    P = grid_h * grid_w

    sae = build_sae(activation_dim=6, dict_size=dict_size, k=2)
    f_img = torch.rand(B, P, dict_size)
    group_labels = torch.randint(0, n_clusters, (B, P))

    ctx = LossContext(
        sae=sae,
        trainer=None,  # not needed by these two losses
        x=torch.zeros(B * P, 6),
        f=f_img.reshape(B * P, dict_size),
        x_hat=torch.zeros(B * P, 6),
        post_act=f_img.reshape(B * P, dict_size),
        x_img=torch.zeros(B, P, 6),
        f_img=f_img,
        x_hat_img=torch.zeros(B, P, 6),
        grid=(grid_h, grid_w),
        step=0,
        group_labels=group_labels,
    )

    group_sparsity = GroupSparsityLoss(weight=1.0, grouping="attention", n_clusters=n_clusters)
    exclusivity = ExclusivityLoss(weight=1.0, grouping="attention", n_clusters=n_clusters)

    gs_value = group_sparsity.compute(ctx)
    es_value = exclusivity.compute(ctx)
    assert gs_value.shape == ()
    assert es_value.shape == ()
    assert gs_value.item() >= 0
    assert es_value.item() >= 0


def test_group_sparsity_attention_grouping_requires_group_labels() -> None:
    from cbm_msae_lab.losses.base import LossContext
    from cbm_msae_lab.losses.group_sparsity import GroupSparsityLoss

    sae = build_sae(activation_dim=6, dict_size=10, k=2)
    ctx = LossContext(
        sae=sae,
        trainer=None,
        x=torch.zeros(8, 6),
        f=torch.zeros(8, 10),
        x_hat=torch.zeros(8, 6),
        post_act=torch.zeros(8, 10),
        x_img=torch.zeros(2, 4, 6),
        f_img=torch.zeros(2, 4, 10),
        x_hat_img=torch.zeros(2, 4, 6),
        grid=(2, 2),
        step=0,
        group_labels=None,
    )
    loss = GroupSparsityLoss(weight=1.0, grouping="attention")
    with pytest.raises(RuntimeError, match="group_labels"):
        loss.compute(ctx)


# --------------------------------------------------------------------------
# Integration tests against the real, non-dummy components. Skipped unless
# the required checkpoint/network access is actually available, since this
# repo deliberately never triggers downloads on its own (see CLAUDE guidance
# in the plan: no sbatch, and no surprise network calls during `pytest`).
# --------------------------------------------------------------------------

CLIP_DINOISER_CHECKPOINT = REPO_ROOT / "checkpoints" / "clip_dinoiser" / "laion2b.pt"


@pytest.mark.skipif(
    not CLIP_DINOISER_CHECKPOINT.exists(),
    reason="run scripts/download_checkpoints.py first",
)
def test_clip_dinoiser_encoder_real_forward() -> None:
    from cbm_msae_lab.encoders.clip_dinoiser import ClipDinoiserEncoder

    encoder = ClipDinoiserEncoder(checkpoint_path=str(CLIP_DINOISER_CHECKPOINT))
    images = torch.rand(1, 3, 224, 224)
    out = encoder(images)
    assert out.features.shape == (1, 512, 14, 14)
    assert out.attention is None  # extract_attention defaults to False


@pytest.mark.skipif(
    not CLIP_DINOISER_CHECKPOINT.exists(),
    reason="run scripts/download_checkpoints.py first",
)
def test_clip_dinoiser_encoder_attention_extraction() -> None:
    from cbm_msae_lab.encoders.clip_dinoiser import ClipDinoiserEncoder

    encoder = ClipDinoiserEncoder(checkpoint_path=str(CLIP_DINOISER_CHECKPOINT), extract_attention=True)
    images = torch.rand(2, 3, 224, 224)
    out = encoder(images)

    assert out.attention is not None
    assert out.attention.shape == (2, 196, 196)
    assert torch.all(out.attention >= 0)
    # Each row sums to < 1 (attention paid to the dropped CLS column is excluded)
    # but not to ~0 (there should be substantial attention mass left among patches).
    row_sums = out.attention.sum(dim=-1)
    assert torch.all(row_sums > 0.1) and torch.all(row_sums <= 1.0 + 1e-4)

    # No hooks should be left registered after forward() returns.
    resblocks = encoder.model.clip_backbone.backbone.visual.transformer.resblocks
    assert sum(len(block._forward_pre_hooks) for block in resblocks) == 0


@pytest.mark.skipif(
    True,
    reason="requires network access to download DINOv3 weights via timm; run manually",
)
def test_dinov3_encoder_real_forward() -> None:
    from cbm_msae_lab.encoders.dinov3 import DINOv3Encoder

    encoder = DINOv3Encoder()
    images = torch.rand(1, 3, 224, 224)
    out = encoder(images)
    assert out.features.shape == (1, 768, 14, 14)
    assert out.attention is None

    encoder_attn = DINOv3Encoder(extract_attention=True)
    out_attn = encoder_attn(images)
    assert out_attn.attention.shape == (1, 196, 196)
    assert sum(len(b.attn._forward_hooks) for b in encoder_attn.model.blocks) == 0


@pytest.mark.skipif(
    True,
    reason="requires network access to download AnyUp weights via torch.hub; run manually",
)
def test_anyup_upsampler_real_forward() -> None:
    from cbm_msae_lab.upsampling.anyup import AnyUpUpsampler

    upsampler = AnyUpUpsampler(target_resolution=28)
    images = torch.rand(1, 3, 224, 224)
    features = torch.randn(1, 16, 14, 14)
    out = upsampler(images, features)
    assert out.shape == (1, 16, 28, 28)
