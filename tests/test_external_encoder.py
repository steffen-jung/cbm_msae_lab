"""Unit tests for the MonosemanticityScore external-encoder pairing."""

from __future__ import annotations

import pytest
import torch

from cbm_msae_lab.encoders.base import Encoder, EncoderOutput
from cbm_msae_lab.external_encoder import EXTERNAL_ENCODER_NAME, build_external_encoder, external_embed


class DummyEncoder(Encoder):
    def __init__(self, output_dim: int = 8, grid_size: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.freeze()

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        B = images.shape[0]
        return EncoderOutput(features=torch.randn(B, self.output_dim, self.grid_size, self.grid_size))


def test_external_encoder_pairing_is_symmetric() -> None:
    clip_target = "cbm_msae_lab.encoders.clip_dinoiser.ClipDinoiserEncoder"
    dinov3_target = "cbm_msae_lab.encoders.dinov3.DINOv3Encoder"

    assert EXTERNAL_ENCODER_NAME[clip_target] == "dinov3"
    assert EXTERNAL_ENCODER_NAME[dinov3_target] == "clip_dinoiser"


def test_build_external_encoder_rejects_unknown_training_encoder() -> None:
    with pytest.raises(ValueError, match="no external-encoder pairing"):
        build_external_encoder("some.other.Encoder", device="cpu")


def test_external_embed_is_l2_normalized_and_pooled() -> None:
    encoder = DummyEncoder(output_dim=6, grid_size=3)
    images = torch.zeros(2, 3, 32, 32)

    embeddings = external_embed(encoder, images)

    assert embeddings.shape == (2, 6)
    norms = embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)
