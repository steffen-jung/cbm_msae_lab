"""Covers the raw-encoder ("cache_encoder") activation-caching path: the
frozen encoder's output is cached, but the trainable `EncoderProjection` must
still run live, with gradients, every step.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from cbm_msae_lab.activation_loader import build_activation_loader
from cbm_msae_lab.caching import compute_raw_cache_key
from cbm_msae_lab.encoders.base import Encoder, EncoderOutput
from cbm_msae_lab.feature_extraction import build_raw_activation_cache
from cbm_msae_lab.pipeline import ConceptPipeline
from cbm_msae_lab.projection import EncoderProjection
from cbm_msae_lab.sae.model import ConfigurableActivationSAE
from cbm_msae_lab.upsampling.bilinear import BilinearUpsampler
from cbm_msae_lab.upsampling.identity import IdentityUpsampler


class DummyEncoder(Encoder):
    def __init__(self, output_dim: int = 8, grid_size: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.freeze()

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        B = images.shape[0]
        return EncoderOutput(features=torch.randn(B, self.output_dim, self.grid_size, self.grid_size))


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 6, image_size: int = 16) -> None:
        self.images = torch.rand(n, 3, image_size, image_size)
        self.labels = torch.zeros(n, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        return self.images[index], int(self.labels[index]), index


def _dummy_cfg(cache_dir: str, batch_size: int = 3, num_workers: int = 0) -> OmegaConf:
    return OmegaConf.create(
        {
            "encoder": {"_target_": "tests.test_caching.DummyEncoder"},
            "dataset": {"_target_": "tests.test_caching.DummyDataset", "image_size": 16},
            "train": {"batch_size": batch_size, "num_workers": num_workers, "cache": {"mode": "cache_encoder"}},
        }
    )


def test_raw_cache_key_stable_and_split_sensitive() -> None:
    encoder_cfg = OmegaConf.create({"_target_": "tests.test_caching.DummyEncoder"})
    key_a = compute_raw_cache_key(encoder_cfg, "ds", "train", 224)
    key_b = compute_raw_cache_key(encoder_cfg, "ds", "train", 224)
    key_val = compute_raw_cache_key(encoder_cfg, "ds", "val", 224)
    assert key_a == key_b
    assert key_a != key_val
    assert key_a.startswith("raw_")


def test_gradients_flow_into_projection_from_raw_cache() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = str(Path(tmp) / "cache")
        encoder = DummyEncoder(output_dim=8, grid_size=4)
        dataset = DummyDataset(n=6, image_size=16)
        cache_key = compute_raw_cache_key(
            OmegaConf.create({"_target_": "tests.test_caching.DummyEncoder"}),
            "tests.test_caching.DummyDataset",
            "train",
            16,
        )
        build_raw_activation_cache(
            encoder=encoder,
            dataset=dataset,
            cache_dir=cache_dir,
            cache_key=cache_key,
            batch_size=3,
            num_workers=0,
            device="cpu",
            resolved_config={},
        )

        cfg = _dummy_cfg(cache_dir)
        cfg.train.cache.dir = cache_dir
        projection = EncoderProjection(in_channels=8, out_channels=6, kernel_size=3)
        sae = ConfigurableActivationSAE(6, 16, k=2, group_sizes=[8, 8])
        pipeline = ConceptPipeline(encoder, projection, IdentityUpsampler(), sae)

        loader = build_activation_loader(cfg, "train", dataset, pipeline, "cpu")
        x_img, _labels, _idx = next(iter(loader))

        assert x_img.shape == (3, 16, 6)  # [B, P=4*4, C_sae]
        x_img.sum().backward()
        assert projection.conv.weight.grad is not None
        assert torch.any(projection.conv.weight.grad != 0)


def test_cache_encoder_rejects_feature_stage_upsampler() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = str(Path(tmp) / "cache")
        encoder = DummyEncoder(output_dim=8, grid_size=4)
        dataset = DummyDataset(n=3, image_size=16)
        projection = EncoderProjection(in_channels=8, out_channels=6, kernel_size=1)
        sae = ConfigurableActivationSAE(6, 16, k=2, group_sizes=[8, 8])
        upsampler = BilinearUpsampler(target_resolution=8, stage="features")
        pipeline = ConceptPipeline(encoder, projection, upsampler, sae)

        cfg = _dummy_cfg(cache_dir)
        with pytest.raises(ValueError, match="cache_encoder"):
            build_activation_loader(cfg, "train", dataset, pipeline, "cpu")
