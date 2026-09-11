"""Covers the raw-encoder ("cache_encoder") activation-caching path: nothing
trainable sits between the encoder and the SAE, so the cache holds exactly the
tensor the SAE consumes and the loader only has to reshape it.
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
from cbm_msae_lab.sae.model import ConfigurableActivationSAE
from cbm_msae_lab.upsampling.bilinear import BilinearUpsampler
from cbm_msae_lab.upsampling.identity import IdentityUpsampler


class DummyEncoder(Encoder):
    """A frozen "encoder" that maps an image to fixed-shape features via a
    random-but-fixed linear map of its spatially pooled pixels -- stands in for
    CLIP-DINOiser/DINOv3 so these tests need no downloads and run in
    milliseconds. Deterministic on purpose: several tests compare a cached
    feature against a freshly computed one, which a `torch.randn`-per-call
    stand-in could never satisfy.
    """

    def __init__(self, output_dim: int = 8, grid_size: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.grid_size = grid_size
        generator = torch.Generator().manual_seed(0)
        self.register_buffer("mix", torch.randn(output_dim, 3, generator=generator))  # [C_out, 3]
        self.freeze()

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        pooled = torch.nn.functional.adaptive_avg_pool2d(images, self.grid_size)  # [B, 3, g, g]
        return EncoderOutput(features=torch.einsum("bchw,oc->bohw", pooled, self.mix))


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
            "train": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "cache": {"mode": "cache_encoder"},
                "eval": {"batch_size": batch_size},
            },
        }
    )


def test_resolve_loader_batch_size_uses_eval_for_val() -> None:
    from cbm_msae_lab.activation_loader import resolve_loader_batch_size

    cfg = OmegaConf.create({"train": {"batch_size": 64, "eval": {"batch_size": 8}}})
    assert resolve_loader_batch_size(cfg, "train") == 64
    assert resolve_loader_batch_size(cfg, "val") == 8
    assert resolve_loader_batch_size(cfg, "test") == 8


def test_raw_cache_key_stable_and_split_sensitive() -> None:
    encoder_cfg = OmegaConf.create({"_target_": "tests.test_caching.DummyEncoder"})
    dataset_cfg = OmegaConf.create({"_target_": "ds", "image_size": 224, "val_fraction": 0.1, "seed": 0})
    key_a = compute_raw_cache_key(encoder_cfg, dataset_cfg, "train")
    key_b = compute_raw_cache_key(encoder_cfg, dataset_cfg, "train")
    key_val = compute_raw_cache_key(encoder_cfg, dataset_cfg, "val")
    assert key_a == key_b
    assert key_a != key_val
    assert key_a.startswith("raw_")


def test_raw_cache_key_tracks_the_split_draw() -> None:
    """A different `seed`/`val_fraction` redraws which images are in train vs.
    val, so it must not reuse the same cache slot -- the cached rows would no
    longer line up with the split being trained on."""
    encoder_cfg = OmegaConf.create({"_target_": "tests.test_caching.DummyEncoder"})
    base = {"_target_": "ds", "image_size": 224, "val_fraction": 0.1, "seed": 0}
    key = compute_raw_cache_key(encoder_cfg, OmegaConf.create(base), "train")

    assert key != compute_raw_cache_key(encoder_cfg, OmegaConf.create({**base, "seed": 1}), "train")
    assert key != compute_raw_cache_key(encoder_cfg, OmegaConf.create({**base, "val_fraction": 0.2}), "train")


def test_cached_batch_matches_the_live_encoder_features() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = str(Path(tmp) / "cache")
        encoder = DummyEncoder(output_dim=8, grid_size=4)
        dataset = DummyDataset(n=6, image_size=16)
        cache_key = compute_raw_cache_key(
            OmegaConf.create({"_target_": "tests.test_caching.DummyEncoder"}),
            OmegaConf.create({"_target_": "tests.test_caching.DummyDataset", "image_size": 16}),
            "train",
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
        sae = ConfigurableActivationSAE(8, 16, k=2, group_sizes=[8, 8])
        pipeline = ConceptPipeline(encoder, IdentityUpsampler(), sae)

        loader = build_activation_loader(cfg, "train", dataset, pipeline, "cpu")
        x_img, _labels, idx = next(iter(loader))

        assert x_img.shape == (3, 16, 8)  # [B, P=4*4, C_sae]

        images = torch.stack([dataset[i][0] for i in idx.tolist()])
        expected = pipeline.extract_features(images)
        assert torch.allclose(x_img, expected, atol=1e-5)

        # Windows cannot delete memmap-backed .dat files while handles are open.
        loader.raw_loader.dataset.close()
        del x_img, _labels, idx, loader


def test_cache_encoder_rejects_feature_stage_upsampler() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = str(Path(tmp) / "cache")
        encoder = DummyEncoder(output_dim=8, grid_size=4)
        dataset = DummyDataset(n=3, image_size=16)
        sae = ConfigurableActivationSAE(8, 16, k=2, group_sizes=[8, 8])
        upsampler = BilinearUpsampler(target_resolution=8, stage="features")
        pipeline = ConceptPipeline(encoder, upsampler, sae)

        cfg = _dummy_cfg(cache_dir)
        with pytest.raises(ValueError, match="cache_encoder"):
            build_activation_loader(cfg, "train", dataset, pipeline, "cpu")
