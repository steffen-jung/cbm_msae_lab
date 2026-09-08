"""Precomputed, memory-mapped activations -- so the (frozen, expensive) encoder
forward pass is run once, not once per training run.

The frozen encoder never changes for a fixed encoder+projection+upsampler
configuration, so many different SAE/loss experiments can share one cache.
`compute_cache_key` hashes exactly the config fields that affect the *values*
stored in the cache (encoder choice+settings, the projection's own weights --
since it's trainable, its weights matter, not just its architecture --
whether/how features are upsampled, and the dataset/split/resolution); two
runs whose SAE or loss config differs but whose cache key matches will reuse
the same cache file, which is the entire point of caching here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.data import Dataset


def fingerprint_module(module: nn.Module) -> str:
    """A short hash of every parameter/buffer's raw bytes, so a change to
    (trainable) weights invalidates any cache keyed on this fingerprint."""
    hasher = hashlib.sha256()
    for _, tensor in sorted(module.state_dict().items()):
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


def compute_cache_key(
    encoder_cfg: DictConfig,
    projection_fingerprint: str,
    upsampler_cfg: DictConfig,
    dataset_name: str,
    split: str,
    image_size: int,
) -> str:
    payload: dict[str, Any] = {
        "encoder": OmegaConf.to_container(encoder_cfg, resolve=True),
        "projection_fingerprint": projection_fingerprint,
        # Only the upsampler matters for the cache if it runs before the SAE;
        # a "latents"-stage (or disabled) upsampler never touches what's cached.
        "upsampler": OmegaConf.to_container(upsampler_cfg, resolve=True)
        if upsampler_cfg.get("stage") == "features"
        else {"stage": "none"},
        "dataset": dataset_name,
        "split": split,
        "image_size": image_size,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class MemmapActivationWriter:
    """Writes [C, H, W] projected features + labels for `num_samples` images
    to a numpy memmap, one batch at a time, in the exact order they're given."""

    def __init__(
        self,
        cache_dir: str,
        cache_key: str,
        num_samples: int,
        channels: int,
        height: int,
        width: int,
    ) -> None:
        self.dir = Path(cache_dir) / cache_key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shape = (num_samples, channels, height, width)

        self.features = np.memmap(self.dir / "features.dat", dtype=np.float32, mode="w+", shape=self.shape)
        self.labels = np.memmap(self.dir / "labels.dat", dtype=np.int64, mode="w+", shape=(num_samples,))
        self._cursor = 0

    def write_batch(self, features: Tensor, labels: Tensor) -> None:
        """features: [b, C, H, W], labels: [b]."""
        n = features.shape[0]
        self.features[self._cursor : self._cursor + n] = features.detach().cpu().numpy()
        self.labels[self._cursor : self._cursor + n] = labels.detach().cpu().numpy()
        self._cursor += n

    def finalize(self, resolved_config: dict) -> None:
        if self._cursor != self.shape[0]:
            raise RuntimeError(f"wrote {self._cursor} samples but expected {self.shape[0]} -- cache is incomplete")
        self.features.flush()
        self.labels.flush()
        meta = {
            "shape": list(self.shape),
            "dtype": "float32",
            "num_samples": self.shape[0],
            "config": resolved_config,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))


class CachedActivationDataset(Dataset):
    """Reads back what `MemmapActivationWriter` wrote. Each item is one image's
    already-projected (and possibly feature-upsampled) [C, H, W] tensor --
    the training loop still has to `flatten_spatial` a whole batch of these
    into [B, P, C] itself, exactly as the live path does."""

    def __init__(self, cache_dir: str, cache_key: str) -> None:
        self.dir = Path(cache_dir) / cache_key
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no cache found at {self.dir} (missing meta.json) -- run scripts/extract_features.py "
                "first, or set train.cache.mode=extract_and_cache to build it automatically."
            )
        meta = json.loads(meta_path.read_text())
        self.shape = tuple(meta["shape"])

        self.features = np.memmap(self.dir / "features.dat", dtype=np.float32, mode="r", shape=self.shape)
        self.labels = np.memmap(self.dir / "labels.dat", dtype=np.int64, mode="r", shape=(self.shape[0],))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        features = torch.from_numpy(np.array(self.features[index]))  # [C, H, W]
        label = int(self.labels[index])
        return features, label, index
