"""Precomputed, memory-mapped raw encoder features -- so the (frozen,
expensive) encoder forward pass is run once, not once per training run.

The frozen encoder's output never changes for a fixed encoder+dataset+split
configuration, regardless of what SAE/loss is trained on top of
it -- so `compute_raw_cache_key` deliberately depends on none of those,
letting one cache be reused across every experiment that shares the same
encoder. See `activation_loader.py`'s `mode="cache_encoder"`, which reads
this cache: it holds exactly the tensor the SAE consumes, since nothing
trainable sits between the encoder and the SAE.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import Dataset


def compute_raw_cache_key(
    encoder_cfg: DictConfig,
    dataset_cfg: DictConfig,
    split: str,
) -> str:
    """The *whole* dataset config goes into the key, not just its class path and
    image size: `val_fraction`/`seed` redraw which images land in train vs. val,
    and a key that ignored them would silently serve a cache whose row order no
    longer matches the split being trained on.
    """
    payload: dict[str, Any] = {
        "encoder": OmegaConf.to_container(encoder_cfg, resolve=True),
        "dataset": OmegaConf.to_container(dataset_cfg, resolve=True),
        "split": split,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return "raw_" + hashlib.sha256(blob).hexdigest()[:16]


class MemmapActivationWriter:
    """Writes [C, H, W] raw encoder features + labels for `num_samples` images
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
    raw encoder-output [C, H, W] tensor -- the training loop only has to
    `flatten_spatial` a whole batch of them, exactly as `activation_loader.py`'s
    `mode="cache_encoder"` transform does."""

    def __init__(self, cache_dir: str, cache_key: str) -> None:
        self.dir = Path(cache_dir) / cache_key
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no cache found at {self.dir} (missing meta.json) -- run scripts/extract_raw_features.py first."
            )
        meta = json.loads(meta_path.read_text())
        self.shape = tuple(meta["shape"])

        self.features = np.memmap(self.dir / "features.dat", dtype=np.float32, mode="r", shape=self.shape)
        self.labels = np.memmap(self.dir / "labels.dat", dtype=np.int64, mode="r", shape=(self.shape[0],))

    def close(self) -> None:
        """Release memmap file handles (needed on Windows before deleting the cache dir)."""
        for name in ("features", "labels"):
            arr = getattr(self, name, None)
            if arr is None:
                continue
            mmap = getattr(arr, "_mmap", None)
            if mmap is not None:
                mmap.close()
            setattr(self, name, None)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        features = torch.from_numpy(np.array(self.features[index]))  # [C, H, W]
        label = int(self.labels[index])
        return features, label, index
