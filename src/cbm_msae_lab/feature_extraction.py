"""Runs a dataset once through the frozen encoder and writes its raw output
to an on-disk activation cache (see `caching.py`).

Shared by `scripts/extract_raw_features.py` (an explicit, standalone
pre-caching step) and, in principle, anything else that wants the same cache
built without going through the CLI.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from cbm_msae_lab.caching import MemmapActivationWriter
from cbm_msae_lab.encoders.base import Encoder


@torch.no_grad()
def build_raw_activation_cache(
    encoder: Encoder,
    dataset: Dataset,
    cache_dir: str,
    cache_key: str,
    batch_size: int,
    num_workers: int,
    device: str,
    resolved_config: dict[str, Any],
) -> None:
    """No `EncoderProjection` involved, so this cache is reusable regardless of
    projection kernel size/weights (see `caching.py::compute_raw_cache_key`).
    Pairs with `activation_loader.py`'s `mode="cache_encoder"`, which runs the
    (trainable) projection live, with gradients, on top of what's cached here.
    """
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    writer: MemmapActivationWriter | None = None
    for images, labels, _idx in loader:
        images = images.to(device)
        features = encoder(images).features  # [b, C_backbone, H0, W0]

        if writer is None:
            _, C, H, W = features.shape
            writer = MemmapActivationWriter(cache_dir, cache_key, len(dataset), C, H, W)
        writer.write_batch(features, labels)

    if writer is None:
        raise RuntimeError(f"dataset is empty, nothing to cache (cache_key={cache_key})")
    writer.finalize(resolved_config)
