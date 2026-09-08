"""Runs a dataset once through `ConceptPipeline.project` and writes the result
to an on-disk activation cache (see `caching.py`).

Shared by `scripts/extract_features.py` (an explicit, standalone pre-caching
step) and `activation_loader.py` (which calls this automatically the first
time `train.cache.mode="extract_and_cache"` finds no existing cache).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from cbm_msae_lab.caching import MemmapActivationWriter
from cbm_msae_lab.pipeline import ConceptPipeline


@torch.no_grad()
def build_activation_cache(
    pipeline: ConceptPipeline,
    dataset: Dataset,
    cache_dir: str,
    cache_key: str,
    batch_size: int,
    num_workers: int,
    device: str,
    resolved_config: dict[str, Any],
) -> None:
    pipeline.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    writer: MemmapActivationWriter | None = None
    for images, labels, _idx in loader:
        images = images.to(device)
        features = pipeline.project(images)  # [b, C_sae, H, W]

        if writer is None:
            _, C, H, W = features.shape
            writer = MemmapActivationWriter(cache_dir, cache_key, len(dataset), C, H, W)
        writer.write_batch(features, labels)

    if writer is None:
        raise RuntimeError(f"dataset is empty, nothing to cache (cache_key={cache_key})")
    writer.finalize(resolved_config)
