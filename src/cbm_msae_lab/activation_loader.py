"""Makes "where do SAE-input activations come from" (a live encoder forward
pass, or a precomputed cache) a single switch, so the training loop never
needs to know which one it's using -- both yield the same
(x_img [B, P, activation_dim], labels [B], idx [B]) batches.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from cbm_msae_lab.caching import (
    CachedActivationDataset,
    compute_cache_key,
    fingerprint_module,
)
from cbm_msae_lab.feature_extraction import build_activation_cache
from cbm_msae_lab.pipeline import ConceptPipeline, flatten_spatial

log = logging.getLogger(__name__)

Batch = tuple[Tensor, Tensor, Tensor]


class ActivationLoader:
    """Wraps a raw `DataLoader` with a per-batch transform, so it can be iterated
    over and over (once per epoch) like any other `DataLoader` -- unlike a bare
    generator, which is exhausted after a single pass."""

    def __init__(self, raw_loader: DataLoader, transform: Callable[[Batch], Batch]) -> None:
        self.raw_loader = raw_loader
        self.transform = transform

    def __iter__(self) -> Iterator[Batch]:
        for batch in self.raw_loader:
            yield self.transform(batch)

    def __len__(self) -> int:
        return len(self.raw_loader)


def build_activation_loader(
    cfg: DictConfig,
    split: str,
    dataset: Dataset,
    pipeline: ConceptPipeline,
    device: str,
) -> ActivationLoader:
    mode = cfg.train.cache.mode
    shuffle = split == "train"

    if mode == "live":
        raw_loader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=shuffle,
            num_workers=cfg.train.num_workers,
        )

        def live_transform(batch: Batch) -> Batch:
            images, labels, idx = batch
            images = images.to(device)
            return pipeline.extract_features(images), labels.to(device), idx

        return ActivationLoader(raw_loader, live_transform)

    if mode not in ("cache", "extract_and_cache"):
        raise ValueError(f"unknown cache.mode {mode!r}; expected 'live', 'cache', or 'extract_and_cache'")

    cache_key = compute_cache_key(
        encoder_cfg=cfg.encoder,
        projection_fingerprint=fingerprint_module(pipeline.projection),
        upsampler_cfg=cfg.upsampler,
        dataset_name=cfg.dataset._target_,
        split=split,
        image_size=cfg.dataset.image_size,
    )

    try:
        cached_dataset = CachedActivationDataset(cfg.train.cache.dir, cache_key)
    except FileNotFoundError:
        if mode == "cache":
            raise
        log.info(f"No cache found for split={split!r} (key={cache_key}); building one now (mode=extract_and_cache).")
        build_activation_cache(
            pipeline=pipeline,
            dataset=dataset,
            cache_dir=cfg.train.cache.dir,
            cache_key=cache_key,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            device=device,
            resolved_config=OmegaConf.to_container(cfg, resolve=True),
        )
        cached_dataset = CachedActivationDataset(cfg.train.cache.dir, cache_key)

    raw_loader = DataLoader(
        cached_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
    )

    def cached_transform(batch: Batch) -> Batch:
        features, labels, idx = batch  # features: [B, C_sae, H, W]
        return flatten_spatial(features.to(device)), labels.to(device), idx

    return ActivationLoader(raw_loader, cached_transform)
