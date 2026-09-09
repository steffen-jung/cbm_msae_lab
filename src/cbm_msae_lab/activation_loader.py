"""Makes "where do SAE-input activations come from" (a live encoder forward
pass, or a precomputed cache of the frozen encoder's raw output) a single
switch, so the training loop never needs to know which one it's using --
both yield the same (x_img [B, P, activation_dim], labels [B], idx [B])
batches, and both run the trainable `EncoderProjection` live, with gradients,
every step.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from cbm_msae_lab.caching import CachedActivationDataset, compute_raw_cache_key
from cbm_msae_lab.pipeline import ConceptPipeline, flatten_spatial

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

    if mode != "cache_encoder":
        raise ValueError(f"unknown cache.mode {mode!r}; expected 'live' or 'cache_encoder'")

    # Skips the expensive frozen-encoder forward (cached), but still runs the
    # *trainable* projection live, with gradients, every step.
    if pipeline.upsampler.stage == "features":
        raise ValueError(
            "mode='cache_encoder' only caches raw encoder features, not the original images -- a "
            "feature-stage upsampler (e.g. AnyUp) needs the image for guidance and can't run from this "
            "cache. Use upsampler.stage='latents' or 'none' instead."
        )
    cache_key = compute_raw_cache_key(
        encoder_cfg=cfg.encoder,
        dataset_name=cfg.dataset._target_,
        split=split,
        image_size=cfg.dataset.image_size,
    )
    try:
        cached_dataset = CachedActivationDataset(cfg.train.cache.dir, cache_key)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"no raw-encoder cache found for split={split!r} (key={cache_key}) -- run "
            "scripts/extract_raw_features.py first. mode='cache_encoder' never computes "
            "the encoder forward pass on the fly."
        ) from e

    raw_loader = DataLoader(
        cached_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
    )

    def cache_encoder_transform(batch: Batch) -> Batch:
        # No upsampler call here: the ValueError above already ruled out
        # stage="features"; "latents"/"none" upsamplers never touch this
        # (pre-SAE) point in the pipeline.
        raw_features, labels, idx = batch  # raw_features: [B, C_backbone, H0, W0]
        raw_features = raw_features.to(device)
        projected = pipeline.projection(raw_features)  # [B, C_sae, H0, W0]
        return flatten_spatial(projected), labels.to(device), idx

    return ActivationLoader(raw_loader, cache_encoder_transform)
