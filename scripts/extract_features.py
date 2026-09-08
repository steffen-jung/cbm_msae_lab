"""Standalone activation-caching step: run the (frozen) encoder + projection
once over a dataset split and write the result to disk, without training
anything. Useful to pre-build a cache ahead of time (e.g. for many later SAE
sweeps that all share the same encoder+projection+upsampler configuration),
rather than paying the encoder's cost inside `scripts/train.py`'s first epoch.

Examples:
    uv run scripts/extract_features.py
    uv run scripts/extract_features.py encoder=dinov3 upsampler=anyup upsampler.target_resolution=64
    uv run scripts/extract_features.py dataset.split=test   # not a real dataset field; see --split below
"""

from __future__ import annotations

import argparse
import logging

import hydra
import torch
from omegaconf import OmegaConf

from cbm_msae_lab.caching import compute_cache_key, fingerprint_module
from cbm_msae_lab.config_schema import Config, register_configs
from cbm_msae_lab.feature_extraction import build_activation_cache
from cbm_msae_lab.pipeline import ConceptPipeline

register_configs()
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style config overrides, e.g. encoder=dinov3 upsampler=anyup",
    )
    args = parser.parse_args()

    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg: Config = hydra.compose(config_name="config", overrides=args.overrides)  # type: ignore[assignment]

    device = args.device
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    projection = hydra.utils.instantiate(cfg.projection).to(device)
    upsampler = hydra.utils.instantiate(cfg.upsampler).to(device)
    # No SAE is needed for feature extraction; `ConceptPipeline.project` (used
    # internally by `build_activation_cache`) never touches it.
    pipeline = ConceptPipeline(encoder, projection, upsampler, sae=None).to(device)  # type: ignore[arg-type]

    dataset = hydra.utils.instantiate(cfg.dataset, split=args.split)

    cache_key = compute_cache_key(
        encoder_cfg=cfg.encoder,
        projection_fingerprint=fingerprint_module(projection),
        upsampler_cfg=cfg.upsampler,
        dataset_name=cfg.dataset._target_,
        split=args.split,
        image_size=cfg.dataset.image_size,
    )
    log.info(f"Extracting features for split={args.split!r} -> cache key {cache_key}")

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
    log.info(f"Done: {cfg.train.cache.dir}/{cache_key}")


if __name__ == "__main__":
    main()
