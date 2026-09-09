"""Standalone raw-encoder-feature caching step: run only the frozen encoder
once over a dataset split and write its output to disk.

Pairs with `train.cache.mode="cache_encoder"`: nothing trainable sits between
the encoder and the SAE, so what is written here is exactly the tensor the SAE
consumes and the training loop can skip the expensive frozen-encoder forward
pass entirely.

Examples:
    uv run scripts/extract_raw_features.py
    uv run scripts/extract_raw_features.py --split val encoder=dinov3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from cbm_msae_lab.caching import compute_raw_cache_key
from cbm_msae_lab.config_schema import Config, register_configs
from cbm_msae_lab.feature_extraction import build_raw_activation_cache

register_configs()
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild the cache even if a complete one already exists for this key",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style config overrides, e.g. encoder=dinov3 (sae/loss overrides are irrelevant here)",
    )
    args = parser.parse_args()

    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg: Config = hydra.compose(config_name="config", overrides=args.overrides)  # type: ignore[assignment]

    device = args.device
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    dataset = hydra.utils.instantiate(cfg.dataset, split=args.split)

    cache_key = compute_raw_cache_key(encoder_cfg=cfg.encoder, dataset_cfg=cfg.dataset, split=args.split)

    # Idempotent by default. `MemmapActivationWriter` opens the feature file with
    # mode="w+", which truncates it -- so re-running this while a training job
    # reads the same cache would pull the data out from under it. Several sbatch
    # scripts extract before training and are meant to be submitted together.
    meta_path = Path(cfg.train.cache.dir) / cache_key / "meta.json"
    if meta_path.exists() and not args.overwrite:
        num_samples = json.loads(meta_path.read_text())["num_samples"]
        if num_samples == len(dataset):
            log.info(f"cache {cache_key} already complete ({num_samples} samples) -- skipping (--overwrite to force)")
            return
        log.warning(f"cache {cache_key} holds {num_samples} samples but the split has {len(dataset)} -- rebuilding")

    log.info(f"Extracting raw encoder features for split={args.split!r} -> cache key {cache_key}")

    build_raw_activation_cache(
        encoder=encoder,
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
