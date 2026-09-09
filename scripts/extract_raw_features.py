"""Standalone raw-encoder-feature caching step: run only the frozen encoder
once over a dataset split and write its output to disk, stopping *before*
the trainable `EncoderProjection`.

Pairs with `train.cache.mode="cache_encoder"`: the training loop still runs
the projection live, with gradients, every step -- so the projection
actually trains (see `pipeline.py`'s "trained jointly with the SAE" design)
while skipping the expensive frozen-encoder forward pass.

Examples:
    uv run scripts/extract_raw_features.py
    uv run scripts/extract_raw_features.py --split val encoder=dinov3
"""

from __future__ import annotations

import argparse
import logging

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
        "overrides",
        nargs="*",
        help="Hydra-style config overrides, e.g. encoder=dinov3 (projection/sae/loss overrides are irrelevant here)",
    )
    args = parser.parse_args()

    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg: Config = hydra.compose(config_name="config", overrides=args.overrides)  # type: ignore[assignment]

    device = args.device
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    dataset = hydra.utils.instantiate(cfg.dataset, split=args.split)

    cache_key = compute_raw_cache_key(
        encoder_cfg=cfg.encoder,
        dataset_name=cfg.dataset._target_,
        split=args.split,
        image_size=cfg.dataset.image_size,
    )
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
