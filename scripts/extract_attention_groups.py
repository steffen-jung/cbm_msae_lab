"""Standalone attention-grouping cache builder for the S2AE-style
`loss.group_sparsity.grouping=attention` / `loss.exclusivity.grouping=attention`
option (see `attention_grouping.py`).

Runs the dataset once through the (frozen) encoder with `extract_attention`
forced on, clusters each image's patch-to-patch attention into groups
(`AgglomerativeClustering`, S2AE's own default hyperparameters), and writes
the resulting per-image `[P]` cluster-label arrays to a memmap cache. Never
touches `EncoderProjection` or the SAE -- group labels depend only on the
frozen encoder's attention, so this cache is reused across every SAE/loss
experiment that shares the same encoder + clustering hyperparameters,
regardless of what projection kernel size, SAE size, or other losses are
used in a given training run.

This clustering is deliberately never done inside `scripts/train.py`'s
training loop: `sklearn.cluster.AgglomerativeClustering` on a CPU, per image,
per batch, would be a severe bottleneck during GPU training. Run this script
once beforehand instead.

Examples:
    uv run scripts/extract_attention_groups.py
    uv run scripts/extract_attention_groups.py encoder=dinov3 loss.attention_grouping.n_clusters=32
"""

from __future__ import annotations

import argparse
import logging

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cbm_msae_lab.attention_grouping import GroupLabelWriter, cluster_patches, compute_group_cache_key
from cbm_msae_lab.config_schema import Config, register_configs

register_configs()
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("overrides", nargs="*", help="Hydra-style config overrides, e.g. encoder=dinov3")
    args = parser.parse_args()

    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg: Config = hydra.compose(config_name="config", overrides=args.overrides)  # type: ignore[assignment]

    device = args.device
    # Force attention extraction on for this script regardless of the config's
    # own `extract_attention` value -- that flag exists so `scripts/train.py`
    # doesn't pay for hooks it isn't using, but this script's entire purpose is
    # to extract attention.
    encoder = hydra.utils.instantiate(cfg.encoder, extract_attention=True).to(device)
    encoder.eval()

    dataset = hydra.utils.instantiate(cfg.dataset, split=args.split)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    n_clusters = cfg.loss.attention_grouping.n_clusters
    spatial_coeff = cfg.loss.attention_grouping.spatial_coeff

    cache_key = compute_group_cache_key(
        encoder_cfg=cfg.encoder,
        n_clusters=n_clusters,
        spatial_coeff=spatial_coeff,
        dataset_name=cfg.dataset._target_,
        split=args.split,
        image_size=cfg.dataset.image_size,
    )
    log.info(f"Building attention-grouping cache for split={args.split!r} -> cache key {cache_key}")

    writer: GroupLabelWriter | None = None
    with torch.no_grad():
        for images, _labels, _idx in loader:
            images = images.to(device)
            out = encoder(images)
            if out.attention is None:
                raise RuntimeError(
                    f"{type(encoder).__name__} did not return an attention map even with extract_attention=True"
                )

            B, P, _ = out.attention.shape
            grid_h, grid_w = out.features.shape[-2], out.features.shape[-1]
            if grid_h * grid_w != P:
                raise RuntimeError(f"attention has {P} patches but the feature grid is {grid_h}x{grid_w}")

            if writer is None:
                writer = GroupLabelWriter(cfg.train.cache.dir, cache_key, len(dataset), P)

            for b in range(B):
                labels = cluster_patches(
                    out.attention[b], grid_h, grid_w, n_clusters=n_clusters, spatial_coeff=spatial_coeff
                )
                writer.write(labels)

    if writer is None:
        raise RuntimeError(f"dataset is empty, nothing to cache (cache_key={cache_key})")
    writer.finalize(OmegaConf.to_container(cfg, resolve=True))
    log.info(f"Done: {cfg.train.cache.dir}/group_labels/{cache_key}")


if __name__ == "__main__":
    main()
