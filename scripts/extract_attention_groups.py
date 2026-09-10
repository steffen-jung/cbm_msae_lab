"""Standalone patch-grouping cache builder for the S2AE-style
`loss.group_sparsity.grouping={attention,feature}` /
`loss.exclusivity.grouping={attention,feature}` options (see `attention_grouping.py`).

Runs the dataset once through the (frozen) encoder, clusters each image's
patches into groups -- either by patch-to-patch attention (`--method
attention`, needs `extract_attention=True`) or by raw per-patch feature
similarity (`--method feature`, attention-free; try this if attention-based
clusters aren't granular enough on the object itself) -- and writes the
resulting per-image `[P]` cluster-label arrays to a memmap cache. Never
touches the SAE -- group labels depend only on the frozen encoder, so this
cache is reused across every SAE/loss experiment that shares the same
encoder + clustering hyperparameters + method, regardless of what SAE size or
losses are used in a given training run.

This clustering is deliberately never done inside `scripts/train.py`'s
training loop: `sklearn.cluster.AgglomerativeClustering` on a CPU, per image,
per batch, would be a severe bottleneck during GPU training. Run this script
once beforehand instead.

Examples:
    uv run scripts/extract_attention_groups.py --method attention
    uv run scripts/extract_attention_groups.py --method feature
    uv run scripts/extract_attention_groups.py encoder=dinov3 loss.attention_grouping.n_clusters=32
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cbm_msae_lab.attention_grouping import (
    GroupLabelWriter,
    cluster_patches,
    cluster_patches_by_features,
    compute_group_cache_key,
)
from cbm_msae_lab.config_schema import Config, register_configs

register_configs()
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument(
        "--method",
        choices=["attention", "feature"],
        default="attention",
        help="cluster by the encoder's self-attention (default) or its raw per-patch feature similarity",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild the cache even if a complete one already exists for this key",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra-style config overrides, e.g. encoder=dinov3")
    args = parser.parse_args()

    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg: Config = hydra.compose(config_name="config", overrides=args.overrides)  # type: ignore[assignment]

    device = args.device
    needs_attention = args.method == "attention"
    # Force attention extraction on only when actually clustering by attention
    # -- that flag exists so `scripts/train.py` doesn't pay for hooks it isn't
    # using, and `--method feature` doesn't need it either.
    encoder = hydra.utils.instantiate(cfg.encoder, extract_attention=needs_attention).to(device)
    encoder.eval()

    dataset = hydra.utils.instantiate(cfg.dataset, split=args.split)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    n_clusters = cfg.loss.attention_grouping.n_clusters
    spatial_coeff = cfg.loss.attention_grouping.spatial_coeff

    cache_key = compute_group_cache_key(
        encoder_cfg=cfg.encoder,
        n_clusters=n_clusters,
        spatial_coeff=spatial_coeff,
        dataset_cfg=cfg.dataset,
        split=args.split,
        method=args.method,
    )
    # Idempotent by default, for the same reason as scripts/extract_raw_features.py:
    # `GroupLabelWriter` opens labels.dat with mode="w+", which truncates it, so a
    # rebuild while another job reads the same cache would pull the data out from
    # under it. Several sbatch scripts build this cache and are meant to be
    # submitted together (see scripts/run_structure_ablation.sh).
    meta_path = Path(cfg.train.cache.dir) / "group_labels" / cache_key / "meta.json"
    if meta_path.exists() and not args.overwrite:
        num_samples = json.loads(meta_path.read_text())["num_samples"]
        if num_samples == len(dataset):
            log.info(f"cache {cache_key} already complete ({num_samples} samples) -- skipping (--overwrite to force)")
            return
        log.warning(f"cache {cache_key} holds {num_samples} samples but the split has {len(dataset)} -- rebuilding")

    log.info(f"Building {args.method}-grouping cache for split={args.split!r} -> cache key {cache_key}")

    writer: GroupLabelWriter | None = None
    with torch.no_grad():
        for images, _labels, _idx in loader:
            images = images.to(device)
            out = encoder(images)

            grid_h, grid_w = out.features.shape[-2], out.features.shape[-1]
            P = grid_h * grid_w
            if needs_attention:
                if out.attention is None:
                    raise RuntimeError(
                        f"{type(encoder).__name__} did not return an attention map even with extract_attention=True"
                    )
                if out.attention.shape[1] != P:
                    raise RuntimeError(f"attention has {out.attention.shape[1]} patches but feature grid is {grid_h}x{grid_w}")

            B = images.shape[0]
            if writer is None:
                writer = GroupLabelWriter(cfg.train.cache.dir, cache_key, len(dataset), P)

            for b in range(B):
                if needs_attention:
                    labels = cluster_patches(
                        out.attention[b], grid_h, grid_w, n_clusters=n_clusters, spatial_coeff=spatial_coeff
                    )
                else:
                    labels = cluster_patches_by_features(
                        out.features[b], grid_h, grid_w, n_clusters=n_clusters, spatial_coeff=spatial_coeff
                    )
                writer.write(labels)

    if writer is None:
        raise RuntimeError(f"dataset is empty, nothing to cache (cache_key={cache_key})")
    writer.finalize(OmegaConf.to_container(cfg, resolve=True))
    log.info(f"Done: {cfg.train.cache.dir}/group_labels/{cache_key}")


if __name__ == "__main__":
    main()
