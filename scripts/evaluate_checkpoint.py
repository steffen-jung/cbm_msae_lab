"""Post-hoc reconstruction/concept-quality evaluation for a finished checkpoint.

The same metrics `scripts/train.py::run_eval` computes during training
(reconstruction MSE/cosine-sim, FVU, and -- unless `--skip-expensive` --
Monosemanticity Score and Tversky-MS), but standalone: run this any time
after training to (re-)evaluate a checkpoint, on any split, without having to
re-run training. Like `scripts/evaluate_task_accuracy.py`, it carries no
Hydra config of its own -- the checkpoint already stores the exact resolved
config it was trained under.

Example:
    uv run scripts/evaluate_checkpoint.py --checkpoint outputs/checkpoints/best.pt
    uv run scripts/evaluate_checkpoint.py --checkpoint outputs/checkpoints/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import hydra.utils
import torch
from torch.utils.data import DataLoader

from cbm_msae_lab.attention_grouping import CachedGroupLabelDataset, compute_group_cache_key
from cbm_msae_lab.checkpointing import load_for_reproduction
from cbm_msae_lab.external_encoder import build_external_encoder, external_embed
from cbm_msae_lab.grouping import CLUSTERED_GROUPINGS, clustered_or_tile, resolve_group_labels
from cbm_msae_lab.metrics.fvu import FVUMetric
from cbm_msae_lab.metrics.monosemanticity import MonosemanticityScore, PachMonosemanticityScore
from cbm_msae_lab.metrics.reconstruction import ReconstructionMetric
from cbm_msae_lab.metrics.region_consistency import RegionConsistencyMetric
from cbm_msae_lab.metrics.sparsity import ActivationFrequencyMetric, L0Metric
from cbm_msae_lab.metrics.tversky_ms import TverskyMS
from cbm_msae_lab.pipeline import ConceptPipeline, flatten_spatial

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="path to a checkpoint written by scripts/train.py")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None, help="default: the whole split")
    parser.add_argument("--skip-expensive", action="store_true", help="skip Monosemanticity Score / Tversky-MS")
    parser.add_argument(
        "--region-grouping",
        choices=["auto", "tile", "attention", "feature"],
        default="auto",
        help="what the region-consistency metric groups patches by; 'auto' (default) matches "
        "whatever clustering the checkpoint's own structural loss trained with, falling back "
        "to 'tile' if none",
    )
    parser.add_argument(
        "--region-tile-size",
        type=int,
        default=2,
        help="tile size the region-consistency metric partitions the patch grid with ('tile' only)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="output JSON path (default: outputs/eval/<checkpoint-stem>_<split>.json)")
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    pipeline: ConceptPipeline,
    loader: DataLoader,
    device: str,
    max_samples: int | None,
    external_encoder,
    region_grouping: str = "tile",
    region_tile_size: int = 2,
    region_n_clusters: int = 20,
    region_group_labels: CachedGroupLabelDataset | None = None,
) -> dict[str, float]:
    pipeline.eval()
    fvu_metric = FVUMetric(pipeline.encoder.output_dim).to(device)
    recon_metric = ReconstructionMetric().to(device)
    l0_metric = L0Metric().to(device)
    frequency_metric = ActivationFrequencyMetric(pipeline.sae.dict_size).to(device)
    # Built on the first batch, once the patch grid (and, for "tile", the region
    # count) is known -- the checkpoint records no grid shape, and it depends on
    # the encoder *and* the upsampler.
    region_metric: RegionConsistencyMetric | None = None
    ms_metric = MonosemanticityScore(pipeline.sae.dict_size).to(device) if external_encoder is not None else None
    pach_ms_metric = (
        PachMonosemanticityScore(pipeline.sae.dict_size).to(device) if external_encoder is not None else None
    )
    tms_metric = TverskyMS(pipeline.sae.dict_size).to(device) if external_encoder is not None else None

    seen = 0
    for images, _labels, idx in loader:
        images = images.to(device)
        out = pipeline.forward(images, use_threshold=True)
        B, C, H, W = out.features.shape

        x_flat = flatten_spatial(out.features).reshape(B * H * W, C)
        x_hat_flat = flatten_spatial(out.reconstruction).reshape(B * H * W, C)
        fvu_metric.update(x_flat, x_hat_flat)
        recon_metric.update(x_flat, x_hat_flat)

        latents_img = out.latents.reshape(B, out.latents.shape[1], H * W).transpose(1, 2)  # [B, P, dict_size]
        l0_metric.update(latents_img.reshape(B * H * W, -1))
        frequency_metric.update(latents_img.reshape(B * H * W, -1))

        region_labels, n_regions = resolve_group_labels(
            region_grouping,
            grid=(H, W),
            tile_size=region_tile_size,
            batch_size=B,
            device=device,
            group_labels=region_group_labels.get_batch(idx).to(device) if region_group_labels is not None else None,
            n_clusters=region_n_clusters,
            consumer="evaluate_checkpoint.py --region-grouping",
        )
        if region_metric is None:
            region_metric = RegionConsistencyMetric(n_regions).to(device)
        region_metric.update(latents_img, region_labels)

        if ms_metric is not None and tms_metric is not None:
            dict_size = out.latents.shape[1]
            concept_activations = out.latents.reshape(B, dict_size, H * W).max(dim=-1).values  # [B, dict_size]
            image_embeddings = external_embed(external_encoder, images)  # [B, embed_dim]
            ms_metric.update(concept_activations, image_embeddings)
            pach_ms_metric.update(concept_activations, image_embeddings)
            tms_metric.update(concept_activations)

        seen += B
        if max_samples is not None and seen >= max_samples:
            break

    results = {"fvu": fvu_metric.compute().item()}
    for name, value in recon_metric.compute().items():
        results[name] = value.item()
    results["l0"] = l0_metric.compute().item()
    results.update(frequency_metric.summary())
    if region_metric is not None:
        for name, value in region_metric.compute().items():
            results[name] = value.item()
    if ms_metric is not None and tms_metric is not None:
        results["monosemanticity_score"] = ms_metric.compute().nanmean().item()
        results["monosemanticity_score_pach"] = pach_ms_metric.compute().nanmean().item()
        results["dead_latent_fraction"] = ms_metric.dead_fraction()
        results["tversky_ms"] = tms_metric.compute().item()
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    cfg, pipeline = load_for_reproduction(str(checkpoint_path), device=args.device)
    pipeline.eval()

    dataset = hydra.utils.instantiate(cfg.dataset, split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    external_encoder = None
    if not args.skip_expensive:
        external_encoder = build_external_encoder(cfg.encoder._target_, args.device)

    # "auto" resolves fresh from the checkpoint's stored loss config (clustered_or_tile),
    # not from its stored train.eval.region_grouping -- that value is frozen at train
    # time and, for checkpoints trained before this field existed, may be absent.
    region_grouping = args.region_grouping if args.region_grouping != "auto" else clustered_or_tile(cfg)
    region_group_labels = None
    if region_grouping in CLUSTERED_GROUPINGS:
        cache_key = compute_group_cache_key(
            encoder_cfg=cfg.encoder,
            n_clusters=cfg.loss.attention_grouping.n_clusters,
            spatial_coeff=cfg.loss.attention_grouping.spatial_coeff,
            dataset_cfg=cfg.dataset,
            split=args.split,
            method=region_grouping,
        )
        region_group_labels = CachedGroupLabelDataset(cfg.train.cache.dir, cache_key)
    log.info(f"region-consistency grouping: {region_grouping!r}")

    results = evaluate(
        pipeline,
        loader,
        args.device,
        args.max_samples,
        external_encoder,
        region_grouping=region_grouping,
        region_tile_size=args.region_tile_size,
        region_n_clusters=cfg.loss.attention_grouping.n_clusters,
        region_group_labels=region_group_labels,
    )
    log.info(f"[{args.split}] {len(dataset)} images | " + " | ".join(f"{k}={v:.4f}" for k, v in results.items()))

    out_path = Path(args.out) if args.out else Path("outputs/eval") / f"{checkpoint_path.stem}_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(
            {"checkpoint": str(checkpoint_path), "split": args.split, "results": results},
            fh,
            indent=2,
        )
    log.info(f"-> {out_path}")


if __name__ == "__main__":
    main()
