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

from cbm_msae_lab.checkpointing import load_for_reproduction
from cbm_msae_lab.external_encoder import build_external_encoder, external_embed
from cbm_msae_lab.metrics.fvu import FVUMetric
from cbm_msae_lab.metrics.monosemanticity import MonosemanticityScore, PachMonosemanticityScore
from cbm_msae_lab.metrics.reconstruction import ReconstructionMetric
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
) -> dict[str, float]:
    pipeline.eval()
    fvu_metric = FVUMetric(pipeline.encoder.output_dim).to(device)
    recon_metric = ReconstructionMetric().to(device)
    ms_metric = MonosemanticityScore(pipeline.sae.dict_size).to(device) if external_encoder is not None else None
    pach_ms_metric = (
        PachMonosemanticityScore(pipeline.sae.dict_size).to(device) if external_encoder is not None else None
    )
    tms_metric = TverskyMS(pipeline.sae.dict_size).to(device) if external_encoder is not None else None

    seen = 0
    for images, _labels, _idx in loader:
        images = images.to(device)
        out = pipeline.forward(images, use_threshold=True)
        B, C, H, W = out.features.shape

        x_flat = flatten_spatial(out.features).reshape(B * H * W, C)
        x_hat_flat = flatten_spatial(out.reconstruction).reshape(B * H * W, C)
        fvu_metric.update(x_flat, x_hat_flat)
        recon_metric.update(x_flat, x_hat_flat)

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
    if ms_metric is not None and tms_metric is not None:
        results["monosemanticity_score"] = ms_metric.compute().mean().item()
        results["monosemanticity_score_pach"] = pach_ms_metric.compute().nanmean().item()
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

    results = evaluate(pipeline, loader, args.device, args.max_samples, external_encoder)
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
