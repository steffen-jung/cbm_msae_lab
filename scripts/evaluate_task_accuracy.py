"""Downstream task-accuracy evaluation for a finished checkpoint.

How much of CUB's 200-way classification signal survives the SAE's concept
bottleneck, compared to the raw (un-bottlenecked) feature it was trained to
reconstruct? Four per-image-pooled representations are extracted from the
*same* checkpoint and each gets its own linear probe (see `probing.py` for
the exact CFM-paper protocol):

    f_raw   the frozen encoder's own patch feature. This is exactly CFM's F+,
            and it is what the published CFM numbers were measured on, so it
            is the regression test that the encoder path is intact: on CUB with
            CLIP-DINOiser it must land at ~73% top-1 / ~93% top-5
            (`cfm_finegrained/results/phase1_residual_cub.json`, `f_plus_max`).
    f_plus  the feature the SAE actually sees -- its reconstruction target, and
            the ceiling: no bottleneck at all. Identical to `f_raw` unless a
            feature-stage upsampler is configured, in which case only the
            upsampling separates them; when they are identical it is skipped
            rather than probed twice.
    a       the SAE's dense (unsparsified) concept activation.
    z       the SAE's inference-time thresholded (sparse) concept activation
            -- this is what CFM actually calls its "concepts".

This script only ever reads a *finished* checkpoint
(`checkpointing.load_for_reproduction`) -- run it after training, never
during. It carries no Hydra config of its own: the checkpoint already stores
the exact resolved config it was trained under, which is all that's needed to
rebuild the encoder/upsampler/SAE and the matching CUB splits.

Example:
    uv run scripts/evaluate_task_accuracy.py --checkpoint outputs/checkpoints/epoch_0049.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import hydra.utils
import numpy as np
import numpy.typing as npt
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cbm_msae_lab.checkpointing import load_for_reproduction
from cbm_msae_lab.pipeline import ConceptPipeline
from cbm_msae_lab.probing import ProbeConfig, ProbeResult, run_probe

log = logging.getLogger(__name__)

REPRESENTATIONS: tuple[str, ...] = ("f_raw", "f_plus", "a", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="path to a checkpoint written by scripts/train.py")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=64, help="feature-extraction batch size (not the probe's)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out", default=None, help="output JSON path (default: outputs/task_accuracy/<checkpoint-stem>.json)"
    )
    return parser.parse_args()


@torch.no_grad()
def extract_pooled_representations(
    pipeline: ConceptPipeline, dataset: Dataset, batch_size: int, device: str, split: str
) -> tuple[dict[str, Tensor], npt.NDArray[np.int64]]:
    """One forward pass over `dataset`; returns per-image max-pooled
    f_raw/f_plus/a/z representations (in dataset order) plus the int64 class labels.

    Uses `pipeline.project` + a manual `sae.encode(..., return_active=True)`
    instead of `pipeline.forward` so both the dense (`a`) and thresholded
    (`z`) activations come out of a single encode call -- `pipeline.forward`
    only ever exposes the thresholded one, and decoding a reconstruction we
    don't need here would be wasted compute.
    """
    loader: DataLoader[tuple[Tensor, int, int]] = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    f_raw_chunks: list[Tensor] = []
    f_plus_chunks: list[Tensor] = []
    a_chunks: list[Tensor] = []
    z_chunks: list[Tensor] = []
    label_chunks: list[Tensor] = []

    for images, labels, _idx in tqdm(loader, desc=f"extract[{split}]", unit="batch"):
        images = images.to(device)  # [B, 3, H_img, W_img]
        raw = pipeline.encoder(images).features  # [B, C_enc, H0, W0] -- CFM's F+
        features = pipeline.project(images)  # [B, C_sae, H, W]
        B, C, H, W = features.shape
        x_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*P, C_sae], P = H*W

        z_flat, _active, a_flat = pipeline.sae.encode(x_flat, return_active=True, use_threshold=True)
        # z_flat, a_flat: [B*P, dict_size] -- thresholded ("z") and dense ("a") activations
        dict_size = z_flat.shape[-1]

        f_plus_img = features.reshape(B, C, H * W)  # [B, C_sae, P]
        a_img = a_flat.reshape(B, H * W, dict_size).permute(0, 2, 1)  # [B, dict_size, P]
        z_img = z_flat.reshape(B, H * W, dict_size).permute(0, 2, 1)  # [B, dict_size, P]

        f_raw_chunks.append(raw.flatten(2).max(dim=-1).values.cpu())  # [B, C_enc]
        f_plus_chunks.append(f_plus_img.max(dim=-1).values.cpu())  # [B, C_sae]
        a_chunks.append(a_img.max(dim=-1).values.cpu())  # [B, dict_size]
        z_chunks.append(z_img.max(dim=-1).values.cpu())  # [B, dict_size]
        label_chunks.append(labels)

    reps = {
        "f_raw": torch.cat(f_raw_chunks, dim=0),  # [N, C_enc]
        "f_plus": torch.cat(f_plus_chunks, dim=0),  # [N, C_sae]
        "a": torch.cat(a_chunks, dim=0),  # [N, dict_size]
        "z": torch.cat(z_chunks, dim=0),  # [N, dict_size]
    }
    labels_all = torch.cat(label_chunks, dim=0).numpy().astype(np.int64)  # [N]
    return reps, labels_all


def build_combined_representations(
    cfg: DictConfig, pipeline: ConceptPipeline, batch_size: int, device: str
) -> tuple[dict[str, Tensor], npt.NDArray[np.int64], npt.NDArray[np.bool_], npt.NDArray[np.bool_], npt.NDArray[np.bool_], int]:
    """`CUBDataset` builds train/val/test as three separate instances (each with
    its own internal split mask), so there's no single combined index to read
    representations against -- this extracts each split independently and
    concatenates them into one [N_total, dim] array per representation, with
    boolean masks marking which rows belong to which split (the interface
    `probing.run_probe` expects, matching `cfm_finegrained`'s own convention).
    """
    per_split: dict[str, tuple[dict[str, Tensor], npt.NDArray[np.int64], int]] = {}
    n_classes = 0
    for split in ("train", "val", "test"):
        dataset = hydra.utils.instantiate(cfg.dataset, split=split)
        n_classes = len(dataset.class_names)
        reps, labels = extract_pooled_representations(pipeline, dataset, batch_size, device, split)
        per_split[split] = (reps, labels, len(dataset))
        log.info(f"[{split}] extracted {len(dataset)} images")

    combined_reps = {
        name: torch.cat([per_split[s][0][name] for s in ("train", "val", "test")], dim=0) for name in REPRESENTATIONS
    }
    labels = np.concatenate([per_split[s][1] for s in ("train", "val", "test")])

    n_train, n_val, n_test = (per_split[s][2] for s in ("train", "val", "test"))
    n_total = n_train + n_val + n_test
    train_mask = np.zeros(n_total, dtype=np.bool_)
    train_mask[:n_train] = True
    val_mask = np.zeros(n_total, dtype=np.bool_)
    val_mask[n_train : n_train + n_val] = True
    test_mask = np.zeros(n_total, dtype=np.bool_)
    test_mask[n_train + n_val :] = True

    return combined_reps, labels, train_mask, val_mask, test_mask, n_classes


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    cfg, pipeline = load_for_reproduction(str(checkpoint_path), device=args.device)
    pipeline.eval()

    reps, labels, train_mask, val_mask, test_mask, n_classes = build_combined_representations(
        cfg, pipeline, args.batch_size, args.device
    )
    log.info(
        f"{len(labels)} images total, {n_classes} classes | "
        f"train {int(train_mask.sum())} / val {int(val_mask.sum())} / test {int(test_mask.sum())}"
    )

    to_probe = list(REPRESENTATIONS)
    if torch.equal(reps["f_raw"], reps["f_plus"]):
        # No feature-stage upsampler: the SAE's target *is* the encoder feature,
        # so probing both would spend the whole grid twice on the same numbers.
        to_probe.remove("f_plus")
        log.info("f_plus == f_raw (no feature-stage upsampler) -- probing f_raw only")

    probe_cfg = ProbeConfig()
    device = torch.device(args.device)
    results: list[ProbeResult] = []
    for name in to_probe:
        result = run_probe(
            name=name,
            x=reps[name],
            labels=labels,
            train=train_mask,
            val=val_mask,
            test=test_mask,
            n_classes=n_classes,
            cfg=probe_cfg,
            device=device,
            seeds=args.seeds,
        )
        results.append(result)
        log.info(
            f"  {name:8s} dim={result.dim:5d}  top1={100 * result.test_top1_mean:5.2f}"
            f"±{100 * result.test_top1_std:4.2f}  top5={100 * result.test_top5_mean:5.2f}  "
            f"(lr={result.selected_lr:g}, lam={result.selected_lambda:g})"
        )

    out_path = Path(args.out) if args.out else Path("outputs/task_accuracy") / f"{checkpoint_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "probe_config": asdict(probe_cfg),
                "seeds": args.seeds,
                "results": [r.as_dict() for r in results],
            },
            fh,
            indent=2,
        )
    log.info(f"-> {out_path}")


if __name__ == "__main__":
    main()
