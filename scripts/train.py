"""Training entry point.

Examples:
    uv run scripts/train.py
        CFM-equivalent baseline: CLIP-DINOiser, no upsampling, ReLU activation,
        reconstruction+auxk losses only. The SAE sees the frozen encoder's patch
        features directly, exactly as CFM does.

    uv run scripts/train.py encoder=dinov3 sae.activation=sigmoid
    uv run scripts/train.py upsampler=anyup upsampler.target_resolution=64 loss.scale_spatial.weight=1.0
    uv run scripts/train.py train.cache.mode=extract_and_cache train.epochs=5

    # S2AE-style attention-based grouping (needs the cache built first):
    #   uv run scripts/extract_attention_groups.py
    uv run scripts/train.py loss.group_sparsity.grouping=attention loss.group_sparsity.weight=0.3
"""

from __future__ import annotations

import logging
import math
import subprocess
import sys
from pathlib import Path

import hydra
import hydra.utils
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

import wandb
from cbm_msae_lab.activation_loader import ActivationLoader, build_activation_loader
from cbm_msae_lab.attention_grouping import CachedGroupLabelDataset, compute_group_cache_key
from cbm_msae_lab.checkpointing import save_checkpoint
from cbm_msae_lab.config_schema import register_configs
from cbm_msae_lab.early_stopping import EarlyStopping
from cbm_msae_lab.encoders.base import Encoder
from cbm_msae_lab.external_encoder import build_external_encoder, external_embed
from cbm_msae_lab.grouping import (
    CLUSTERED_GROUPINGS,
    clustering_method,
    needs_attention_grouping,
    resolve_group_labels,
    resolve_region_grouping,
)
from cbm_msae_lab.metrics.fvu import FVUMetric
from cbm_msae_lab.metrics.monosemanticity import MonosemanticityScore, PachMonosemanticityScore
from cbm_msae_lab.metrics.reconstruction import ReconstructionMetric
from cbm_msae_lab.metrics.region_consistency import RegionConsistencyMetric
from cbm_msae_lab.metrics.sparsity import ActivationFrequencyMetric, L0Metric
from cbm_msae_lab.metrics.tversky_ms import TverskyMS
from cbm_msae_lab.pipeline import ConceptPipeline
from cbm_msae_lab.sae.trainer import ComposableLossTrainer
from cbm_msae_lab.timing import TimedLoader
from cbm_msae_lab.upsampling.base import Upsampler

register_configs()
log = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    return requested if requested == "cuda" and torch.cuda.is_available() else "cpu"


def build_pipeline_shell(cfg: DictConfig, device: str) -> tuple[Encoder, Upsampler]:
    """Builds everything except the SAE (which `ComposableLossTrainer` owns)."""
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    upsampler = hydra.utils.instantiate(cfg.upsampler).to(device)
    return encoder, upsampler


@torch.no_grad()
def infer_grid_shape(
    encoder: Encoder,
    upsampler: Upsampler,
    image_size: int,
    device: str,
) -> tuple[int, int]:
    """One dummy forward pass to find (H, W) of the SAE-input grid -- simpler and
    less error-prone than re-deriving it analytically for every
    encoder+upsampler combination (patch sizes, target resolutions, ...)."""
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    features = encoder(dummy).features
    if upsampler.stage == "features":
        features = upsampler(dummy, features)
    return features.shape[-2], features.shape[-1]


def build_datasets(cfg: DictConfig) -> tuple[Dataset, Dataset]:
    train_dataset = hydra.utils.instantiate(cfg.dataset, split="train")
    val_dataset = hydra.utils.instantiate(cfg.dataset, split="val")
    return train_dataset, val_dataset




def load_group_label_dataset(cfg: DictConfig, split: str, method: str) -> CachedGroupLabelDataset:
    """Raises FileNotFoundError (with a message pointing at
    scripts/extract_attention_groups.py) if the cache hasn't been built yet --
    clustered grouping never falls back to computing it live."""
    cache_key = compute_group_cache_key(
        encoder_cfg=cfg.encoder,
        n_clusters=cfg.loss.attention_grouping.n_clusters,
        spatial_coeff=cfg.loss.attention_grouping.spatial_coeff,
        dataset_cfg=cfg.dataset,
        split=split,
        method=method,
    )
    return CachedGroupLabelDataset(cfg.train.cache.dir, cache_key)


def region_group_count(cfg: DictConfig, grid: tuple[int, int]) -> int:
    """Number of regions the region-consistency metric partitions a patch grid into."""
    if resolve_region_grouping(cfg) == "tile":
        s = cfg.train.eval.region_tile_size
        return (grid[0] // s) * (grid[1] // s)
    return cfg.loss.attention_grouping.n_clusters


@torch.no_grad()
def run_eval(
    pipeline: ConceptPipeline,
    trainer: ComposableLossTrainer,
    val_loader: ActivationLoader,
    cfg: DictConfig,
    device: str,
    external_encoder: Encoder | None = None,
    val_image_dataset: Dataset | None = None,
    val_group_labels: CachedGroupLabelDataset | None = None,
) -> dict[str, float]:
    pipeline.eval()
    fvu_metric = FVUMetric(cfg.encoder.output_dim).to(device)
    recon_metric = ReconstructionMetric().to(device)
    l0_metric = L0Metric().to(device)
    frequency_metric = ActivationFrequencyMetric(cfg.sae.dict_size).to(device)
    region_metric = RegionConsistencyMetric(region_group_count(cfg, trainer.grid_shape)).to(device)

    expensive = cfg.train.eval.enable_expensive_metrics
    ms_metric = MonosemanticityScore(cfg.sae.dict_size).to(device) if expensive else None
    pach_ms_metric = PachMonosemanticityScore(cfg.sae.dict_size).to(device) if expensive else None
    tms_metric = TverskyMS(cfg.sae.dict_size).to(device) if expensive else None
    if expensive and (external_encoder is None or val_image_dataset is None):
        raise ValueError("enable_expensive_metrics=true needs external_encoder and val_image_dataset for MS")

    seen = 0
    for x_img, _labels, idx in val_loader:
        B, P, D = x_img.shape
        x_flat = x_img.reshape(B * P, D)

        f = trainer.ae.encode(x_flat, use_threshold=True)  # [B*P, dict_size]
        x_hat = trainer.ae.decode(f)  # [B*P, D]
        fvu_metric.update(x_flat, x_hat)
        recon_metric.update(x_flat, x_hat)
        l0_metric.update(f)
        frequency_metric.update(f)

        region_grouping = resolve_region_grouping(cfg)
        labels, _n_groups = resolve_group_labels(
            region_grouping,
            grid=trainer.grid_shape,
            tile_size=cfg.train.eval.region_tile_size,
            batch_size=B,
            device=device,
            group_labels=val_group_labels.get_batch(idx).to(device) if val_group_labels is not None else None,
            n_clusters=cfg.loss.attention_grouping.n_clusters,
            consumer="train.eval.region_grouping",
        )
        region_metric.update(f.reshape(B, P, -1), labels)

        if ms_metric is not None and tms_metric is not None:
            f_img = f.reshape(B, P, -1)
            concept_activations = f_img.max(
                dim=1
            ).values  # [B, dict_size] -- one activation strength per image per concept
            images = torch.stack([val_image_dataset[i][0] for i in idx.tolist()]).to(device)  # [B, 3, H, W]
            image_embeddings = external_embed(external_encoder, images)  # [B, embed_dim], L2-normalized
            ms_metric.update(concept_activations, image_embeddings)
            pach_ms_metric.update(concept_activations, image_embeddings)
            tms_metric.update(concept_activations)

        seen += B
        if seen >= cfg.train.eval.max_eval_samples:
            break

    results = {"val/fvu": fvu_metric.compute().item()}
    for name, value in recon_metric.compute().items():
        results[f"val/{name}"] = value.item()
    results["val/l0"] = l0_metric.compute().item()
    results.update(frequency_metric.summary(prefix="val/"))
    for name, value in region_metric.compute().items():
        results[f"val/{name}"] = value.item()
    if ms_metric is not None and tms_metric is not None:
        results["val/monosemanticity_score"] = ms_metric.compute().nanmean().item()
        results["val/monosemanticity_score_pach"] = pach_ms_metric.compute().nanmean().item()
        results["val/dead_latent_fraction"] = ms_metric.dead_fraction()
        results["val/tversky_ms"] = tms_metric.compute().item()

    pipeline.train()
    return results


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = resolve_device(cfg.train.device)
    torch.manual_seed(cfg.seed)
    log.info(f"device={device}\n{OmegaConf.to_yaml(cfg)}")

    train_dataset, val_dataset = build_datasets(cfg)
    steps_per_epoch = math.ceil(len(train_dataset) / cfg.train.batch_size)
    total_steps = steps_per_epoch * cfg.train.epochs

    encoder, upsampler = build_pipeline_shell(cfg, device)
    grid_shape = infer_grid_shape(encoder, upsampler, cfg.dataset.image_size, device)
    log.info(f"SAE-input grid shape: {grid_shape}")

    trainer = ComposableLossTrainer(
        grid_shape=grid_shape,
        loss_config=cfg.loss,
        activation=cfg.sae.activation,
        steps=total_steps,
        activation_dim=cfg.encoder.output_dim,
        dict_size=cfg.sae.dict_size,
        k=cfg.sae.k,
        layer=0,  # unused for a vision encoder; dictionary_learning's trainer expects a value
        lm_name="vision-encoder",
        group_fractions=list(cfg.sae.group_fractions),
        lr=cfg.sae.lr,
        warmup_steps=cfg.sae.warmup_steps,
        decay_start=cfg.sae.decay_start,
        threshold_beta=cfg.sae.threshold_beta,
        threshold_start_step=cfg.sae.threshold_start_step,
        seed=cfg.sae.seed,
        device=device,
    )

    # Nothing outside the SAE is trainable, so `trainer` owns the only optimizer.
    pipeline = ConceptPipeline(encoder, upsampler, trainer.ae).to(device)

    train_loader = TimedLoader(build_activation_loader(cfg, "train", train_dataset, pipeline, device))
    val_loader = build_activation_loader(cfg, "val", val_dataset, pipeline, device)

    train_group_labels = (
        load_group_label_dataset(cfg, "train", clustering_method(cfg)) if needs_attention_grouping(cfg) else None
    )
    # The val split needs a group-label cache only for the region-consistency
    # metric -- Group-TopK is training-only and the structural losses never run
    # during eval, so nothing else on the val path asks for one.
    region_grouping = resolve_region_grouping(cfg)
    val_group_labels = (
        load_group_label_dataset(cfg, "val", region_grouping) if region_grouping in CLUSTERED_GROUPINGS else None
    )
    external_encoder = (
        build_external_encoder(cfg.encoder._target_, device) if cfg.train.eval.enable_expensive_metrics else None
    )

    wandb.init(
        project=cfg.train.wandb.project,
        name=cfg.train.wandb.run_name,
        mode=cfg.train.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = (
        EarlyStopping(patience=cfg.train.early_stopping.patience, min_delta=cfg.train.early_stopping.min_delta)
        if cfg.train.early_stopping.enabled
        else None
    )
    monitor = cfg.train.early_stopping.monitor
    best_monitored = float("inf")
    best_checkpoint_path = checkpoint_dir / "best.pt"

    step = 0
    for epoch in range(cfg.train.epochs):
        pipeline.train()
        pbar = tqdm(train_loader, total=steps_per_epoch, desc=f"epoch {epoch}")
        for x_img, _labels, idx in pbar:
            if train_group_labels is not None:
                trainer.group_labels = train_group_labels.get_batch(idx).to(device)

            loss_value = trainer.update(step, x_img)

            pbar.set_postfix(
                loss=f"{loss_value:.4f}",
                wait_ms=f"{train_loader.data_wait_s * 1000:.0f}",
                compute_ms=f"{train_loader.compute_s * 1000:.0f}",
            )

            if step % cfg.train.log_every_n_steps == 0:
                logs = {
                    "train/loss": loss_value,
                    "train/epoch": epoch,
                    "train/dataloader_wait_ms": train_loader.data_wait_s * 1000,
                    "train/dataloader_compute_ms": train_loader.compute_s * 1000,
                    "train/dataloader_bottleneck_frac": train_loader.bottleneck_fraction,
                }
                for name, value in trainer.get_logging_parameters().items():
                    logs[f"train/{name}"] = value
                for name, value in trainer.last_per_loss_values.items():
                    logs[f"train/loss_{name}"] = value
                wandb.log(logs, step=step)
            step += 1

        is_last_epoch = epoch == cfg.train.epochs - 1
        should_stop = False
        if (epoch + 1) % cfg.train.eval.every_n_epochs == 0 or is_last_epoch:
            eval_logs = run_eval(
                pipeline,
                    trainer,
                    val_loader,
                    cfg,
                    device,
                    external_encoder=external_encoder,
                    val_image_dataset=val_dataset,
                    val_group_labels=val_group_labels,
            )
            wandb.log(eval_logs, step=step)
            log.info(f"epoch {epoch}: {eval_logs}")

            if eval_logs[monitor] < best_monitored:
                best_monitored = eval_logs[monitor]
                save_checkpoint(
                    str(best_checkpoint_path),
                    pipeline,
                    trainer,
                    cfg,
                    step,
                    epoch,
                    wandb_run_id=wandb.run.id,
                )
                log.info(f"new best checkpoint ({monitor}={best_monitored:.6g}): {best_checkpoint_path}")

            if early_stopping is not None and early_stopping.step(eval_logs[monitor]):
                log.info(
                    f"early stopping at epoch {epoch}: {monitor} hasn't improved by >= "
                    f"{cfg.train.early_stopping.min_delta:.1%} for {cfg.train.early_stopping.patience} "
                    f"eval checks (best {monitor}={early_stopping.best:.6g})"
                )
                should_stop = True

        if (epoch + 1) % cfg.train.checkpoint_every_n_epochs == 0 or is_last_epoch or should_stop:
            path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(str(path), pipeline, trainer, cfg, step, epoch, wandb_run_id=wandb.run.id)
            log.info(f"saved checkpoint: {path}")

        if should_stop:
            break

    wandb.finish()

    if best_checkpoint_path.exists():
        log.info(f"running downstream task-accuracy probe on the best checkpoint: {best_checkpoint_path}")
        subprocess.run(
            [sys.executable, "scripts/evaluate_task_accuracy.py", "--checkpoint", str(best_checkpoint_path)],
            check=True,
        )


if __name__ == "__main__":
    main()
