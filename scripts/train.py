"""Training entry point.

Examples:
    uv run scripts/train.py
        CFM-equivalent baseline: CLIP-DINOiser, 3x3 projection, no upsampling,
        ReLU activation, reconstruction+auxk losses only.

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
from pathlib import Path

import hydra
import hydra.utils
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from cbm_msae_lab.activation_loader import ActivationLoader, build_activation_loader
from cbm_msae_lab.attention_grouping import CachedGroupLabelDataset, compute_group_cache_key
from cbm_msae_lab.checkpointing import save_checkpoint
from cbm_msae_lab.config_schema import register_configs
from cbm_msae_lab.encoders.base import Encoder
from cbm_msae_lab.metrics.fvu import FVUMetric
from cbm_msae_lab.metrics.monosemanticity import MonosemanticityScore
from cbm_msae_lab.metrics.reconstruction import ReconstructionMetric
from cbm_msae_lab.metrics.tversky_ms import TverskyMS
from cbm_msae_lab.pipeline import ConceptPipeline
from cbm_msae_lab.projection import EncoderProjection
from cbm_msae_lab.sae.trainer import ComposableLossTrainer
from cbm_msae_lab.timing import TimedLoader
from cbm_msae_lab.upsampling.base import Upsampler

register_configs()
log = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    return requested if requested == "cuda" and torch.cuda.is_available() else "cpu"


def build_pipeline_shell(cfg: DictConfig, device: str) -> tuple[Encoder, EncoderProjection, Upsampler]:
    """Builds everything except the SAE (which `ComposableLossTrainer` owns)."""
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    projection = hydra.utils.instantiate(cfg.projection).to(device)
    upsampler = hydra.utils.instantiate(cfg.upsampler).to(device)
    return encoder, projection, upsampler


@torch.no_grad()
def infer_grid_shape(
    encoder: Encoder,
    projection: EncoderProjection,
    upsampler: Upsampler,
    image_size: int,
    device: str,
) -> tuple[int, int]:
    """One dummy forward pass to find (H, W) of the SAE-input grid -- simpler and
    less error-prone than re-deriving it analytically for every
    encoder+upsampler combination (patch sizes, target resolutions, ...)."""
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    features = projection(encoder(dummy).features)
    if upsampler.stage == "features":
        features = upsampler(dummy, features)
    return features.shape[-2], features.shape[-1]


def build_datasets(cfg: DictConfig) -> tuple[Dataset, Dataset]:
    train_dataset = hydra.utils.instantiate(cfg.dataset, split="train")
    val_dataset = hydra.utils.instantiate(cfg.dataset, split="val")
    return train_dataset, val_dataset


def needs_attention_grouping(cfg: DictConfig) -> bool:
    return cfg.loss.group_sparsity.grouping == "attention" or cfg.loss.exclusivity.grouping == "attention"


def load_group_label_dataset(cfg: DictConfig, split: str) -> CachedGroupLabelDataset:
    """Raises FileNotFoundError (with a message pointing at
    scripts/extract_attention_groups.py) if the cache hasn't been built yet --
    attention-based grouping never falls back to computing clustering live."""
    cache_key = compute_group_cache_key(
        encoder_cfg=cfg.encoder,
        n_clusters=cfg.loss.attention_grouping.n_clusters,
        spatial_coeff=cfg.loss.attention_grouping.spatial_coeff,
        dataset_name=cfg.dataset._target_,
        split=split,
        image_size=cfg.dataset.image_size,
    )
    return CachedGroupLabelDataset(cfg.train.cache.dir, cache_key)


@torch.no_grad()
def run_eval(
    pipeline: ConceptPipeline,
    trainer: ComposableLossTrainer,
    val_loader: ActivationLoader,
    cfg: DictConfig,
    device: str,
) -> dict[str, float]:
    pipeline.eval()
    fvu_metric = FVUMetric(cfg.projection.out_channels).to(device)
    recon_metric = ReconstructionMetric().to(device)

    expensive = cfg.train.eval.enable_expensive_metrics
    ms_metric = MonosemanticityScore(cfg.sae.dict_size).to(device) if expensive else None
    tms_metric = TverskyMS(cfg.sae.dict_size).to(device) if expensive else None

    seen = 0
    for x_img, _labels, _idx in val_loader:
        B, P, D = x_img.shape
        x_flat = x_img.reshape(B * P, D)

        f = trainer.ae.encode(x_flat, use_threshold=True)  # [B*P, dict_size]
        x_hat = trainer.ae.decode(f)  # [B*P, D]
        fvu_metric.update(x_flat, x_hat)
        recon_metric.update(x_flat, x_hat)

        if ms_metric is not None and tms_metric is not None:
            f_img = f.reshape(B, P, -1)
            concept_activations = f_img.max(
                dim=1
            ).values  # [B, dict_size] -- one activation strength per image per concept
            image_embeddings = x_img.mean(
                dim=1
            )  # [B, D] -- mean-pooled encoder feature as this metric's "external embedding"
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            ms_metric.update(concept_activations, image_embeddings)
            tms_metric.update(concept_activations)

        seen += B
        if seen >= cfg.train.eval.max_eval_samples:
            break

    results = {"val/fvu": fvu_metric.compute().item()}
    for name, value in recon_metric.compute().items():
        results[f"val/{name}"] = value.item()
    if ms_metric is not None and tms_metric is not None:
        results["val/monosemanticity_score"] = ms_metric.compute().mean().item()
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

    encoder, projection, upsampler = build_pipeline_shell(cfg, device)
    grid_shape = infer_grid_shape(encoder, projection, upsampler, cfg.dataset.image_size, device)
    log.info(f"SAE-input grid shape: {grid_shape}")

    trainer = ComposableLossTrainer(
        grid_shape=grid_shape,
        loss_config=cfg.loss,
        activation=cfg.sae.activation,
        steps=total_steps,
        activation_dim=cfg.projection.out_channels,
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

    # `EncoderProjection` is trained jointly with the SAE, but with its own,
    # separate optimizer -- see pipeline.py's module docstring for why folding
    # it into dictionary_learning's own Adam/scheduler isn't done here.
    projection_optimizer = torch.optim.Adam(projection.parameters(), lr=trainer.lr)

    pipeline = ConceptPipeline(encoder, projection, upsampler, trainer.ae).to(device)

    train_loader = TimedLoader(build_activation_loader(cfg, "train", train_dataset, pipeline, device))
    val_loader = build_activation_loader(cfg, "val", val_dataset, pipeline, device)

    train_group_labels = load_group_label_dataset(cfg, "train") if needs_attention_grouping(cfg) else None

    wandb.init(
        project=cfg.train.wandb.project,
        name=cfg.train.wandb.run_name,
        mode=cfg.train.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(cfg.train.epochs):
        pipeline.train()
        pbar = tqdm(train_loader, total=steps_per_epoch, desc=f"epoch {epoch}")
        for x_img, _labels, idx in pbar:
            if train_group_labels is not None:
                trainer.group_labels = train_group_labels.get_batch(idx).to(device)

            projection_optimizer.zero_grad()
            loss_value = trainer.update(
                step, x_img
            )  # runs loss.backward() internally, populating projection's .grad too
            projection_optimizer.step()

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
        if (epoch + 1) % cfg.train.eval.every_n_epochs == 0 or is_last_epoch:
            eval_logs = run_eval(pipeline, trainer, val_loader, cfg, device)
            wandb.log(eval_logs, step=step)
            log.info(f"epoch {epoch}: {eval_logs}")

        if (epoch + 1) % cfg.train.checkpoint_every_n_epochs == 0 or is_last_epoch:
            path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(str(path), pipeline, trainer, projection_optimizer, cfg, step, epoch)
            log.info(f"saved checkpoint: {path}")

    wandb.finish()


if __name__ == "__main__":
    main()
