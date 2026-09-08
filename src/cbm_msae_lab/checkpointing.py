"""Single-file checkpoints: weights + optimizer state + the FULL resolved config.

Unlike CFM (which saves SAE weights and a hyperparameter JSON as two separate
files), every checkpoint here is one ``torch.save``'d dict containing
everything needed to exactly reproduce or resume a run. The frozen encoder's
weights are deliberately *not* included -- they're either a fixed pretrained
download (CLIP-DINOiser) or auto-fetched by name (DINOv3 via timm), so the
config alone (``checkpoint_path`` / ``model_name``) is enough to rebuild them
identically without bloating every checkpoint with a large frozen backbone.
"""

from __future__ import annotations

from typing import Any

import hydra.utils
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer

from cbm_msae_lab.pipeline import ConceptPipeline
from cbm_msae_lab.sae.model import ConfigurableActivationSAE
from cbm_msae_lab.sae.trainer import ComposableLossTrainer


def save_checkpoint(
    path: str,
    pipeline: ConceptPipeline,
    trainer: ComposableLossTrainer,
    projection_optimizer: Optimizer,
    cfg: DictConfig,
    step: int,
    epoch: int,
) -> None:
    checkpoint: dict[str, Any] = {
        "model_state_dict": {
            "projection": pipeline.projection.state_dict(),
            "sae": pipeline.sae.state_dict(),
        },
        "optimizer_state_dict": {
            "sae": trainer.optimizer.state_dict(),
            "projection": projection_optimizer.state_dict(),
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
        "step": step,
        "epoch": epoch,
    }
    torch.save(checkpoint, path)


def load_for_reproduction(path: str, device: str = "cpu") -> tuple[DictConfig, ConceptPipeline]:
    """Rebuilds the exact pipeline (encoder + projection + upsampler + SAE) a
    checkpoint was trained with, from that checkpoint alone. Returns the
    resolved config too, e.g. to re-launch the identical training run.
    """
    checkpoint = torch.load(path, map_location=device)
    cfg = OmegaConf.create(checkpoint["config"])

    encoder = hydra.utils.instantiate(cfg.encoder)
    upsampler = hydra.utils.instantiate(cfg.upsampler)

    projection = hydra.utils.instantiate(cfg.projection)
    projection.load_state_dict(checkpoint["model_state_dict"]["projection"])

    sae: ConfigurableActivationSAE = ConfigurableActivationSAE.from_state_dict(checkpoint["model_state_dict"]["sae"])

    pipeline = ConceptPipeline(encoder, projection, upsampler, sae).to(device)
    pipeline.eval()
    return cfg, pipeline


def load_optimizer_states(
    path: str,
    trainer: ComposableLossTrainer,
    projection_optimizer: Optimizer,
    device: str = "cpu",
) -> tuple[int, int]:
    """For resuming training (not needed for pure inference/reproduction).
    Returns (step, epoch) to resume from."""
    checkpoint = torch.load(path, map_location=device)
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"]["sae"])
    projection_optimizer.load_state_dict(checkpoint["optimizer_state_dict"]["projection"])
    return checkpoint["step"], checkpoint["epoch"]
