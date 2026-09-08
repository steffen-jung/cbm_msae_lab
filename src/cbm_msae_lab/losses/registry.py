"""Builds the dict of `Loss` objects a `ComposableLossTrainer` will sum."""

from __future__ import annotations

from cbm_msae_lab.config_schema import LossConfig
from cbm_msae_lab.losses.auxk import AuxKLoss
from cbm_msae_lab.losses.base import Loss
from cbm_msae_lab.losses.exclusivity import ExclusivityLoss
from cbm_msae_lab.losses.group_sparsity import GroupSparsityLoss
from cbm_msae_lab.losses.reconstruction import ReconstructionLoss
from cbm_msae_lab.losses.scale_spatial import ScaleSpatialLoss


def build_losses(cfg: LossConfig) -> dict[str, Loss]:
    """Every entry is always built (even with weight=0.0); `ComposableLossTrainer.loss()`
    skips calling `.compute()` on zero-weight entries, so a disabled loss costs nothing."""
    return {
        "reconstruction": ReconstructionLoss(cfg.reconstruction.weight),
        "auxk": AuxKLoss(cfg.auxk.weight),
        "scale_spatial": ScaleSpatialLoss(
            weight=cfg.scale_spatial.weight,
            scales=cfg.scale_spatial.scales,
            scale_weights=cfg.scale_spatial.scale_weights,
            alpha=cfg.scale_spatial.alpha,
            b_min=cfg.scale_spatial.b_min,
        ),
        "group_sparsity": GroupSparsityLoss(
            weight=cfg.group_sparsity.weight,
            grouping=cfg.group_sparsity.grouping,
            tile_size=cfg.group_sparsity.tile_size,
            n_clusters=cfg.attention_grouping.n_clusters,
        ),
        "exclusivity": ExclusivityLoss(
            weight=cfg.exclusivity.weight,
            grouping=cfg.exclusivity.grouping,
            tile_size=cfg.exclusivity.tile_size,
            n_clusters=cfg.attention_grouping.n_clusters,
        ),
    }
