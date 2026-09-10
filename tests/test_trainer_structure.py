"""Integration tests for how Group-TopK and the PR loss reach the training step.

These check the seam rather than the maths (`test_group_topk.py` /
`test_participation_ratio.py` cover that): does the mask actually reach `f`, does
everything downstream see the masked version, and does the PR loss show up in the
per-term breakdown the training script logs.
"""

from __future__ import annotations

import pytest
import torch

from cbm_msae_lab.config_schema import LossConfig
from cbm_msae_lab.sae.trainer import ComposableLossTrainer


def build_trainer(loss_config: LossConfig, grid: tuple[int, int] = (4, 4)) -> ComposableLossTrainer:
    return ComposableLossTrainer(
        grid_shape=grid,
        loss_config=loss_config,
        activation_dim=8,
        dict_size=32,
        k=6,
        group_fractions=[0.25, 0.75],
        layer=0,
        lm_name="test",
        steps=100,
        warmup_steps=10,
        device="cpu",
        seed=0,
    )


def _batch(batch_size: int = 3, grid: tuple[int, int] = (4, 4)) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch_size, grid[0] * grid[1], 8, generator=generator)


def test_group_topk_is_off_by_default() -> None:
    assert build_trainer(LossConfig()).group_topk is None


def test_group_topk_only_removes_activations() -> None:
    """Same batch, same weights, with and without the rule: enabling it may zero
    latents but must never create one, and must never change a surviving value."""
    x = _batch()

    baseline = build_trainer(LossConfig())
    structured = build_trainer(LossConfig())
    structured.ae.load_state_dict(baseline.ae.state_dict())
    structured.group_topk = build_trainer(_with_group_topk()).group_topk

    plain = baseline.loss(x, step=0, logging=True).f
    masked = structured.loss(x, step=0, logging=True).f

    assert ((masked != 0) <= (plain != 0)).all()
    assert (masked[masked != 0] == plain[masked != 0]).all()
    assert (masked != 0).sum() < (plain != 0).sum()  # it actually did something


def _with_group_topk(k_group: int = 4) -> LossConfig:
    cfg = LossConfig()
    cfg.group_topk.enabled = True
    cfg.group_topk.grouping = "tile"
    cfg.group_topk.tile_size = 2
    cfg.group_topk.k_group = k_group
    return cfg


def test_group_topk_caps_the_features_used_per_region() -> None:
    """The regional budget must hold in the latents the loss actually sees, not
    just in the mask: at most k_l distinct features per (image, region, level)."""
    trainer = build_trainer(_with_group_topk(k_group=4))
    f = trainer.loss(_batch(), step=0, logging=True).f.reshape(3, 16, 32)

    # tile_size 2 on a 4x4 grid -> four 2x2 regions of four patches each.
    regions = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
    blocks = trainer.ae.matryoshka_blocks
    ks = trainer.group_topk._ks(trainer.ae, blocks)

    for image in range(3):
        for patches in regions:
            for (start, end), k in zip(blocks, ks, strict=True):
                used = (f[image][patches][:, start:end] != 0).any(dim=0)
                assert int(used.sum()) <= k


def test_participation_ratio_reaches_the_per_loss_breakdown() -> None:
    cfg = LossConfig()
    cfg.participation_ratio.weight = 0.5
    trainer = build_trainer(cfg)

    log = trainer.loss(_batch(), step=0, logging=True)
    assert "participation_ratio" in log.losses
    assert log.losses["participation_ratio"] >= 1.0  # PR's floor is one region


def test_zero_weight_participation_ratio_is_never_computed() -> None:
    """The registry builds every loss unconditionally; a weight of 0 must still
    skip the compute() call, so an unbuilt group cache can't break a run that
    doesn't use it."""
    cfg = LossConfig()
    cfg.participation_ratio.grouping = "attention"  # would raise without group_labels
    trainer = build_trainer(cfg)

    log = trainer.loss(_batch(), step=0, logging=True)
    assert "participation_ratio" not in log.losses


def test_participation_ratio_without_group_labels_names_the_cache_script() -> None:
    cfg = LossConfig()
    cfg.participation_ratio.weight = 0.5
    cfg.participation_ratio.grouping = "feature"
    trainer = build_trainer(cfg)

    with pytest.raises(RuntimeError, match="extract_attention_groups.py"):
        trainer.loss(_batch(), step=0)
