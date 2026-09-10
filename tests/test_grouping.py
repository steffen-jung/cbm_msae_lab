"""Unit tests for `resolve_region_grouping`'s "auto" resolution.

Region consistency should measure whatever patch clusters a run's own
structural loss actually trains with, not an unrelated fixed partition -- these
pin the resolution logic against the config shapes it has to read.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from cbm_msae_lab.config_schema import LossConfig, TrainConfig
from cbm_msae_lab.grouping import resolve_region_grouping


def _cfg():
    """A minimal stand-in for the Hydra config: `resolve_region_grouping` only
    ever reads `cfg.loss.*` and `cfg.train.eval.region_grouping`. A real
    `DictConfig` (not a plain namespace), because `active_grouping_consumers`
    reads through `OmegaConf.select`, which needs OmegaConf's own node types."""
    return OmegaConf.structured({"loss": LossConfig(), "train": TrainConfig()})


def test_auto_falls_back_to_tile_when_nothing_clusters() -> None:
    """The plain BatchTopK baseline has no structural loss at all -- "auto" must
    not invent a clustering requirement it doesn't have."""
    cfg = _cfg()
    cfg.train.eval.region_grouping = "auto"
    assert resolve_region_grouping(cfg) == "tile"


def test_auto_follows_an_active_group_topk() -> None:
    cfg = _cfg()
    cfg.train.eval.region_grouping = "auto"
    cfg.loss.group_topk.enabled = True
    cfg.loss.group_topk.grouping = "attention"
    assert resolve_region_grouping(cfg) == "attention"


def test_auto_follows_an_active_participation_ratio_loss() -> None:
    cfg = _cfg()
    cfg.train.eval.region_grouping = "auto"
    cfg.loss.participation_ratio.weight = 0.1
    cfg.loss.participation_ratio.grouping = "feature"
    assert resolve_region_grouping(cfg) == "feature"


def test_auto_ignores_a_clustered_grouping_on_a_disabled_loss() -> None:
    """A `grouping=attention` sitting on a `weight=0.0` loss is not actually
    used -- "auto" must not pull in a cache requirement for it."""
    cfg = _cfg()
    cfg.train.eval.region_grouping = "auto"
    cfg.loss.exclusivity.grouping = "attention"
    cfg.loss.exclusivity.weight = 0.0
    assert resolve_region_grouping(cfg) == "tile"


def test_auto_raises_when_active_losses_disagree() -> None:
    cfg = _cfg()
    cfg.train.eval.region_grouping = "auto"
    cfg.loss.group_topk.enabled = True
    cfg.loss.group_topk.grouping = "attention"
    cfg.loss.participation_ratio.weight = 0.1
    cfg.loss.participation_ratio.grouping = "feature"
    with pytest.raises(ValueError, match="disagree on clustering method"):
        resolve_region_grouping(cfg)


@pytest.mark.parametrize("explicit", ["tile", "attention", "feature"])
def test_explicit_value_always_wins_over_auto_resolution(explicit: str) -> None:
    """An explicit override forces one fixed partition across every ablation
    arm -- it must never be second-guessed by what a loss happens to train with."""
    cfg = _cfg()
    cfg.train.eval.region_grouping = explicit
    cfg.loss.group_topk.enabled = True
    cfg.loss.group_topk.grouping = "attention" if explicit != "attention" else "feature"
    assert resolve_region_grouping(cfg) == explicit


def test_auto_treats_a_missing_loss_field_as_absent_not_an_error() -> None:
    """A checkpoint trained before a loss field existed has no such key in its
    stored config at all -- `active_grouping_consumers` must read that as "not
    configured" (equivalent to weight=0.0) rather than raise, since post-hoc
    scripts (`evaluate_checkpoint.py`) run this against arbitrarily old
    checkpoints. Reproduced the way a real checkpoint's config actually looks:
    `checkpointing.load_for_reproduction` builds it as a plain (non-structured)
    `OmegaConf.create(dict)` from whatever was resolved at train time, so a
    field added later is simply absent from that dict -- not present-but-None.
    """
    plain = OmegaConf.to_container(_cfg(), resolve=True)
    del plain["loss"]["participation_ratio"]
    del plain["loss"]["group_topk"]
    cfg = OmegaConf.create(plain)
    assert resolve_region_grouping(cfg) == "tile"
