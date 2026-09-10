"""One place that turns a ``grouping`` config string into per-patch group ids.

Three groupings exist in this codebase and they arrive by very different routes:
``"tile"`` is a fixed s x s partition of the patch grid, identical for every
image and derivable on the spot; ``"attention"`` and ``"feature"`` are per-image
clusterings that have to be precomputed and read from a cache
(``attention_grouping.py``, built by ``scripts/extract_attention_groups.py``).

The S2AE losses each carry their own copy of that branch. Group-TopK, the
participation-ratio loss and the region-consistency metric all need the same
thing, so they share this resolver instead -- otherwise "which grouping am I
looking at" would be answered in five slightly different places.
"""

from __future__ import annotations

from omegaconf import OmegaConf
from torch import Tensor

from cbm_msae_lab.losses.tiling import tile_group_labels

GROUPINGS = ("tile", "attention", "feature")
CLUSTERED_GROUPINGS = ("attention", "feature")


def validate_grouping(grouping: str, field: str) -> str:
    if grouping not in GROUPINGS:
        raise ValueError(f"unknown {field} grouping {grouping!r}; expected one of {list(GROUPINGS)}")
    return grouping


def resolve_group_labels(
    grouping: str,
    *,
    grid: tuple[int, int],
    tile_size: int,
    batch_size: int,
    device,
    group_labels: Tensor | None = None,
    n_clusters: int = 20,
    consumer: str = "this loss",
) -> tuple[Tensor, int]:
    """Returns `(labels [B, P] int64, n_groups)`.

    For `"tile"` the labels are computed from the grid and broadcast across the
    batch (every image gets the same partition). For the clustered groupings the
    caller must already have put this batch's cached labels on the context;
    `consumer` only shapes the error message when they are missing.
    """
    if grouping == "tile":
        grid_h, grid_w = grid
        labels = tile_group_labels(grid_h, grid_w, tile_size).to(device)  # [P]
        gh, gw = grid_h // tile_size, grid_w // tile_size
        return labels.unsqueeze(0).expand(batch_size, -1), gh * gw

    if group_labels is None:
        raise RuntimeError(
            f"{consumer} uses grouping={grouping!r} but no group_labels were supplied to this "
            f"training step -- build a cache with scripts/extract_attention_groups.py --method "
            f"{grouping} first."
        )
    return group_labels, n_clusters


def grouping_consumers(cfg) -> dict[str, str]:
    """Every config entry that resolves a `grouping` string, by name.

    Four of them: the two S2AE losses, the participation-ratio loss and
    Group-TopK. They all read the *same* `trainer.group_labels` tensor during
    training, which is why they are listed in one place instead of being
    checked pairwise.

    Reads through `OmegaConf.select` with a default rather than plain
    attribute access: a checkpoint's stored config is frozen at train time, so
    a checkpoint trained before e.g. `loss.participation_ratio` existed simply
    has no such key -- that must read as "not configured" (and, paired with
    `active_grouping_consumers` below, as "not active"), not raise.
    """
    return {
        "group_sparsity": OmegaConf.select(cfg, "loss.group_sparsity.grouping", default="tile"),
        "exclusivity": OmegaConf.select(cfg, "loss.exclusivity.grouping", default="tile"),
        "participation_ratio": OmegaConf.select(cfg, "loss.participation_ratio.grouping", default="tile"),
        "group_topk": OmegaConf.select(cfg, "loss.group_topk.grouping", default="tile"),
    }


def active_grouping_consumers(cfg) -> dict[str, str]:
    """Only the consumers that are actually switched on -- a `grouping=attention`
    sitting on a loss with `weight=0.0` must not drag a cache requirement in.
    A loss missing from an old checkpoint's config reads as `weight=0.0` /
    `enabled=False`, i.e. as if it had never been switched on (see
    `grouping_consumers`)."""
    enabled = {
        "group_sparsity": OmegaConf.select(cfg, "loss.group_sparsity.weight", default=0.0) != 0.0,
        "exclusivity": OmegaConf.select(cfg, "loss.exclusivity.weight", default=0.0) != 0.0,
        "participation_ratio": OmegaConf.select(cfg, "loss.participation_ratio.weight", default=0.0) != 0.0,
        "group_topk": bool(OmegaConf.select(cfg, "loss.group_topk.enabled", default=False)),
    }
    return {name: g for name, g in grouping_consumers(cfg).items() if enabled[name]}


def needs_attention_grouping(cfg) -> bool:
    return any(g in CLUSTERED_GROUPINGS for g in active_grouping_consumers(cfg).values())


def clustering_method(cfg) -> str:
    """All active grouping consumers share one cached `group_labels` tensor, so
    any that use a clustered grouping must agree on the method -- this returns
    it and raises if they disagree."""
    clustered = {name: g for name, g in active_grouping_consumers(cfg).items() if g in CLUSTERED_GROUPINGS}
    methods = set(clustered.values())
    if len(methods) > 1:
        listing = ", ".join(f"{name}.grouping={g!r}" for name, g in sorted(clustered.items()))
        raise ValueError(
            f"{listing} disagree on clustering method -- they all read the same cached "
            "group_labels, so they must match."
        )
    return next(iter(methods))


def clustered_or_tile(cfg) -> str:
    """The grouping a run's own structural loss would call for: whatever
    `clustering_method` returns if any grouping consumer is active, else
    `"tile"`. Shared by `resolve_region_grouping`'s `"auto"` case and by
    post-hoc scripts (e.g. `evaluate_checkpoint.py`'s own `--region-grouping
    auto`) that want that same loss-driven default computed fresh from a
    checkpoint's stored `loss` config -- rather than through `cfg.train.eval.
    region_grouping`, which is frozen at train time and, for a checkpoint
    trained before that field existed, may not even be present.
    """
    return clustering_method(cfg) if needs_attention_grouping(cfg) else "tile"


def resolve_region_grouping(cfg) -> str:
    """What `RegionConsistencyMetric` should actually group by, used by
    `scripts/train.py` during training/live eval (where `cfg.train.eval.
    region_grouping` is always present and current, since it is being read in
    the same process that composed it).

    `"auto"` (the default) measures region consistency on the *same* patch
    clusters a run's own structural loss trains with -- a fixed tile partition
    only tells you something meaningful when the run's own loss has no notion
    of "region" at all (`clustered_or_tile`'s fallback). Pass `"tile"`/
    `"attention"`/`"feature"` explicitly to force one fixed partition across
    every arm of an ablation for cross-run comparability instead.
    """
    configured = cfg.train.eval.region_grouping
    if configured != "auto":
        return configured
    return clustered_or_tile(cfg)
