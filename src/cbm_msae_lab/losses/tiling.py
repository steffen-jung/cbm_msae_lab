"""Spatial-tile grouping shared by the scale/spatial loss and the S2AE losses.

Ported from ``cfm_finegrained/src/cfm_fg/scale_loss.py`` (``tile_mean``,
``prefix_len``), generalized from a square ``grid x grid`` assumption to a
rectangular ``grid_h x grid_w`` since an upsampled or DINOv3 feature map need
not be square-patched the way CLIP-DINOiser's 14x14 grid always was.
"""

from __future__ import annotations

from math import ceil

import torch
from torch import Tensor


def tile_mean(x: Tensor, grid_h: int, grid_w: int, s: int) -> Tensor:
    """Average-pools an s x s tile of the patch grid.

    x: [B, grid_h * grid_w, C] -> [B, (grid_h // s) * (grid_w // s), C]

    `s` must evenly divide both `grid_h` and `grid_w`, so every tile has
    exactly `s*s` member patches (this is what makes a scale's prefix length
    `b_s` constant across all of that scale's tiles).
    """
    if grid_h % s != 0 or grid_w % s != 0:
        raise ValueError(f"tile size s={s} does not evenly tile a {grid_h}x{grid_w} grid")
    B, P, C = x.shape
    if P != grid_h * grid_w:
        raise ValueError(f"{P} patches does not match grid {grid_h}x{grid_w}")
    gh, gw = grid_h // s, grid_w // s
    return x.reshape(B, gh, s, gw, s, C).mean(dim=(2, 4)).reshape(B, gh * gw, C)


def tile_l2_norm(x: Tensor, grid_h: int, grid_w: int, s: int) -> Tensor:
    """L2-aggregates an s x s tile of the patch grid, per channel (used for S2AE's group profile).

    x: [B, grid_h * grid_w, C] -> [B, (grid_h // s) * (grid_w // s), C],
    where each output value is sqrt(sum of squares over the s*s member patches).
    """
    if grid_h % s != 0 or grid_w % s != 0:
        raise ValueError(f"tile size s={s} does not evenly tile a {grid_h}x{grid_w} grid")
    B, P, C = x.shape
    if P != grid_h * grid_w:
        raise ValueError(f"{P} patches does not match grid {grid_h}x{grid_w}")
    gh, gw = grid_h // s, grid_w // s
    tiles = x.reshape(B, gh, s, gw, s, C)
    return tiles.pow(2).sum(dim=(2, 4)).sqrt().reshape(B, gh * gw, C)


def prefix_len(dict_size: int, group_size: int, alpha: float, b_min: int) -> int:
    """b = ceil(dict_size * group_size^(-alpha)), clamped to [b_min, dict_size].

    For the scale loss, `group_size = s*s` (an s x s tile has s*s member
    patches), so this computes `b_s = ceil(m * s^(-2*alpha))` exactly as in
    the loss's docstring formula.
    """
    b = ceil(dict_size * float(group_size) ** (-alpha))
    return int(min(max(b, b_min), dict_size))


def tile_group_labels(grid_h: int, grid_w: int, s: int) -> Tensor:
    """The same s x s partition `tile_mean`/`tile_l2_norm` pool over, expressed as
    a per-patch group id instead: [grid_h * grid_w] int64 in [0, (grid_h//s) * (grid_w//s)).

    Lets the tile grouping go through the same label-driven code path as the
    per-image attention/feature clusterings (see `grouping.py`), so Group-TopK
    and the participation-ratio loss need only one implementation.
    """
    if grid_h % s != 0 or grid_w % s != 0:
        raise ValueError(f"tile size s={s} does not evenly tile a {grid_h}x{grid_w} grid")
    gh, gw = grid_h // s, grid_w // s
    tile_ids = torch.arange(gh * gw, dtype=torch.int64).reshape(gh, 1, gw, 1)
    return tile_ids.expand(gh, s, gw, s).reshape(grid_h * grid_w)
