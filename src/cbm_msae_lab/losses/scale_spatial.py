"""The user's own scale-grounded loss, ported from ``cfm_finegrained/src/cfm_fg/scale_loss.py``.

CFM's plain Matryoshka loss ties which concept lands in which dictionary
prefix purely to what the optimizer finds convenient -- nothing connects a
coarse concept to a short prefix. This loss ties the two together spatially:
average the codes over an s-by-s tile of the patch grid, and a prefix of
length ``b_s = ceil(m * s^(-2*alpha))`` should suffice to reconstruct that
tile's average feature. Coarse region (large s) -> short prefix.

    L      = sum_s w_s * L_s
    L_s    = (1/|G_s|) sum_g (1/d) || Pi^-1(z_g[:b_s]) - F_bar_g ||^2
    z_g    = mean_{p in g} Pi(F_p)              (mean SAE latent over the tile)
    F_bar_g= mean_{p in g} F_p                  (mean input feature over the tile)
    Pi^-1(z[:b]) = z[:b] @ W_dec[:b] + b_dec     (decode using only the first b dict entries)

``s = 1`` (one patch per tile) is the plain per-patch full reconstruction,
i.e. `L_1` at `alpha=0` degenerates to the same loss as `ReconstructionLoss`'s
finest term.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from cbm_msae_lab.losses.base import Loss, LossContext
from cbm_msae_lab.losses.tiling import prefix_len, tile_mean


class ScaleSpatialLoss(Loss):
    def __init__(
        self,
        weight: float,
        scales: Sequence[int] = (1, 2, 7, 14),
        scale_weights: Sequence[float] | None = None,
        alpha: float = 1.0,
        b_min: int = 1,
    ) -> None:
        super().__init__(weight)
        if not scales:
            raise ValueError("need at least one scale")
        self.scales = tuple(scales)
        self.scale_weights = (
            tuple(scale_weights) if scale_weights is not None else tuple(1.0 / len(scales) for _ in scales)
        )
        if len(self.scale_weights) != len(self.scales):
            raise ValueError(f"{len(self.scale_weights)} weights for {len(self.scales)} scales")
        self.alpha = alpha
        self.b_min = b_min

    def compute(self, ctx: LossContext) -> Tensor:
        grid_h, grid_w = ctx.grid
        sae = ctx.sae
        d = ctx.x.shape[-1]

        total = torch.zeros((), device=ctx.x.device, dtype=ctx.x.dtype)
        for s, w in zip(self.scales, self.scale_weights):
            if grid_h % s != 0 or grid_w % s != 0:
                raise ValueError(f"scale s={s} does not evenly tile the {grid_h}x{grid_w} feature grid")
            b = prefix_len(sae.dict_size, s * s, self.alpha, self.b_min)

            if s == 1:
                rec = ctx.x_hat_img  # [B, P, d]
                target = ctx.x_img  # [B, P, d]
            else:
                z_g = tile_mean(ctx.f_img, grid_h, grid_w, s)  # [B, G_s, dict_size]
                target = tile_mean(ctx.x_img, grid_h, grid_w, s)  # [B, G_s, d]
                rec = z_g[..., :b] @ sae.W_dec[:b] + sae.b_dec  # [B, G_s, d]

            term = (rec - target).pow(2).sum(dim=-1).mean() / d
            total = total + w * term

        return total
