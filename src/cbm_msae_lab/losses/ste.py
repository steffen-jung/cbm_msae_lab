"""Straight-through estimator (STE) for binarizing S2AE's group-activation
profiles before the group-sparsity/exclusivity losses (arXiv:2607.08605, Sec. 4.2):

    Z_hat = bin(Z) + Z - Z.detach()

The paper states this formula but doesn't specify `bin(.)`, and the public
S2AE repo's own reference implementation (github.com/liaoweiduo/s2ae,
`sae/trainer/sae_trainer.py`) uses the same "hard.detach() + (s - s.detach())"
construction verbatim. For non-negative `s` (true here: the group profiles
this is applied to are L2 norms) that construction is a mathematical no-op --
`d(hard + s - s.detach())/ds == 1` everywhere `s` appears, identical to the
subgradient of plain `s.abs()`. Verified directly:
`ste_binarize(s).sum().backward()` gives the same `grad == ones_like(s)` as
`s.abs().sum().backward()`, and an actual training run with the literal-STE
version reproduced the exact same threshold-collapse/dead-feature trajectory
as the un-binarized loss (see git history of this file).

So this decouples magnitude from feature *selection* only where the gradient
actually saturates: `s` is passed through `tanh` before the STE identity
term, not used raw. `d(tanh(s))/ds = 1 - tanh(s)^2` is close to 1 for `s`
near the decision boundary (0) -- matching plain L1 there -- but decays
toward 0 as `s` grows, so a latent that's clearly active stops being pushed
toward zero by this loss; only borderline latents (whose group-membership is
actually in question) keep getting nudged. This is the standard "saturating"
variant of a straight-through/binary-concrete estimator (as used for e.g.
binary neural networks), not a literal transcription of the paper's own
underspecified formula.
"""

from __future__ import annotations

import torch
from torch import Tensor


def ste_binarize(s: Tensor) -> Tensor:
    """`s` must be >= 0 (true for the L2-norm group profiles this is used on:
    `tile_l2_norm`/`onehot_l2_aggregate` are both sums of squares under a
    sqrt), so `bin(s) = 1[s > 0]` is the natural "did this group use this
    feature at all" gate."""
    hard = (s > 0).to(s.dtype)
    surrogate = torch.tanh(s)
    return hard + surrogate - surrogate.detach()
