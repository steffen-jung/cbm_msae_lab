"""Straight-through estimator (STE) for binarizing S2AE's group-activation
profiles before the group-sparsity/exclusivity losses (arXiv:2607.08605, Sec. 4.2):

    Z_hat = bin(Z) + Z - Z.detach()

Forward pass: a hard binary gate (`bin(Z)`, since `Z - Z.detach()` is exactly
0 as a value). Backward pass: identity gradient (the comparison in `bin(.)`
and the `.detach()` both carry no gradient, so `d(Z_hat)/dZ == 1` everywhere).

This decouples the losses from activation *magnitude*: without it, `L_gs`/
`L_es` are minimized by shrinking the selected activations' magnitude toward
zero (exactly the "shrinkage bias" the paper names as the reason for
binarizing) rather than by actually dropping features from a group -- see
`group_sparsity.py`/`exclusivity.py` for how this showed up in practice
(BatchTopK's inference threshold collapsing toward 0, a growing share of dead
features).
"""

from __future__ import annotations

from torch import Tensor


def ste_binarize(s: Tensor) -> Tensor:
    """`s` must be >= 0 (true for the L2-norm group profiles this is used on:
    `tile_l2_norm`/`onehot_l2_aggregate` are both sums of squares under a
    sqrt), so `bin(s) = 1[s > 0]` is the natural "did this group use this
    feature at all" gate."""
    hard = (s > 0).to(s.dtype)
    return hard + s - s.detach()
