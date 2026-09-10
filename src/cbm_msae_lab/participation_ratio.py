"""Participation ratio: over how many regions does a single feature spread its
activation mass?

For feature j of one image, with per-patch activations z_ij >= 0 and a group
assignment g(i):

    s_j    = sum_i z_ij                      (the feature's total mass)
    q_ij   = z_ij / s_j                       (sum_i q_ij = 1, exactly)
    m_gj   = sum_{i in g} q_ij                (sum_g m_gj = 1, exactly)
    PR_j   = 1 / sum_g m_gj^2

PR is the *effective number of groups*: mass in a single group gives PR = 1,
two equal groups give 2, three equal groups give 3. The square is what carries
the distribution -- sum_g m_gj is 1 by construction and says nothing.

Two properties this module exists to preserve exactly:

**Scale invariance.** q is defined with a plain division, no epsilon. Adding one
would break `q(alpha * z) = q(z)`, and with it the whole point: PR is homogeneous
of degree 0, so by Euler `<grad PR, z> = 0` -- the term has no radial component
and cannot shrink a feature's overall magnitude the way an L1/L2 sparsity
penalty does. It may only *redistribute* mass between groups. That matters here
because a shrinkage-biased objective has already cost this repo one silent
collapse (the removed encoder projection).

**Inactive features are excluded, not epsilon-ed.** A feature with s_j = 0 has no
defined q at all. It is masked out and never contributes to any mean, rather than
being pushed through a regularised division that would invent a PR for it.

Shared by `losses/participation_ratio.py` (as a loss) and
`metrics/region_consistency.py` (as a measurement), so the two can never drift.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class ParticipationRatio(NamedTuple):
    pr: Tensor  # [B, N] -- effective number of groups; meaningless where `active` is False
    active: Tensor  # [B, N] bool -- features with nonzero mass in this image
    dominant_share: Tensor  # [B, N] -- mass fraction held by the feature's single largest group


def participation_ratio(z_img: Tensor, group_labels: Tensor, n_groups: int) -> ParticipationRatio:
    """z_img: [B, P, N] non-negative activations, group_labels: [B, P] int64 in
    [0, n_groups) -> per-image, per-feature participation ratio.

    Computed per image: `P` is the patch count of one image, so "how many regions
    does this feature cover" is asked about that image's own regions. The group
    reduction is a `scatter_add` segment sum, never a Python loop over groups.
    """
    mass = z_img.sum(dim=1)  # [B, N]
    active = mass > 0  # [B, N]

    # No epsilon: the denominator is replaced by 1 exactly where the numerator is
    # identically zero, which leaves q = 0 there and keeps q(alpha*z) = q(z) exact
    # for every active feature. `active` is a non-differentiable bool, so gradient
    # still flows through `mass` for the active columns.
    safe_mass = torch.where(active, mass, torch.ones_like(mass))  # [B, N]
    q = z_img / safe_mass.unsqueeze(1)  # [B, P, N]

    # Segment sum via scatter_add_, not a one-hot matmul: this runs every training
    # step over the whole dictionary, where the one-hot form's B*P*G*N cost is
    # prohibitive against scatter_add_'s B*P*N. Differentiable in `q`.
    B, P, N = q.shape
    index = group_labels.unsqueeze(-1).expand(B, P, N)  # [B, P, N]
    group_mass = torch.zeros(B, n_groups, N, dtype=q.dtype, device=q.device)
    group_mass = group_mass.scatter_add(1, index, q)  # [B, G, N]

    sum_sq = group_mass.pow(2).sum(dim=1)  # [B, N]
    pr = 1.0 / torch.where(active, sum_sq, torch.ones_like(sum_sq))
    return ParticipationRatio(pr=pr, active=active, dominant_share=group_mass.max(dim=1).values)


def mean_over_active(values: Tensor, active: Tensor) -> Tensor:
    """Mean of `values` over the entries `active` marks, as a differentiable scalar.

    Returns 0 when nothing is active (an all-dead batch is not an error, it just
    carries no signal). Uses a masked sum rather than boolean indexing so the
    shape stays static and the result is safe to backprop through.
    """
    count = active.sum()
    if count == 0:
        return values.new_zeros(())
    return (values * active).sum() / count
