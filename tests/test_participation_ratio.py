"""Unit tests for the participation ratio and its loss.

The three worked examples from the specification are checked literally, and so
are the two properties the whole design rests on: exact scale invariance and the
absence of a radial gradient component.
"""

from __future__ import annotations

import pytest
import torch

from cbm_msae_lab.participation_ratio import mean_over_active, participation_ratio


def _labels(group_of_patch: list[int]) -> torch.Tensor:
    return torch.tensor(group_of_patch, dtype=torch.int64).unsqueeze(0)  # [1, P]


@pytest.mark.parametrize(
    ("mass_per_patch", "expected_pr"),
    [
        ([1.0, 0.0, 0.0, 0.0], 1.0),  # all mass in group 0
        ([0.5, 0.5, 0.0, 0.0], 2.0),  # two equal groups
        ([1 / 3, 1 / 3, 1 / 3, 0.0], 3.0),  # three equal groups
    ],
)
def test_pr_is_the_effective_number_of_groups(mass_per_patch: list[float], expected_pr: float) -> None:
    """The specification's own examples: m = (1,0,..) -> 1, (.5,.5,0,..) -> 2,
    (1/3,1/3,1/3,0,..) -> 3."""
    z = torch.tensor(mass_per_patch).reshape(1, 4, 1)  # one patch per group, one feature
    result = participation_ratio(z, _labels([0, 1, 2, 3]), n_groups=4)

    assert result.active.item() is True
    assert result.pr.item() == pytest.approx(expected_pr, abs=1e-5)


def test_pr_is_bounded_by_one_and_the_group_count() -> None:
    """Because sum_g m_gj == 1 exactly, PR is confined to [1, G]: one group at the
    bottom, perfectly uniform spread at the top."""
    generator = torch.Generator().manual_seed(0)
    z = torch.rand(3, 12, 7, generator=generator)
    labels = torch.randint(0, 4, (3, 12), generator=generator)

    pr = participation_ratio(z, labels, n_groups=4).pr
    assert (pr >= 1.0 - 1e-5).all()
    assert (pr <= 4.0 + 1e-5).all()


@pytest.mark.parametrize("alpha", [0.01, 0.5, 3.0, 1000.0])
def test_pr_is_exactly_scale_invariant(alpha: float) -> None:
    """PR(alpha * z) == PR(z). This is what an epsilon in the normalization would
    destroy, and with it the loss's freedom from shrinkage pressure."""
    generator = torch.Generator().manual_seed(1)
    z = torch.rand(2, 9, 5, generator=generator)
    labels = torch.randint(0, 3, (2, 9), generator=generator)

    base = participation_ratio(z, labels, n_groups=3).pr
    scaled = participation_ratio(alpha * z, labels, n_groups=3).pr
    assert torch.allclose(base, scaled, atol=1e-6)


def test_pr_gradient_has_no_radial_component() -> None:
    """Euler's theorem for a degree-0 homogeneous function: <grad L, z> == 0.

    This is the actual proof that the loss cannot systematically shrink
    activations -- it may only move mass between groups. Checked per feature, not
    just in aggregate, so a cancellation across features can't hide a per-feature
    radial pull.
    """
    generator = torch.Generator().manual_seed(2)
    z = (torch.rand(1, 8, 4, generator=generator) + 0.1).requires_grad_(True)
    labels = torch.randint(0, 3, (1, 8), generator=generator)

    result = participation_ratio(z, labels, n_groups=3)
    loss = mean_over_active(result.pr, result.active)
    (grad,) = torch.autograd.grad(loss, z)  # [1, 8, 4]

    radial_per_feature = (grad * z).sum(dim=1)  # <grad, z> along the patch axis, per feature
    assert torch.allclose(radial_per_feature, torch.zeros_like(radial_per_feature), atol=1e-5)


def test_inactive_features_are_excluded_rather_than_regularised() -> None:
    """A feature with zero mass has no defined q; it must be masked out of the
    mean, not pushed through a stabilised division."""
    z = torch.zeros(1, 4, 3)
    z[0, :, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0])  # only feature 0 fires
    labels = _labels([0, 1, 2, 3])

    result = participation_ratio(z, labels, n_groups=4)
    assert result.active.tolist() == [[True, False, False]]

    # The mean equals feature 0's own PR (= 2, mass split over two groups):
    # the two dead features contribute nothing at all.
    assert mean_over_active(result.pr, result.active).item() == pytest.approx(2.0, abs=1e-5)


def test_all_inactive_gives_zero_and_no_nan() -> None:
    z = torch.zeros(2, 6, 5)
    labels = torch.zeros(2, 6, dtype=torch.int64)

    result = participation_ratio(z, labels, n_groups=3)
    loss = mean_over_active(result.pr, result.active)
    assert loss.item() == 0.0
    assert torch.isfinite(result.pr).all()
