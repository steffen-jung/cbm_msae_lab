"""Unit tests for the sparsity and region-consistency metrics."""

from __future__ import annotations

import pytest
import torch

from cbm_msae_lab.metrics.region_consistency import RegionConsistencyMetric
from cbm_msae_lab.metrics.sparsity import ActivationFrequencyMetric, L0Metric


def test_l0_counts_nonzero_latents_per_patch() -> None:
    latents = torch.tensor(
        [
            [1.0, 0.0, 0.0, 2.0],  # 2 active
            [0.0, 0.0, 0.0, 0.0],  # 0 active
            [3.0, 4.0, 5.0, 6.0],  # 4 active
        ]
    )
    metric = L0Metric()
    metric.update(latents)
    assert metric.compute().item() == pytest.approx(2.0)


def test_l0_accumulates_across_batches() -> None:
    metric = L0Metric()
    metric.update(torch.ones(4, 3))  # 4 patches, 3 active each
    metric.update(torch.zeros(4, 3))  # 4 patches, 0 active each
    assert metric.compute().item() == pytest.approx(1.5)


def test_l0_of_an_empty_pass_is_zero_not_a_division_by_zero() -> None:
    assert L0Metric().compute().item() == 0.0


def test_activation_frequency_is_per_feature_firing_rate() -> None:
    latents = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    metric = ActivationFrequencyMetric(dict_size=3)
    metric.update(latents)
    assert metric.compute().tolist() == pytest.approx([0.75, 0.25, 0.0])


def test_activation_frequency_summary_reports_the_always_on_tail() -> None:
    """The statistic a low mean L0 can hide: a few features firing nearly everywhere."""
    latents = torch.zeros(10, 4)
    latents[:, 0] = 1.0  # fires on every patch
    latents[:2, 1] = 1.0  # fires on 20%
    metric = ActivationFrequencyMetric(dict_size=4, hot_threshold=0.5)
    metric.update(latents)

    summary = metric.summary(prefix="val/")
    assert summary["val/activation_frequency_mean"] == pytest.approx(0.3)
    assert summary["val/activation_frequency_hot_share"] == pytest.approx(0.25)  # only feature 0


def test_region_consistency_reports_pr_and_dominant_share() -> None:
    """One feature confined to a single region, one split evenly over two."""
    latents = torch.zeros(1, 4, 2)
    latents[0, :, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0])  # both patches of region 0
    latents[0, :, 1] = torch.tensor([1.0, 0.0, 1.0, 0.0])  # one patch each in regions 0 and 1
    labels = torch.tensor([[0, 0, 1, 1]])

    metric = RegionConsistencyMetric(n_groups=2)
    metric.update(latents, labels)
    result = metric.compute()

    assert result["region_pr"].item() == pytest.approx((1.0 + 2.0) / 2, abs=1e-5)
    assert result["region_dominant_share"].item() == pytest.approx((1.0 + 0.5) / 2, abs=1e-5)


def test_region_consistency_skips_images_where_nothing_fires() -> None:
    """An all-dead image carries no signal and must not pull the mean toward zero."""
    live = torch.zeros(1, 4, 1)
    live[0, :, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    labels = torch.tensor([[0, 0, 1, 1]])

    metric = RegionConsistencyMetric(n_groups=2)
    metric.update(live, labels)
    metric.update(torch.zeros(1, 4, 1), labels)
    assert metric.compute()["region_pr"].item() == pytest.approx(1.0, abs=1e-5)
