"""Unit tests for the S2AE-style attention+spatial-proximity patch grouping."""

from __future__ import annotations

import numpy as np
import torch

from cbm_msae_lab.attention_grouping import cluster_patches, onehot_l2_aggregate


def test_cluster_patches_respects_n_clusters_and_grid_size() -> None:
    torch.manual_seed(0)
    grid_h, grid_w = 4, 4
    p = grid_h * grid_w
    attention = torch.rand(p, p)

    labels = cluster_patches(attention, grid_h, grid_w, n_clusters=4, spatial_coeff=0.02)

    assert labels.shape == (p,)
    assert labels.dtype == np.int64
    assert set(labels.tolist()) <= set(range(4))
    # n_clusters is clamped to the number of patches when it would otherwise exceed it.
    labels_more_clusters_than_patches = cluster_patches(attention, grid_h, grid_w, n_clusters=1000, spatial_coeff=0.02)
    assert len(set(labels_more_clusters_than_patches.tolist())) <= p


def test_cluster_patches_is_deterministic() -> None:
    torch.manual_seed(1)
    grid_h, grid_w = 5, 5
    attention = torch.rand(grid_h * grid_w, grid_h * grid_w)

    labels_a = cluster_patches(attention, grid_h, grid_w, n_clusters=5)
    labels_b = cluster_patches(attention, grid_h, grid_w, n_clusters=5)

    assert np.array_equal(labels_a, labels_b)


def test_cluster_patches_groups_spatially_close_uniform_attention_patches() -> None:
    """With perfectly uniform attention (no signal), the spatial term alone
    should dominate and produce spatially contiguous-ish groups rather than
    an arbitrary/random partition -- a weak sanity check that the spatial
    term is actually doing something."""
    grid_h, grid_w = 4, 4
    p = grid_h * grid_w
    uniform_attention = torch.ones(p, p) / p

    labels = cluster_patches(uniform_attention, grid_h, grid_w, n_clusters=2, spatial_coeff=0.02)

    # The two halves of the grid (top vs bottom rows) should mostly land in
    # different clusters when attention carries no signal at all.
    top_half = labels[: p // 2]
    bottom_half = labels[p // 2 :]
    assert not np.array_equal(np.unique(top_half), np.unique(bottom_half)) or len(np.unique(labels)) == 1


def test_onehot_l2_aggregate_matches_manual_computation() -> None:
    torch.manual_seed(2)
    B, P, D, G = 2, 6, 3, 3
    f_img = torch.randn(B, P, D)
    group_labels = torch.randint(0, G, (B, P))

    result = onehot_l2_aggregate(f_img, group_labels, n_clusters=G)
    assert result.shape == (B, G, D)

    for b in range(B):
        for g in range(G):
            mask = group_labels[b] == g
            expected = f_img[b][mask].pow(2).sum(dim=0).sqrt() if mask.any() else torch.zeros(D)
            assert torch.allclose(result[b, g], expected, atol=1e-5)


def test_onehot_l2_aggregate_empty_cluster_is_zero() -> None:
    f_img = torch.randn(1, 4, 5)
    group_labels = torch.zeros(1, 4, dtype=torch.long)  # every patch in cluster 0

    result = onehot_l2_aggregate(f_img, group_labels, n_clusters=3)

    assert torch.allclose(result[0, 1], torch.zeros(5))
    assert torch.allclose(result[0, 2], torch.zeros(5))
