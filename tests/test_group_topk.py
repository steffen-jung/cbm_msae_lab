"""Unit tests for Group-TopK, the hard regional selection rule."""

from __future__ import annotations

import pytest
import torch

from cbm_msae_lab.attention_grouping import onehot_mean_aggregate, scatter_groups_to_patches
from cbm_msae_lab.losses.tiling import tile_group_labels
from cbm_msae_lab.sae.group_topk import GroupTopK, group_topk_mask, k_per_level
from cbm_msae_lab.sae.model import ConfigurableActivationSAE


def test_k_per_level_splits_the_budget_proportionally() -> None:
    """A total budget follows the Matryoshka group sizes, so no second list has to
    be kept in sync with `sae.group_fractions`."""
    blocks = [(0, 100), (100, 300), (300, 1000)]
    assert k_per_level(100, group_sizes=[100, 200, 700], blocks=blocks) == [10, 20, 70]


def test_k_per_level_never_starves_a_level_or_exceeds_it() -> None:
    """A tiny level still gets one feature (0 would switch it off for every region
    at once); a k larger than the level is clamped to the level's width."""
    blocks = [(0, 2), (2, 1000)]
    ks = k_per_level(10, group_sizes=[2, 998], blocks=blocks)
    assert ks[0] == 1
    assert k_per_level(10_000, group_sizes=[2, 998], blocks=blocks) == [2, 998]


def test_at_most_k_features_survive_per_region_and_level() -> None:
    """The core guarantee: within one region and one Matryoshka block, no more than
    k_l distinct features are allowed."""
    generator = torch.Generator().manual_seed(0)
    post_act = torch.rand(2, 16, 12, generator=generator)
    labels = torch.randint(0, 4, (2, 16), generator=generator)
    blocks, ks = [(0, 6), (6, 12)], [2, 3]

    mask = group_topk_mask(post_act, labels, n_groups=4, blocks=blocks, ks=ks)

    for b in range(2):
        for g in range(4):
            patches = labels[b] == g
            if not patches.any():
                continue
            for (start, end), k in zip(blocks, ks, strict=True):
                allowed = mask[b][patches][:, start:end]
                # Every patch of a region sees exactly the same decision...
                assert (allowed == allowed[0]).all()
                # ...and that decision names at most k features.
                assert int(allowed[0].sum()) == k


def test_mask_selects_the_features_with_the_largest_regional_mean() -> None:
    """One region, one block, hand-built activations: the mask must name exactly
    the top-k by mean over the region's patches -- not by max, not by a single patch."""
    post_act = torch.zeros(1, 4, 5)
    #                     feature: 0    1    2    3    4
    post_act[0, 0] = torch.tensor([9.0, 1.0, 0.0, 0.0, 0.0])  # feature 0 spikes on one patch only
    post_act[0, 1] = torch.tensor([0.0, 1.0, 2.0, 0.0, 0.0])
    post_act[0, 2] = torch.tensor([0.0, 1.0, 2.0, 0.0, 0.0])
    post_act[0, 3] = torch.tensor([0.0, 1.0, 2.0, 0.0, 0.0])
    # means: f0 = 2.25, f1 = 1.0, f2 = 1.5, f3 = f4 = 0
    labels = torch.zeros(1, 4, dtype=torch.int64)

    mask = group_topk_mask(post_act, labels, n_groups=1, blocks=[(0, 5)], ks=[2])
    assert mask[0, 0].tolist() == [True, False, True, False, False]


def test_regions_decide_independently() -> None:
    """Two regions with opposite preferences must end up with opposite masks --
    the whole point of a *regional* rule."""
    post_act = torch.zeros(1, 4, 4)
    post_act[0, 0] = post_act[0, 1] = torch.tensor([5.0, 4.0, 0.0, 0.0])  # region 0 prefers f0, f1
    post_act[0, 2] = post_act[0, 3] = torch.tensor([0.0, 0.0, 4.0, 5.0])  # region 1 prefers f2, f3
    labels = torch.tensor([[0, 0, 1, 1]])

    mask = group_topk_mask(post_act, labels, n_groups=2, blocks=[(0, 4)], ks=[2])
    assert mask[0, 0].tolist() == [True, True, False, False]
    assert mask[0, 2].tolist() == [False, False, True, True]


def test_applying_the_mask_can_only_remove_activations() -> None:
    """Group-TopK intersects with BatchTopK; it must never revive a latent that
    BatchTopK already discarded."""
    generator = torch.Generator().manual_seed(1)
    post_act = torch.rand(2, 16, 10, generator=generator)
    batch_topk = post_act * (post_act > 0.7)  # stand-in for the BatchTopK selection
    labels = torch.randint(0, 4, (2, 16), generator=generator)

    mask = group_topk_mask(post_act, labels, n_groups=4, blocks=[(0, 10)], ks=[3])
    masked = batch_topk * mask

    assert ((masked != 0) <= (batch_topk != 0)).all()
    assert (masked[masked != 0] == batch_topk[masked != 0]).all()  # surviving values are untouched


def test_tile_grouping_resolves_to_the_grid_partition() -> None:
    """`GroupTopK(grouping="tile")` must derive exactly the tile labels from the
    grid, so the tile path and an explicit label tensor produce the same mask."""
    generator = torch.Generator().manual_seed(2)
    post_act = torch.rand(2, 16, 8, generator=generator)

    sae = ConfigurableActivationSAE(activation_dim=4, dict_size=8, k=2, group_sizes=[8])
    op = GroupTopK(grouping="tile", tile_size=2, k_group=3)
    from_wrapper = op.mask(post_act, sae, grid=(4, 4), group_labels=None)

    explicit = tile_group_labels(4, 4, 2).unsqueeze(0).expand(2, -1)
    from_explicit = group_topk_mask(post_act, explicit, n_groups=4, blocks=[(0, 8)], ks=[3])
    assert torch.equal(from_wrapper, from_explicit)


def test_clustered_grouping_without_labels_names_the_cache_script() -> None:
    """The failure mode is a missing cache, so the error has to say which script
    builds it -- the same contract the S2AE losses already follow."""
    sae = ConfigurableActivationSAE(activation_dim=4, dict_size=8, k=2, group_sizes=[8])
    op = GroupTopK(grouping="attention")

    with pytest.raises(RuntimeError, match="extract_attention_groups.py"):
        op.mask(torch.rand(1, 4, 8), sae, grid=(2, 2), group_labels=None)


def test_k_per_level_override_must_match_the_level_count() -> None:
    sae = ConfigurableActivationSAE(activation_dim=4, dict_size=8, k=2, group_sizes=[4, 4])
    op = GroupTopK(grouping="tile", tile_size=2, k_per_level=[1, 1, 1])

    with pytest.raises(ValueError, match="3 entries but the SAE has 2"):
        op.mask(torch.rand(1, 4, 8), sae, grid=(2, 2), group_labels=None)


def test_segment_reductions_match_a_naive_one_hot_reference() -> None:
    """`onehot_mean_aggregate`/`scatter_groups_to_patches` are hand-optimized into
    scatter_add/gather to keep the group count out of the cost. Pin them against
    the obvious one-hot formulation they replaced."""
    generator = torch.Generator().manual_seed(3)
    f_img = torch.rand(3, 10, 6, generator=generator)
    labels = torch.randint(0, 4, (3, 10), generator=generator)
    onehot = torch.nn.functional.one_hot(labels, num_classes=4).to(f_img.dtype)

    sums = torch.einsum("bpg,bpj->bgj", onehot, f_img)
    expected_mean = sums / onehot.sum(dim=1).unsqueeze(-1).clamp_min(1.0)
    assert torch.allclose(onehot_mean_aggregate(f_img, labels, 4), expected_mean, atol=1e-6)

    group_values = torch.rand(3, 4, 6, generator=generator)
    expected_scatter = torch.einsum("bpg,bgj->bpj", onehot, group_values)
    assert torch.allclose(scatter_groups_to_patches(group_values, labels, 4), expected_scatter, atol=1e-6)
