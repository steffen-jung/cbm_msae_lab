"""Group-TopK: a hard regional selection rule, not a loss.

For every region g and Matryoshka block B_l, rank the block's features by their
*mean pre-activation over that region's patches*

    a_bar^g_j = (1 / M_g) * sum_{i in g} a_ij ,

keep the k_l strongest, and forbid the rest inside that region:

    z_ij = a_ij * 1[j in S_g^l]   for every patch i in g.

Because it is a mask and not a penalty it contributes no gradient of its own and
therefore no shrinkage pressure -- the same reason it sits here rather than in
`losses/`.

Relationship to BatchTopK: the two are complementary and both apply. BatchTopK
answers "which activations survive at all" across the whole batch; Group-TopK
answers "which features may be used inside this region". The final mask is their
*intersection*, so Group-TopK can only ever remove an activation, never revive
one BatchTopK discarded.

Applied during training only (in `ComposableLossTrainer.loss`), by explicit
choice: inference keeps the learned BatchTopK threshold and needs no group
labels. That is a train/inference asymmetry and belongs in any report of results
obtained with it.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cbm_msae_lab.attention_grouping import onehot_mean_aggregate, scatter_groups_to_patches
from cbm_msae_lab.grouping import resolve_group_labels, validate_grouping


def k_per_level(k_group: int, group_sizes: list[int], blocks: list[tuple[int, int]]) -> list[int]:
    """Splits one total budget across the Matryoshka levels in proportion to their size.

    `k_l = max(1, round(k_group * |B_l| / dict_size))`, so the split follows
    `sae.group_fractions` automatically and no second list has to be kept in sync
    with it. Every level keeps at least one feature per region -- a level allowed
    zero features would be silently switched off for every region at once.
    """
    dict_size = sum(group_sizes)
    ks = [max(1, round(k_group * (end - start) / dict_size)) for start, end in blocks]
    return [min(k, end - start) for k, (start, end) in zip(ks, blocks, strict=True)]


def group_topk_mask(
    post_act_img: Tensor,
    group_labels: Tensor,
    n_groups: int,
    blocks: list[tuple[int, int]],
    ks: list[int],
) -> Tensor:
    """post_act_img: [B, P, dict_size] pre-selection activations, group_labels: [B, P]
    -> boolean mask [B, P, dict_size], True where the feature is allowed in that patch's region.

    Columns outside `blocks` (Matryoshka levels beyond `active_groups`) are False:
    `encode` has already zeroed them, so forbidding them changes nothing and keeps
    the mask's meaning literal.
    """
    group_means = onehot_mean_aggregate(post_act_img, group_labels, n_groups)  # [B, G, dict_size]
    mask = torch.zeros_like(post_act_img, dtype=torch.bool)

    for (start, end), k in zip(blocks, ks, strict=True):
        block_means = group_means[:, :, start:end]  # [B, G, |B_l|]
        top_indices = block_means.topk(k, dim=-1, sorted=False).indices  # [B, G, k]
        allowed = torch.zeros_like(block_means, dtype=block_means.dtype)
        allowed.scatter_(-1, top_indices, 1.0)  # [B, G, |B_l|], 1.0 for the k strongest
        # Route each group's row back to its own patches; the one-hot makes this a
        # gather, so every patch receives exactly its own region's decision.
        mask[:, :, start:end] = scatter_groups_to_patches(allowed, group_labels, n_groups) > 0

    return mask


class GroupTopK:
    """Config-shaped wrapper around `group_topk_mask`, held by `ComposableLossTrainer`.

    Kept out of `ConfigurableActivationSAE` on purpose: the SAE sees flat
    `[N, activation_dim]` tokens and knows nothing about which image or region a
    token came from, and the selection is training-only anyway.
    """

    def __init__(
        self,
        grouping: str = "tile",
        tile_size: int = 2,
        n_clusters: int = 20,
        k_group: int = 48,
        k_per_level: list[int] | None = None,
    ) -> None:
        self.grouping = validate_grouping(grouping, "group_topk")
        self.tile_size = tile_size
        self.n_clusters = n_clusters
        self.k_group = k_group
        self.k_per_level = k_per_level

    def _ks(self, sae, blocks: list[tuple[int, int]]) -> list[int]:
        if self.k_per_level is None:
            return k_per_level(self.k_group, sae.group_sizes.tolist(), blocks)
        if len(self.k_per_level) != len(blocks):
            raise ValueError(
                f"group_topk.k_per_level has {len(self.k_per_level)} entries but the SAE has "
                f"{len(blocks)} active Matryoshka levels"
            )
        return list(self.k_per_level)

    def mask(
        self,
        post_act_img: Tensor,
        sae,
        grid: tuple[int, int],
        group_labels: Tensor | None,
    ) -> Tensor:
        """post_act_img: [B, P, dict_size] -> boolean mask of the same shape."""
        labels, n_groups = resolve_group_labels(
            self.grouping,
            grid=grid,
            tile_size=self.tile_size,
            batch_size=post_act_img.shape[0],
            device=post_act_img.device,
            group_labels=group_labels,
            n_clusters=self.n_clusters,
            consumer="group_topk",
        )
        blocks = sae.matryoshka_blocks
        return group_topk_mask(post_act_img, labels, n_groups, blocks, self._ks(sae, blocks))
