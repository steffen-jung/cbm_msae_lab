"""Shared interface every pluggable loss term implements.

``ComposableLossTrainer.loss()`` runs the SAE encode/decode exactly once per
step, packages the results into one ``LossContext``, and then calls
``.compute(ctx)`` on every ``Loss`` whose weight is nonzero, summing the
weighted results. This is how the user's own scale/spatial loss and the S2AE
losses can be mixed freely with CFM's original reconstruction+auxk loss.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from cbm_msae_lab.sae.model import ConfigurableActivationSAE
    from cbm_msae_lab.sae.trainer import ComposableLossTrainer


@dataclass
class LossContext:
    """Everything a `Loss.compute()` might need, computed once per training step.

    Flat tensors (`x`, `f`, `x_hat`, `post_act`) have shape [N, *] where
    N = B * P (all patches of all images in the batch, stacked); the matching
    `*_img` tensors reshape the same values back to [B, P, *] so spatial
    losses can tell which patches belong to the same image.
    """

    sae: ConfigurableActivationSAE
    trainer: ComposableLossTrainer
    x: Tensor  # [N, activation_dim] -- the SAE's reconstruction target
    f: Tensor  # [N, dict_size] -- encoded latents (post BatchTopK, pre-Matryoshka-truncation already applied)
    x_hat: Tensor  # [N, activation_dim] -- full decode(f), i.e. the SAE's reconstruction
    post_act: Tensor  # [N, dict_size] -- raw post-nonlinearity activations, before BatchTopK selection
    x_img: Tensor  # [B, P, activation_dim]
    f_img: Tensor  # [B, P, dict_size]
    x_hat_img: Tensor  # [B, P, activation_dim]
    grid: tuple[int, int]  # (H, W) patch-grid shape for this batch, with P == H * W
    step: int | None
    group_labels: Tensor | None = None  # [B, P] int64, per-image S2AE-style cluster ids; see attention_grouping.py.
    # None unless the training script looked them up from a group-label cache for this batch
    # (only needed when loss.group_sparsity.grouping or loss.exclusivity.grouping == "attention").


class Loss(ABC):
    def __init__(self, weight: float) -> None:
        self.weight = weight

    @abstractmethod
    def compute(self, ctx: LossContext) -> Tensor:
        """Returns a scalar tensor (not yet multiplied by `self.weight`)."""
        raise NotImplementedError
