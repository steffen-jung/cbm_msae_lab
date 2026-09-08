"""The trainer: CFM's proven optimizer machinery + a composable, configurable loss.

Subclasses ``dictionary_learning``'s ``MatryoshkaBatchTopKTrainer`` and
overrides only ``loss()`` and ``geometric_median()`` -- exactly the pattern
``cfm_finegrained/src/cfm_fg/scale_loss.py::ScaleMatryoshkaTrainer`` already
validated. Everything else (the Adam step, decoder-unit-norm renormalization,
gradient-parallel-to-decoder removal, the learned inference threshold's EMA
update, dead-feature bookkeeping's schedule) is inherited unchanged from
``dictionary_learning``, so this codebase never re-derives that logic.
"""

from __future__ import annotations

from collections import namedtuple
from functools import partial
from typing import Any

import torch
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKTrainer,
)
from torch import Tensor

from cbm_msae_lab.config_schema import LossConfig
from cbm_msae_lab.losses.base import Loss, LossContext
from cbm_msae_lab.losses.registry import build_losses
from cbm_msae_lab.sae.activations import Activation
from cbm_msae_lab.sae.model import ConfigurableActivationSAE


class ComposableLossTrainer(MatryoshkaBatchTopKTrainer):
    """Batches arrive as ``[B, P, D]`` (images, not loose tokens), since the
    spatial losses need to know which patches are neighbours / belong to the
    same image. The SAE itself still sees the flattened ``[B*P, D]`` view, so
    BatchTopK selects across the whole batch exactly as it does in CFM.
    """

    def __init__(
        self,
        *,
        grid_shape: tuple[int, int],
        loss_config: LossConfig,
        activation: Activation = "relu",
        **kwargs: Any,
    ) -> None:
        dict_class = partial(ConfigurableActivationSAE, activation=activation)
        super().__init__(dict_class=dict_class, **kwargs)
        self.grid_shape = grid_shape
        self.activation: Activation = activation
        self.losses: dict[str, Loss] = build_losses(loss_config)
        # Set by the training script right before `update()`/`loss()` is called,
        # when a loss needs S2AE-style attention-based patch groups for this
        # batch (see attention_grouping.py) -- an instance attribute rather
        # than a `loss()` parameter because `update()` itself is inherited
        # unchanged from `dictionary_learning` and calls `self.loss(x, step=step)`
        # with a fixed signature we don't want to have to override just for this.
        self.group_labels: Tensor | None = None

    @staticmethod
    def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-5) -> Tensor:
        """points: [B, P, D] -> [D]. Flattens to [B*P, D] before delegating to the base algorithm."""
        flat = points.reshape(-1, points.shape[-1])
        return MatryoshkaBatchTopKTrainer.geometric_median(flat, max_iter, tol)

    def loss(self, x_img: Tensor, step: int | None = None, logging: bool = False) -> Any:
        if x_img.dim() != 3:
            raise ValueError(f"expected [B, P, D] image batches, got {tuple(x_img.shape)}")
        B, P, D = x_img.shape
        x = x_img.reshape(B * P, D)  # [N, D], N = B * P

        f, active_indices_F, post_act = self.ae.encode(
            x, return_active=True, use_threshold=False
        )  # f, post_act: [N, dict_size]

        if step is not None and step > self.threshold_start_step:
            self.update_threshold(f)

        x_hat = self.ae.decode(f)  # [N, D] -- full reconstruction (all Matryoshka groups already summed via `f`)
        self.effective_l0 = self.k

        # Dead-feature bookkeeping (ported from the base trainer's loss()): a
        # feature's "not fired in N tokens" counter resets whenever it fires.
        num_tokens_in_step = x.size(0)
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        ctx = LossContext(
            sae=self.ae,
            trainer=self,
            x=x,
            f=f,
            x_hat=x_hat,
            post_act=post_act,
            x_img=x_img,
            f_img=f.reshape(B, P, -1),
            x_hat_img=x_hat.reshape(B, P, D),
            grid=self.grid_shape,
            step=step,
            group_labels=self.group_labels,
        )

        total = torch.zeros((), device=x.device, dtype=x.dtype)
        per_loss_values: dict[str, float] = {}
        for name, loss_obj in self.losses.items():
            if loss_obj.weight == 0.0:
                continue
            value = loss_obj.compute(ctx)
            total = total + loss_obj.weight * value
            per_loss_values[name] = float(value.item())

        if not logging:
            return total
        return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
            x, x_hat, f, {**per_loss_values, "loss": float(total.item())}
        )

    @property
    def config(self) -> dict[str, Any]:
        cfg = dict(super().config)
        cfg.update(
            {
                "trainer_class": "ComposableLossTrainer",
                "dict_class": "ConfigurableActivationSAE",
                "activation": self.activation,
                "grid_shape": list(self.grid_shape),
                "active_losses": {name: loss_obj.weight for name, loss_obj in self.losses.items()},
            }
        )
        return cfg
