"""Fraction of Variance Unexplained: how much of the input's variance the SAE fails to reconstruct.

    FVU = Var[x - x_hat] / Var[x]

FVU = 0 means perfect reconstruction; FVU = 1 means the reconstruction is no
better than always predicting the mean. This differs from CFM's own
`log_stats` (which computes `1 - Var[x-x_hat]/Var[x]` per-batch, using each
batch's own mean as the variance reference) only in bookkeeping: this class
accumulates sums across every `update()` call and computes one exact,
whole-dataset variance in `compute()`, instead of averaging many per-batch
estimates -- a more accurate estimator when used across a full validation
pass rather than a single batch.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric


class FVUMetric(Metric):
    def __init__(self, activation_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.activation_dim = activation_dim
        self.add_state("sum_x", default=torch.zeros(activation_dim), dist_reduce_fx="sum")
        self.add_state("sum_x_sq", default=torch.zeros(activation_dim), dist_reduce_fx="sum")
        self.add_state("sum_r", default=torch.zeros(activation_dim), dist_reduce_fx="sum")
        self.add_state("sum_r_sq", default=torch.zeros(activation_dim), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, x: Tensor, x_hat: Tensor) -> None:
        """x, x_hat: [N, activation_dim]."""
        residual = x - x_hat
        self.sum_x += x.sum(dim=0)
        self.sum_x_sq += (x * x).sum(dim=0)
        self.sum_r += residual.sum(dim=0)
        self.sum_r_sq += (residual * residual).sum(dim=0)
        self.count += x.shape[0]

    def compute(self) -> Tensor:
        mean_x = self.sum_x / self.count
        total_var = (self.sum_x_sq / self.count - mean_x * mean_x).sum()

        mean_r = self.sum_r / self.count
        residual_var = (self.sum_r_sq / self.count - mean_r * mean_r).sum()

        return residual_var / total_var.clamp_min(1e-12)
