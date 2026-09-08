"""Streaming reconstruction-quality metrics: MSE and per-sample cosine similarity."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric


class ReconstructionMetric(Metric):
    """`compute()` returns a dict with keys "mse" and "cosine_similarity"."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_sq_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_cosine", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_elements", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, x: Tensor, x_hat: Tensor) -> None:
        """x, x_hat: [N, activation_dim]."""
        self.sum_sq_error += (x - x_hat).pow(2).sum()
        self.sum_cosine += F.cosine_similarity(x, x_hat, dim=-1).sum()
        self.num_elements += x.numel()
        self.num_samples += x.shape[0]

    def compute(self) -> dict[str, Tensor]:
        return {
            "mse": self.sum_sq_error / self.num_elements.clamp_min(1),
            "cosine_similarity": self.sum_cosine / self.num_samples.clamp_min(1),
        }
