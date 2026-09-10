"""How sparse is the code, and how evenly is the dictionary used?

Two quantities the structural ablation needs to report separately from
reconstruction quality, because the whole hypothesis is that structure can be
bought without paying in FVU -- which is only checkable if L0 is held in view:

``L0Metric``            mean number of nonzero latents per patch. BatchTopK fixes
                        this at ``k`` during training but *not* at inference,
                        where the learned threshold decides; and Group-TopK only
                        ever removes activations, so the two can diverge.
``ActivationFrequency`` per-feature firing rate over the evaluation set. A
                        dictionary where a handful of features fire on almost
                        every patch and the rest almost never is not usefully
                        sparse however good its mean L0 looks.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric


class L0Metric(Metric):
    """Mean count of nonzero latents per patch."""

    full_state_update = False
    higher_is_better = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("nonzero", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("patches", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, latents: Tensor) -> None:
        """latents: [N, dict_size] -- any per-patch latent tensor."""
        self.nonzero += (latents != 0).sum().to(self.nonzero.dtype)
        self.patches += latents.shape[0]

    def compute(self) -> Tensor:
        if self.patches == 0:
            return torch.zeros((), device=self.nonzero.device)
        return self.nonzero / self.patches


class ActivationFrequencyMetric(Metric):
    """Per-feature firing rate: in what fraction of patches is feature j nonzero?

    `compute()` returns the full [dict_size] vector; the training script logs
    summary statistics off it (mean, median, and the share of features that fire
    on more than `hot_threshold` of all patches -- the "fires everywhere" tail
    that a low mean L0 can hide).
    """

    full_state_update = False

    def __init__(self, dict_size: int, hot_threshold: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dict_size = dict_size
        self.hot_threshold = hot_threshold
        self.add_state("fired", default=torch.zeros(dict_size), dist_reduce_fx="sum")
        self.add_state("patches", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, latents: Tensor) -> None:
        """latents: [N, dict_size]."""
        self.fired += (latents != 0).sum(dim=0).to(self.fired.dtype)
        self.patches += latents.shape[0]

    def compute(self) -> Tensor:
        if self.patches == 0:
            return torch.zeros(self.dict_size, device=self.fired.device)
        return self.fired / self.patches

    def summary(self, prefix: str = "") -> dict[str, float]:
        """The scalars worth logging, keyed `<prefix>activation_frequency_*`."""
        rates = self.compute()
        return {
            f"{prefix}activation_frequency_mean": float(rates.mean()),
            f"{prefix}activation_frequency_median": float(rates.median()),
            f"{prefix}activation_frequency_hot_share": float((rates > self.hot_threshold).float().mean()),
        }
