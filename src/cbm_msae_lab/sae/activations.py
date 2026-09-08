"""The SAE's configurable encoder nonlinearity: ReLU (default) or Sigmoid."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor

Activation = Literal["relu", "sigmoid"]

# Recorded as a buffer in every checkpoint (see `ConfigurableActivationSAE`),
# so a sigmoid dictionary can never be silently loaded as a ReLU one.
ACTIVATION_IDS: dict[Activation, int] = {"relu": 1, "sigmoid": 2}
ACTIVATION_BY_ID: dict[int, Activation] = {v: k for k, v in ACTIVATION_IDS.items()}


def activation_fn(name: Activation) -> Callable[[Tensor], Tensor]:
    """Sigmoid changes what BatchTopK/threshold selection means: sigma(x) > 0 everywhere,
    so *every* dictionary feature has a nonzero raw activation (unlike ReLU's hard
    zero), and sigma(0) = 0.5. BatchTopK still selects exactly k*B of them at train
    time, so training is unaffected; but the `use_threshold=True` inference path
    (see `MatryoshkaBatchTopKSAE.encode`) keeps everything above a learned scalar
    threshold, and with a floor of 0.5 rather than 0, that active set can be much
    larger under sigmoid than under ReLU.
    """
    if name == "relu":
        return torch.relu
    if name == "sigmoid":
        return torch.sigmoid
    raise ValueError(f"unknown activation {name!r}; expected one of {list(ACTIVATION_IDS)}")
