"""Stops training once a tracked (lower-is-better) metric stops improving by
more than a relative margin, at whatever cadence the caller calls `step()` --
`scripts/train.py` calls it once per eval check (`train.eval.every_n_epochs`
epochs), not every epoch, since that's the only cadence the eval metrics are
actually computed on. Which metric is tracked is the caller's choice
(`train.early_stopping.monitor`, default `val/fvu`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 1e-3
    best: float = float("inf")
    num_bad_checks: int = 0

    def step(self, value: float) -> bool:
        """Records one check of the tracked metric. Returns True once
        `patience` consecutive checks failed to beat the best value seen so
        far by at least `min_delta` (a *relative* fraction, e.g. 0.001 = 0.1%)."""
        if value < self.best * (1.0 - self.min_delta):
            self.best = value
            self.num_bad_checks = 0
        else:
            self.num_bad_checks += 1
        return self.num_bad_checks >= self.patience
