"""Instrumentation to answer "is the DataLoader the bottleneck?" without guessing.

Wraps any batch iterable (here always an `ActivationLoader`) and times two
disjoint things per batch:

- ``data_wait_s``: wall-clock time spent inside `next()` producing this batch
  (image loading + any live encoder/projection forward pass, or a cache read).
  If `num_workers > 0` and the loader keeps up, this is close to 0 (the next
  batch was already prefetched while the previous training step ran).
- ``compute_s``: wall-clock time the *caller* spent between receiving the
  previous batch and asking for this one -- i.e. the training step itself
  (SAE forward/backward, optimizer step).

If ``data_wait_s`` is consistently large relative to ``compute_s``, the
DataLoader (not the GPU) is the bottleneck.

Note the one-step lag: at the moment batch ``i`` is handed back, ``data_wait_s``
already reflects batch ``i`` (just measured), but ``compute_s`` still reflects
the *previous* step (processing batch ``i-1``) -- the current step's compute
time isn't known until the caller asks for the next batch. Fine for a rolling
progress-bar/wandb signal, just not exactly index-aligned pointwise.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


class TimedLoader:
    """Wraps a reusable batch iterable, timing every `next()` call separately
    from the caller's own time between calls.

    Reusable across epochs like `ActivationLoader`: `__iter__` opens a fresh
    inner iterator every time it's called, rather than being a one-shot
    generator that's exhausted after a single pass.
    """

    def __init__(self, loader: Iterable[T]) -> None:
        self._loader = loader
        self.data_wait_s: float = 0.0
        self.compute_s: float = 0.0

    @property
    def bottleneck_fraction(self) -> float:
        """Fraction of (wait + compute) spent waiting for data, in [0, 1].

        0.0 before the first batch (both timers start at 0.0).
        """
        total = self.data_wait_s + self.compute_s
        return self.data_wait_s / total if total > 0 else 0.0

    def __iter__(self) -> Iterator[T]:
        it = iter(self._loader)
        t_handoff = time.perf_counter()  # moment we last handed a batch to the caller
        while True:
            t_before_next = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                return
            t_after_next = time.perf_counter()

            self.data_wait_s = t_after_next - t_before_next
            self.compute_s = t_before_next - t_handoff

            # Taken *before* `yield`, not after: code after `yield` only runs once
            # the caller calls `next()` again, which is already past the compute
            # window we're trying to measure -- by then `t_before_next` above and
            # this timestamp would be back-to-back, always reading ~0.
            t_handoff = time.perf_counter()
            yield batch

    def __len__(self) -> int:
        return len(self._loader)  # type: ignore[arg-type]
