"""Unit tests for `TimedLoader`, the DataLoader-bottleneck instrumentation."""

from __future__ import annotations

import time

from cbm_msae_lab.timing import TimedLoader


class SleepyLoader:
    """A minimal reusable batch iterable: sleeps `item_delay_s` before yielding
    each of `n_items` batches, simulating a slow data source (e.g. a live
    encoder forward pass or disk-bound cache reads)."""

    def __init__(self, n_items: int, item_delay_s: float) -> None:
        self.n_items = n_items
        self.item_delay_s = item_delay_s

    def __iter__(self):
        for i in range(self.n_items):
            time.sleep(self.item_delay_s)
            yield i


def test_data_wait_s_reflects_loader_delay() -> None:
    loader = TimedLoader(SleepyLoader(n_items=3, item_delay_s=0.05))

    batches = list(iter(loader))

    assert batches == [0, 1, 2]
    # The last batch's fetch time is what's left in `data_wait_s` after the loop.
    assert loader.data_wait_s >= 0.04  # some slack below the 0.05s sleep for timer jitter


def test_compute_s_reflects_time_spent_between_batches() -> None:
    loader = TimedLoader(SleepyLoader(n_items=3, item_delay_s=0.0))

    for _ in loader:
        time.sleep(0.05)  # simulate a slow training step

    # After the loop, `compute_s` reflects the caller's time before the *last*
    # `next()` call, i.e. the simulated training step just before it.
    assert loader.compute_s >= 0.04


def test_bottleneck_fraction_is_zero_before_any_batch() -> None:
    loader = TimedLoader(SleepyLoader(n_items=1, item_delay_s=0.01))
    assert loader.bottleneck_fraction == 0.0


def test_bottleneck_fraction_is_high_when_loader_dominates() -> None:
    loader = TimedLoader(SleepyLoader(n_items=5, item_delay_s=0.05))

    for _ in loader:
        pass  # no simulated compute time at all -- loader is the only cost

    assert loader.bottleneck_fraction > 0.9


def test_reusable_across_multiple_iterations() -> None:
    """Like `ActivationLoader`, `TimedLoader` must support being iterated more
    than once (once per training epoch), not just a single exhausted pass."""
    loader = TimedLoader(SleepyLoader(n_items=2, item_delay_s=0.0))

    first_pass = list(iter(loader))
    second_pass = list(iter(loader))

    assert first_pass == [0, 1]
    assert second_pass == [0, 1]


def test_len_delegates_to_wrapped_loader() -> None:
    class WithLen:
        def __iter__(self):
            return iter([])

        def __len__(self) -> int:
            return 42

    assert len(TimedLoader(WithLen())) == 42
