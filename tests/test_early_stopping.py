from __future__ import annotations

from cbm_msae_lab.early_stopping import EarlyStopping


def test_stops_after_patience_checks_without_improvement() -> None:
    es = EarlyStopping(patience=3, min_delta=0.0)
    assert es.step(1.0) is False  # first check always "improves" on inf
    assert es.step(1.0) is False  # bad check 1 (no improvement)
    assert es.step(1.0) is False  # bad check 2
    assert es.step(1.0) is True  # bad check 3 == patience -> stop


def test_improvement_resets_patience() -> None:
    es = EarlyStopping(patience=2, min_delta=0.0)
    assert es.step(1.0) is False
    assert es.step(1.0) is False  # bad check 1
    assert es.step(0.5) is False  # improves -> resets
    assert es.step(0.5) is False  # bad check 1 again
    assert es.step(0.5) is True  # bad check 2 == patience -> stop
    assert es.best == 0.5


def test_min_delta_requires_relative_improvement() -> None:
    es = EarlyStopping(patience=1, min_delta=0.1)  # need >= 10% relative improvement
    assert es.step(1.0) is False
    assert es.step(0.95) is True  # only 5% better -> counts as a bad check -> stop (patience=1)


def test_min_delta_accepts_sufficient_improvement() -> None:
    es = EarlyStopping(patience=1, min_delta=0.1)
    assert es.step(1.0) is False
    assert es.step(0.85) is False  # 15% better -> resets, no stop
    assert es.best == 0.85
