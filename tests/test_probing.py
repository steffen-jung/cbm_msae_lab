"""Unit tests for `probing.py`'s linear-probe recipe.

Only checks shapes and that the pipeline runs end-to-end on a tiny synthetic
problem (a handful of well-separated Gaussian blobs) -- not convergence to any
particular accuracy, which is `scripts/evaluate_task_accuracy.py`'s job on
real checkpoints, not a unit test's.
"""

from __future__ import annotations

import numpy as np
import torch

from cbm_msae_lab.probing import ProbeConfig, run_probe


def make_synthetic_problem(
    n_per_class: int = 20, n_classes: int = 3, dim: int = 8, seed: int = 0
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Well-separated class clusters (one one-hot-ish direction per class),
    split 60/20/20 (train/val/test), so even a handful of training epochs is
    enough to separate them -- keeps the unit test fast without needing real
    representations."""
    rng = np.random.default_rng(seed)
    centers = np.eye(n_classes, dim, dtype=np.float32) * 5.0  # [n_classes, dim]

    x_parts, y_parts, split_parts = [], [], []
    for c in range(n_classes):
        noise = rng.normal(scale=0.1, size=(n_per_class, dim)).astype(np.float32)
        x_parts.append(centers[c] + noise)
        y_parts.append(np.full(n_per_class, c, dtype=np.int64))
        # First 60% train, next 20% val, last 20% test, per class (stratified).
        n_train = int(0.6 * n_per_class)
        n_val = int(0.2 * n_per_class)
        splits = np.array(
            ["train"] * n_train + ["val"] * n_val + ["test"] * (n_per_class - n_train - n_val)
        )
        split_parts.append(splits)

    x = torch.from_numpy(np.concatenate(x_parts, axis=0))  # [N, dim]
    labels = np.concatenate(y_parts)  # [N]
    splits = np.concatenate(split_parts)  # [N]
    train = splits == "train"
    val = splits == "val"
    test = splits == "test"
    return x, labels, train, val, test


def test_run_probe_shapes_and_ranges() -> None:
    x, labels, train, val, test = make_synthetic_problem()
    cfg = ProbeConfig(epochs=50, batch_size=16, lr_grid=(1e-2,), lambda_grid=(0.0,))

    result = run_probe(
        name="synthetic",
        x=x,
        labels=labels,
        train=train,
        val=val,
        test=test,
        n_classes=3,
        cfg=cfg,
        device=torch.device("cpu"),
        seeds=[0, 1],
    )

    assert result.dim == 8
    assert result.chance == 1.0 / 3
    assert 0.0 <= result.test_top1_mean <= 1.0
    assert 0.0 <= result.test_top5_mean <= 1.0
    assert result.test_top5_mean >= result.test_top1_mean  # top-5 can only be at least as good
    assert len(result.per_seed) == 2
    assert result.selected_lr == 1e-2
    assert result.selected_lambda == 0.0


def test_run_probe_separates_well_separated_clusters() -> None:
    """Sanity check that the recipe actually learns something on an easy
    problem, not just that it runs without error."""
    x, labels, train, val, test = make_synthetic_problem(n_per_class=30, dim=6)
    cfg = ProbeConfig(epochs=200, batch_size=32, lr_grid=(1e-2,), lambda_grid=(0.0,))

    result = run_probe(
        name="synthetic_easy",
        x=x,
        labels=labels,
        train=train,
        val=val,
        test=test,
        n_classes=3,
        cfg=cfg,
        device=torch.device("cpu"),
        seeds=[0],
    )

    assert result.test_top1_mean > 0.9  # well-separated clusters should be near-trivial


def test_as_dict_round_trips_per_seed_results() -> None:
    x, labels, train, val, test = make_synthetic_problem()
    cfg = ProbeConfig(epochs=10, batch_size=16, lr_grid=(1e-2,), lambda_grid=(0.0,))

    result = run_probe(
        name="synthetic",
        x=x,
        labels=labels,
        train=train,
        val=val,
        test=test,
        n_classes=3,
        cfg=cfg,
        device=torch.device("cpu"),
        seeds=[0],
    )

    d = result.as_dict()
    assert d["name"] == "synthetic"
    assert isinstance(d["per_seed"], list)
    assert d["per_seed"][0]["seed"] == 0
