"""Local JSONL metrics writer (wandb-independent)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cbm_msae_lab.metrics_logging import JsonlMetricsWriter


def test_jsonl_metrics_writer_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    writer = JsonlMetricsWriter(path)
    writer.log({"train/loss": 1.5, "train/epoch": 0}, step=0, epoch=0)
    writer.log({"val/fvu": torch.tensor(0.25)}, step=10, epoch=0)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    row1 = json.loads(lines[1])
    assert row0 == {"step": 0, "epoch": 0, "train/loss": 1.5, "train/epoch": 0}
    assert row1["step"] == 10
    assert row1["val/fvu"] == 0.25


def test_jsonl_metrics_writer_overwrite_by_default(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    JsonlMetricsWriter(path).log({"a": 1}, step=0)
    JsonlMetricsWriter(path).log({"b": 2}, step=1)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["b"] == 2
