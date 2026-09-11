"""Append-only JSONL metrics log so a run is inspectable without wandb.

One object per line. Train-step and eval rows share the same file; keys are
already namespaced (`train/...`, `val/...`) the same way wandb receives them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return float(value)


class JsonlMetricsWriter:
    """Writes one JSON object per line to ``path``.

    ``overwrite=True`` (default) truncates at construction so a fresh training
    launch does not append onto a previous run that reused the same
    ``checkpoint_dir``. Pass ``overwrite=False`` only when intentionally
    resuming into an existing log.
    """

    def __init__(self, path: str | Path, *, overwrite: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def log(self, metrics: dict[str, Any], *, step: int, epoch: int | None = None) -> None:
        row: dict[str, Any] = {"step": int(step)}
        if epoch is not None:
            row["epoch"] = int(epoch)
        for key, value in metrics.items():
            row[key] = _to_jsonable(value)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            f.flush()
