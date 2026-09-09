"""A small dataset registry so adding a second dataset later is a one-file addition.

Every entry maps a short name (matching a `conf/dataset/*.yaml` file) to the
`Dataset` class Hydra's `_target_` should instantiate. This registry itself
is not required for Hydra to work (`_target_` already names the class
directly), but it gives ``scripts/extract_raw_features.py`` and other tooling
a single place to enumerate "every dataset this codebase knows about" without
importing every dataset module.
"""

from __future__ import annotations

from torch.utils.data import Dataset

DATASET_REGISTRY: dict[str, type[Dataset]] = {}


def register_dataset(name: str):
    """Class decorator: `@register_dataset("cub")` on a `Dataset` subclass."""

    def _decorator(cls: type[Dataset]) -> type[Dataset]:
        DATASET_REGISTRY[name] = cls
        return cls

    return _decorator
