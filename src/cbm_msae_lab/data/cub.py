"""CUB-200-2011, ported from ``cfm_cub/cub_data.py``.

Reads the raw CUB text files directly (images.txt / image_class_labels.txt /
train_test_split.txt / classes.txt) rather than any pickled split, since only
image + class label are needed here. Preprocessing matches CFM's own
CLIP-DINOiser preprocessing exactly (Resize -> CenterCrop -> ToTensor, no
normalization -- the encoders normalize internally), so cached features stay
compatible with both CFM's original checkpoint and this repo's encoders.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import torchvision.transforms as T
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from cbm_msae_lab.data.base import register_dataset

Split = Literal["train", "val", "test"]


def _read_pairs(root: str, fname: str) -> list[list[str]]:
    with open(os.path.join(root, fname)) as f:
        return [ln.strip().split(None, 1) for ln in f if ln.strip()]


def _load_index(root: str) -> tuple[NDArray, NDArray, NDArray, list[str]]:
    """Returns (paths[N], labels[N] in [0, 199], is_train[N] bool, class_names[200])."""
    paths = {int(i): p for i, p in _read_pairs(root, "images.txt")}
    labels = {int(i): int(c) - 1 for i, c in _read_pairs(root, "image_class_labels.txt")}
    split = {int(i): int(s) for i, s in _read_pairs(root, "train_test_split.txt")}
    ids = sorted(paths)
    class_names = [n.split(".", 1)[1].replace("_", " ") for _, n in _read_pairs(root, "classes.txt")]
    return (
        np.array([paths[i] for i in ids]),
        np.array([labels[i] for i in ids]),
        np.array([split[i] == 1 for i in ids]),
        class_names,
    )


def _stratified_val(labels: NDArray, is_train: NDArray, frac: float, seed: int) -> NDArray:
    """A class-stratified `frac` fraction of the train split, held out for validation."""
    rng = np.random.RandomState(seed)
    val = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx = np.where(is_train & (labels == c))[0]
        n = max(1, int(round(frac * len(idx))))
        val[rng.permutation(idx)[:n]] = True
    return val


@register_dataset("cub")
class CUBDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: Split = "train",
        image_size: int = 224,
        val_fraction: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.root = root
        self.split = split

        paths, labels, is_train, class_names = _load_index(root)
        val_mask = _stratified_val(labels, is_train, val_fraction, seed)
        if split == "train":
            mask = is_train & ~val_mask
        elif split == "val":
            mask = val_mask
        elif split == "test":
            mask = ~is_train
        else:
            raise ValueError(f"unknown split {split!r}")

        self.paths = paths[mask]
        self.labels = labels[mask]
        self.class_names = class_names

        self.transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        """Returns (image [3, H, W] in [0, 1], class label, dataset index)."""
        image = Image.open(os.path.join(self.root, "images", self.paths[index])).convert("RGB")
        return self.transform(image), int(self.labels[index]), index
