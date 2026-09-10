"""Fitzpatrick17k, a dermatology dataset spanning Fitzpatrick skin types
(Groh et al., https://github.com/mattgroh/fitzpatrick17k).

The dataset ships only `fitzpatrick17k.csv` (md5hash, label, nine_partition_label,
three_partition_label, fitzpatrick_scale, url); images must be fetched from
their source URLs by `download_images.py` (dermaamin.com / atlasdermatologico.com.br)
into `root/images/<md5hash>.jpg`. Some source URLs 404 or otherwise fail, so the
index only includes rows whose image actually downloaded. `label_col` selects
which CSV column to use as the classification target -- the fine-grained
`label` (114 classes) by default, or `nine_partition_label` / `three_partition_label`
for the coarser groupings the paper also reports.
"""

from __future__ import annotations

import csv
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
LabelCol = Literal["label", "nine_partition_label", "three_partition_label"]


def _load_index(root: str, label_col: str) -> tuple[NDArray, NDArray, list[str]]:
    """Returns (md5hashes[N], labels[N], class_names) for rows whose image exists on disk."""
    image_dir = os.path.join(root, "images")
    with open(os.path.join(root, "fitzpatrick17k.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if os.path.exists(os.path.join(image_dir, f"{r['md5hash']}.jpg"))]
    class_names = sorted({r[label_col] for r in rows})
    label_of = {c: i for i, c in enumerate(class_names)}
    hashes = np.array([r["md5hash"] for r in rows])
    labels = np.array([label_of[r[label_col]] for r in rows])
    return hashes, labels, class_names


def _stratified_split(labels: NDArray, val_frac: float, test_frac: float, seed: int) -> tuple[NDArray, NDArray]:
    """Class-stratified (val_mask, test_mask) partition of `labels`, disjoint from each other."""
    rng = np.random.RandomState(seed)
    val = np.zeros(len(labels), dtype=bool)
    test = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        n_val = max(1, int(round(val_frac * len(idx))))
        n_test = max(1, int(round(test_frac * len(idx))))
        val[idx[:n_val]] = True
        test[idx[n_val : n_val + n_test]] = True
    return val, test


@register_dataset("fitzpatrick17k")
class Fitzpatrick17kDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: Split = "train",
        image_size: int = 224,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 0,
        label_col: LabelCol = "label",
    ) -> None:
        self.root = root
        self.split = split

        hashes, labels, class_names = _load_index(root, label_col)
        val_mask, test_mask = _stratified_split(labels, val_fraction, test_fraction, seed)
        if split == "train":
            mask = ~(val_mask | test_mask)
        elif split == "val":
            mask = val_mask
        elif split == "test":
            mask = test_mask
        else:
            raise ValueError(f"unknown split {split!r}")

        self.hashes = hashes[mask]
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
        return len(self.hashes)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        """Returns (image [3, H, W] in [0, 1], class label, dataset index)."""
        image = Image.open(os.path.join(self.root, "images", f"{self.hashes[index]}.jpg")).convert("RGB")
        return self.transform(image), int(self.labels[index]), index
