"""Places365, following the official train-standard/val split layout torchvision's
own downloader produces (see scripts/download_places365_train.py).

Train pool = places365_train_standard.txt, read against `<root>/data_256_standard/`
(the folder torchvision's `Places365(split="train-standard", small=True)` extracts
to). `val_fraction` carves a stratified validation split out of that pool.
`split="test"` uses MIT's own labeled val split (`places365_val.txt` + `val_256/`,
already on disk) as our held-out test set, disjoint from anything used for training.
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


def _class_names(root: str) -> list[str]:
    """365 category names in label-index order, stripping the leading letter bucket
    (e.g. "/a/airfield" -> "airfield"), matching cfm_fg/data.py's convention."""
    names_by_index: dict[int, str] = {}
    with open(os.path.join(root, "categories_places365.txt")) as f:
        for line in f:
            if not line.strip():
                continue
            path_part, idx = line.rsplit(" ", 1)
            names_by_index[int(idx)] = path_part.lstrip("/").split("/", 1)[-1]
    return [names_by_index[i] for i in range(len(names_by_index))]


def _read_split_file(path: str) -> tuple[list[str], list[int]]:
    rel_paths, labels = [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            name, _, label = line.strip().rpartition(" ")
            rel_paths.append(name.lstrip("/"))
            labels.append(int(label))
    return rel_paths, labels


def _stratified_val(labels: NDArray, frac: float, seed: int) -> NDArray:
    """A class-stratified `frac` fraction of `labels`, held out for validation."""
    rng = np.random.RandomState(seed)
    val = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n = max(1, int(round(frac * len(idx))))
        val[rng.permutation(idx)[:n]] = True
    return val


@register_dataset("places365")
class Places365Dataset(Dataset):
    def __init__(
        self,
        root: str,
        split: Split = "train",
        image_size: int = 224,
        val_fraction: float = 0.02,
        seed: int = 0,
    ) -> None:
        self.root = root
        self.split = split
        self.class_names = _class_names(root)

        if split == "test":
            rel_paths, labels = _read_split_file(os.path.join(root, "places365_val.txt"))
            self.image_dir = os.path.join(root, "val_256")
            self.paths = np.array(rel_paths)
            self.labels = np.array(labels)
        elif split in ("train", "val"):
            rel_paths, labels = _read_split_file(os.path.join(root, "places365_train_standard.txt"))
            self.image_dir = os.path.join(root, "data_256_standard")
            paths, labels_arr = np.array(rel_paths), np.array(labels)
            val_mask = _stratified_val(labels_arr, val_fraction, seed)
            mask = val_mask if split == "val" else ~val_mask
            self.paths, self.labels = paths[mask], labels_arr[mask]
        else:
            raise ValueError(f"unknown split {split!r}")

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
        image = Image.open(os.path.join(self.image_dir, self.paths[index])).convert("RGB")
        return self.transform(image), int(self.labels[index]), index
