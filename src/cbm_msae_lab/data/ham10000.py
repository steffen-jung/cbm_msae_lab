"""HAM10000 (skin lesion dermoscopy images), from the download this repo's
scripts/download_ham10000.py performs into `<root>/images/` + `<root>/HAM10000_metadata.csv`.

Splitting is lesion-grouped, not just class-stratified like the other datasets: HAM10000
has multiple images per `lesion_id` (repeat photos of the same lesion), so splitting by
image would leak the same lesion across train/val/test. Groups are assigned to a split
first (stratified by each lesion's `dx`), then every image of a lesion follows its group.

`age`/`sex`/`localization` are exposed as extra per-sample attributes (natural concept
labels for a CBM setting) alongside the `dx` classification target, but aren't wired into
the loss/metric pipeline here.
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

_DX_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def _load_metadata(root: str) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Returns (image_ids, lesion_ids, labels, age, sex, localization) for every row."""
    dx_to_idx = {name: i for i, name in enumerate(_DX_CLASSES)}
    image_ids, lesion_ids, labels, age, sex, localization = [], [], [], [], [], []
    with open(os.path.join(root, "HAM10000_metadata.csv"), newline="") as f:
        for row in csv.DictReader(f):
            image_ids.append(row["image_id"])
            lesion_ids.append(row["lesion_id"])
            labels.append(dx_to_idx[row["dx"]])
            age.append(float(row["age"]) if row["age"] else float("nan"))
            sex.append(row["sex"])
            localization.append(row["localization"])
    return (
        np.array(image_ids),
        np.array(lesion_ids),
        np.array(labels),
        np.array(age),
        np.array(sex),
        np.array(localization),
    )


def _lesion_grouped_split(
    lesion_ids: NDArray, labels: NDArray, val_fraction: float, test_fraction: float, seed: int
) -> NDArray:
    """Assigns every row a split ("train"/"val"/"test"), grouped by lesion_id so no
    lesion's images cross a split boundary, stratified by each lesion's dx class."""
    rng = np.random.RandomState(seed)
    unique_lesions, first_idx = np.unique(lesion_ids, return_index=True)
    lesion_labels = labels[first_idx]

    split_by_lesion: dict[str, str] = {}
    for c in np.unique(lesion_labels):
        lesions_c = unique_lesions[lesion_labels == c]
        order = rng.permutation(len(lesions_c))
        n_val = max(1, int(round(val_fraction * len(lesions_c))))
        n_test = max(1, int(round(test_fraction * len(lesions_c))))
        for i in order[:n_val]:
            split_by_lesion[lesions_c[i]] = "val"
        for i in order[n_val : n_val + n_test]:
            split_by_lesion[lesions_c[i]] = "test"
        for i in order[n_val + n_test :]:
            split_by_lesion[lesions_c[i]] = "train"

    return np.array([split_by_lesion[lid] for lid in lesion_ids])


@register_dataset("ham10000")
class HAM10000Dataset(Dataset):
    def __init__(
        self,
        root: str,
        split: Split = "train",
        image_size: int = 224,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.root = root
        self.split = split
        self.class_names = list(_DX_CLASSES)

        image_ids, lesion_ids, labels, age, sex, localization = _load_metadata(root)
        split_of_row = _lesion_grouped_split(lesion_ids, labels, val_fraction, test_fraction, seed)
        mask = split_of_row == split

        self.image_ids = image_ids[mask]
        self.labels = labels[mask]
        self.age = age[mask]
        self.sex = sex[mask]
        self.localization = localization[mask]

        self.transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        """Returns (image [3, H, W] in [0, 1], class label, dataset index)."""
        image = Image.open(os.path.join(self.root, "images", f"{self.image_ids[index]}.jpg")).convert("RGB")
        return self.transform(image), int(self.labels[index]), index
