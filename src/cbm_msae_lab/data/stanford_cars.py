"""StanfordCars, ported from ``cfm_cars/cars_data.py``.

Reads the devkit .mat annotations directly: cars_train_annos.mat gives fname + class
for the 8144 training images, cars_test_annos_withlabels.mat the same for the 8041
test images, cars_meta.mat the 196 class names. Preprocessing matches CUB's exactly
(Resize -> CenterCrop -> ToTensor, no normalization -- the encoders normalize internally).
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import scipy.io as sio
import torchvision.transforms as T
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from cbm_msae_lab.data.base import register_dataset

Split = Literal["train", "val", "test"]


def _read_annos(mat_path: str, img_dir: str) -> list[tuple[str, int]]:
    """Returns [(relative path, 0-based label)] in the file's own order."""
    annos = sio.loadmat(mat_path)["annotations"][0]
    return [(os.path.join(img_dir, str(a["fname"][0])), int(a["class"][0][0]) - 1) for a in annos]


def _load_index(root: str) -> tuple[NDArray, NDArray, NDArray, list[str]]:
    """Returns (paths[N], labels[N] in [0, 195], is_train[N] bool, class_names[196])."""
    tr = _read_annos(os.path.join(root, "devkit", "cars_train_annos.mat"), "cars_train")
    te = _read_annos(os.path.join(root, "cars_test_annos_withlabels.mat"), "cars_test")
    meta = sio.loadmat(os.path.join(root, "devkit", "cars_meta.mat"))["class_names"][0]
    class_names = [str(c[0]) for c in meta]
    paths = np.array([p for p, _ in tr] + [p for p, _ in te])
    labels = np.array([y for _, y in tr] + [y for _, y in te])
    is_train = np.array([True] * len(tr) + [False] * len(te))
    return paths, labels, is_train, class_names


def _stratified_val(labels: NDArray, is_train: NDArray, frac: float, seed: int) -> NDArray:
    """A class-stratified `frac` fraction of the train split, held out for validation."""
    rng = np.random.RandomState(seed)
    val = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx = np.where(is_train & (labels == c))[0]
        n = max(1, int(round(frac * len(idx))))
        val[rng.permutation(idx)[:n]] = True
    return val


@register_dataset("stanford_cars")
class StanfordCarsDataset(Dataset):
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
        image = Image.open(os.path.join(self.root, self.paths[index])).convert("RGB")
        return self.transform(image), int(self.labels[index]), index
