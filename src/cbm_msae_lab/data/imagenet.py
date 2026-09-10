"""ImageNet-1k, ported from ``cfm_cub/imagenet_data.py``.

Walks the ILSVRC `train`/`val` synset folders directly (sorted-synset order, matching
standard torchvision/ImageFolder ordering). `val_fraction` carves a stratified
validation split out of the 1.28M-image train folder; the official, already-categorized
`val` folder (50k images, real per-image labels) is exposed as `split="test"` -- it is
ImageNet's own held-out evaluation set, disjoint from anything used for SAE training.
"""

from __future__ import annotations

import json
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

_IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")


def _class_names(root: str) -> list[str]:
    """Human-readable class names in sorted-synset order, from imagenet_class_index.json."""
    with open(os.path.join(root, "imagenet_class_index.json")) as f:
        index = json.load(f)
    by_synset = {synset: name.replace("_", " ") for _, (synset, name) in index.items()}
    return [by_synset[s] for s in sorted(by_synset)]


def _load_split(root: str, split_dir: str) -> tuple[NDArray, NDArray]:
    """Returns (paths[N], labels[N] in [0, 999]) for one ILSVRC synset-folder split."""
    d = os.path.join(root, split_dir)
    synsets = sorted(s for s in os.listdir(d) if s.startswith("n") and os.path.isdir(os.path.join(d, s)))
    if len(synsets) != 1000:
        raise RuntimeError(f"expected 1000 synsets in {d}, got {len(synsets)}")
    paths, labels = [], []
    for label, synset in enumerate(synsets):
        sd = os.path.join(d, synset)
        for fname in os.listdir(sd):
            if fname.lower().endswith(_IMAGE_EXTENSIONS):
                paths.append(os.path.join(sd, fname))
                labels.append(label)
    return np.array(paths), np.array(labels)


def _stratified_val(labels: NDArray, frac: float, seed: int) -> NDArray:
    """A class-stratified `frac` fraction of `labels`, held out for validation."""
    rng = np.random.RandomState(seed)
    val = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n = max(1, int(round(frac * len(idx))))
        val[rng.permutation(idx)[:n]] = True
    return val


@register_dataset("imagenet")
class ImageNetDataset(Dataset):
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
            self.paths, self.labels = _load_split(root, "val")
        elif split in ("train", "val"):
            paths, labels = _load_split(root, "train")
            val_mask = _stratified_val(labels, val_fraction, seed)
            mask = val_mask if split == "val" else ~val_mask
            self.paths, self.labels = paths[mask], labels[mask]
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
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), int(self.labels[index]), index
