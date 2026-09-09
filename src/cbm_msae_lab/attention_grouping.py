"""S2AE-style attention + spatial-proximity patch grouping.

Adapted from S2AE's public repo (github.com/liaoweiduo/s2ae,
``SaeTrainer.clustering()``), which the paper "When Structured Sparse
Autoencoders Learn Consistent Concepts Across Modalities" (arXiv:2607.08605)
uses to form the patch groups that its group-sparsity/exclusivity losses
operate on. S2AE extracts its attention from an LLM decoder's causal
self-attention (their SAE sits inside a VLM's language-model stack); this
codebase has no LLM decoder, only frozen ViT encoders, so the one substitution
made here is: attention comes from the ViT encoder's own self-attention
(``Encoder(extract_attention=True)``, see ``encoders/base.py``) instead. The
clustering math itself -- symmetrize, invert to a distance, combine with
spatial proximity, agglomerative clustering -- is unchanged, including
S2AE's own default hyperparameters (``n_clusters=20``, ``spatial_coeff=0.02``).

This is an *alternative* to the default, much cheaper tile-based grouping in
``losses/tiling.py`` (selected via ``loss.group_sparsity.grouping="attention"``
/ ``loss.exclusivity.grouping="attention"``), never on by default: clustering
is per-image (unlike a shared spatial tile) and needs `scikit-learn`'s
`AgglomerativeClustering`, too slow to run inside the training loop on every
step -- so this module also provides the on-disk caching
(`GroupLabelWriter`/`CachedGroupLabelDataset`) that lets it be computed once
per image and reused, exactly like `caching.py`'s activation cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import AgglomerativeClustering
from torch import Tensor
from torch.utils.data import Dataset


def _manhattan_distance_matrix(grid_h: int, grid_w: int) -> NDArray[np.float64]:
    """Pairwise Manhattan distance between patch-grid coordinates, normalized to [0, 1]. -> [P, P], P = grid_h*grid_w."""
    n = grid_h * grid_w
    idx = np.arange(n)
    rows, cols = idx // grid_w, idx % grid_w
    dist = np.abs(rows[:, None] - rows[None, :]) + np.abs(cols[:, None] - cols[None, :])
    max_dist = dist.max()
    return dist / max_dist if max_dist > 0 else dist.astype(np.float64)


def cluster_patches(
    attention: Tensor, grid_h: int, grid_w: int, n_clusters: int = 20, spatial_coeff: float = 0.02
) -> NDArray[np.int64]:
    """One image's patch-to-patch attention -> per-patch cluster ids.

    attention: [P, P] (P = grid_h * grid_w) -> labels: [P], values in [0, min(n_clusters, P)).

    Ported from S2AE's ``SaeTrainer.clustering()``:
        A_sym     = (A + A^T) / 2                                    # symmetrize
        d_attn    = normalize_to_[0,1]( max(A_sym) - A_sym )         # similarity -> distance
        d_spatial = normalize_to_[0,1]( manhattan_distance(grid) )
        distance  = d_attn * (d_spatial ** spatial_coeff)
        labels    = AgglomerativeClustering(metric="precomputed", linkage="average",
                                             n_clusters=min(n_clusters, P)).fit_predict(distance)
    """
    a = attention.detach().cpu().numpy().astype(np.float64)
    a_sym = (a + a.T) / 2

    d_attn = a_sym.max() - a_sym
    d_min, d_max = d_attn.min(), d_attn.max()
    d_range = d_max - d_min
    d_attn = (d_attn - d_min) / d_range if d_range > 1e-12 else np.zeros_like(d_attn)

    d_spatial = _manhattan_distance_matrix(grid_h, grid_w)
    distance = d_attn * (d_spatial**spatial_coeff)

    n_patches = distance.shape[0]
    agnes = AgglomerativeClustering(n_clusters=min(n_clusters, n_patches), metric="precomputed", linkage="average")
    return agnes.fit_predict(distance).astype(np.int64)


def onehot_l2_aggregate(f_img: Tensor, group_labels: Tensor, n_clusters: int) -> Tensor:
    """The attention-grouping analogue of ``losses/tiling.py::tile_l2_norm``.

    f_img: [B, P, dict_size] (per-patch SAE latents), group_labels: [B, P] (int64,
    per-image cluster ids in [0, n_clusters)) -> [B, n_clusters, dict_size]:
    sqrt(sum of squares) of latents within each (image, cluster) group -- this
    is S2AE's group-level activation profile ``s^g``.

    Clusters differ per image (unlike a shared spatial tile, which is the same
    partition for every image), so a clean reshape doesn't apply here; a
    batched one-hot matmul does the equivalent per-image masked sum instead.
    Clusters that end up empty for a given image (e.g. if `n_clusters` exceeds
    the actual patch/cluster count) simply contribute an all-zero row.
    """
    onehot = F.one_hot(group_labels, num_classes=n_clusters).to(f_img.dtype)  # [B, P, n_clusters]
    sum_sq = torch.einsum("bpg,bpj->bgj", onehot, f_img.pow(2))  # [B, n_clusters, dict_size]
    return sum_sq.clamp_min(0).sqrt()


def compute_group_cache_key(
    encoder_cfg: DictConfig, n_clusters: int, spatial_coeff: float, dataset_name: str, split: str, image_size: int
) -> str:
    """Analogous to `caching.compute_raw_cache_key`, but its own namespace: group
    labels depend on the frozen encoder's attention and the clustering
    hyperparameters, not on anything the raw-feature cache key covers.
    """
    payload: dict[str, Any] = {
        "encoder": OmegaConf.to_container(encoder_cfg, resolve=True),
        "n_clusters": n_clusters,
        "spatial_coeff": spatial_coeff,
        "dataset": dataset_name,
        "split": split,
        "image_size": image_size,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class GroupLabelWriter:
    """Writes per-image ``[P]`` cluster-label arrays to a memmap, one image at a time."""

    def __init__(self, cache_dir: str, cache_key: str, num_samples: int, num_patches: int) -> None:
        self.dir = Path(cache_dir) / "group_labels" / cache_key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shape = (num_samples, num_patches)
        self.labels = np.memmap(self.dir / "labels.dat", dtype=np.int64, mode="w+", shape=self.shape)
        self._cursor = 0

    def write(self, labels: NDArray[np.int64]) -> None:
        """labels: [P] for one image."""
        self.labels[self._cursor] = labels
        self._cursor += 1

    def finalize(self, resolved_config: dict) -> None:
        if self._cursor != self.shape[0]:
            raise RuntimeError(f"wrote {self._cursor} label rows but expected {self.shape[0]} -- cache is incomplete")
        self.labels.flush()
        meta = {"shape": list(self.shape), "dtype": "int64", "num_samples": self.shape[0], "config": resolved_config}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))


class CachedGroupLabelDataset(Dataset):
    """Reads back what `GroupLabelWriter` wrote. Supports both per-index
    `__getitem__` (standard `Dataset` interface) and a batched `get_batch`
    (used by the training loop, which already has a batch of dataset indices
    from `ActivationLoader`)."""

    def __init__(self, cache_dir: str, cache_key: str) -> None:
        self.dir = Path(cache_dir) / "group_labels" / cache_key
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no attention-grouping cache found at {self.dir} (missing meta.json) -- run "
                "scripts/extract_attention_groups.py first. loss.*.grouping='attention' never "
                "computes clustering on the fly inside the training loop."
            )
        meta = json.loads(meta_path.read_text())
        self.shape = tuple(meta["shape"])
        self.labels = np.memmap(self.dir / "labels.dat", dtype=np.int64, mode="r", shape=self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: int) -> Tensor:
        return torch.from_numpy(np.array(self.labels[index]))  # [P]

    def get_batch(self, idx: Tensor) -> Tensor:
        """idx: [B] dataset indices -> [B, P] group labels."""
        return torch.from_numpy(np.array(self.labels[idx.cpu().numpy()]))
