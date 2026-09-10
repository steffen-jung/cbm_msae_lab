"""One-off download of the official Places365 train-standard split (small, 256px).

`val_256` (36,500 labeled images) is already on disk; this fetches the ~24GB training
split the same way, via torchvision's own Places365 downloader (URLs/md5s built in).
Extracts to `<root>/data_256_standard/`. Safe to re-run -- torchvision skips
already-downloaded/extracted archives.

Usage: uv run scripts/download_places365_train.py
"""

import torchvision

ROOT = "/ceph/faroesch/datasets/places365"


def main() -> None:
    ds = torchvision.datasets.Places365(root=ROOT, split="train-standard", small=True, download=True)
    print(f"[ok] {len(ds)} images, {len(ds.classes)} classes")
    print(f"     images_dir = {ds.images_dir}")


if __name__ == "__main__":
    main()
