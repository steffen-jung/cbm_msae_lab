"""One-off download of HAM10000 (skin lesion images + metadata) from Harvard Dataverse
(doi:10.7910/DVN/DBW86T, public, no login required).

Fetches the two image archives and the metadata table, unzips both image parts into a
single flat `<root>/images/` folder (each `ISIC_<id>.jpg`), and writes
`<root>/HAM10000_metadata.csv`. Safe to re-run -- skips files/extraction already done.

Usage: uv run scripts/download_ham10000.py
"""

import csv
import os
import urllib.request
import zipfile

ROOT = "/ceph/faroesch/datasets/ham10000"
BASE_URL = "https://dataverse.harvard.edu/api/access/datafile/"
# {local filename: dataverse file id}
FILES = {
    "HAM10000_images_part_1.zip": 3172585,
    "HAM10000_images_part_2.zip": 3172584,
    "HAM10000_metadata.tab": 4338392,
}


def _download(fname: str, file_id: int) -> str:
    dest = os.path.join(ROOT, fname)
    if os.path.exists(dest):
        print(f"[skip] {fname} already present")
        return dest
    print(f"[get]  {fname}")
    tmp = dest + ".part"
    urllib.request.urlretrieve(f"{BASE_URL}{file_id}", tmp)
    os.rename(tmp, dest)
    return dest


def _extract_images(zip_path: str, images_dir: str) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = os.path.basename(member)
            if not name.lower().endswith(".jpg"):
                continue
            out_path = os.path.join(images_dir, name)
            if os.path.exists(out_path):
                continue
            with zf.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())


def _tab_to_csv(tab_path: str, csv_path: str) -> None:
    with open(tab_path, newline="") as fin, open(csv_path, "w", newline="") as fout:
        reader = csv.reader(fin, delimiter="\t", quotechar='"')
        writer = csv.writer(fout)
        for row in reader:
            writer.writerow(row)


def main() -> None:
    os.makedirs(ROOT, exist_ok=True)
    images_dir = os.path.join(ROOT, "images")
    os.makedirs(images_dir, exist_ok=True)

    for fname, file_id in FILES.items():
        _download(fname, file_id)

    for part in ("HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"):
        print(f"[unzip] {part}")
        _extract_images(os.path.join(ROOT, part), images_dir)

    _tab_to_csv(os.path.join(ROOT, "HAM10000_metadata.tab"), os.path.join(ROOT, "HAM10000_metadata.csv"))

    n_images = len([f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")])
    print(f"[ok] {n_images} images in {images_dir}")


if __name__ == "__main__":
    main()
