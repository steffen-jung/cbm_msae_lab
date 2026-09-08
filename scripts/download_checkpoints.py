"""Fetches/copies the pretrained encoder checkpoints this repo needs.

CLIP-DINOiser's checkpoint has no stable public download URL (see the module
docstring in `src/cbm_msae_lab/encoders/clip_dinoiser_backend/clip_dinoiser.py`)
-- CFM ships it directly in its own repo, so this script copies it from a
sibling CFM checkout rather than downloading it. DINOv3 (via timm) and AnyUp
(via torch.hub) both auto-download on first use and need no separate step
here; this script does not touch them.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Only the laion2b-backbone checkpoint is known to exist anywhere on this
# machine (see the CFM exploration this repo's plan was based on) -- there is
# no `checkpoints_openai/last.pt` sibling despite CFM having a config
# (`clip_dinoiser_openai.yaml`) that references one.
DEFAULT_SOURCE = Path("/home/faroesch/workspace/CFM/cfm/clip_dinoiser_backbone/checkpoints/last.pt")
DEFAULT_DEST = REPO_ROOT / "checkpoints" / "clip_dinoiser" / "laion2b.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to CFM's clip_dinoiser last.pt checkpoint.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Where to copy it inside this repo.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(
            f"CLIP-DINOiser checkpoint not found at {args.source}. This codebase does not know a "
            "stable public download URL for it (see encoders/clip_dinoiser_backend/clip_dinoiser.py) "
            "-- pass --source pointing at a CFM checkout's cfm/clip_dinoiser_backbone/checkpoints/last.pt."
        )

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, args.dest)
    print(f"Copied CLIP-DINOiser checkpoint: {args.source} -> {args.dest}")

    openai_dest = REPO_ROOT / "checkpoints" / "clip_dinoiser" / "openai.pt"
    print(
        f"NOTE: no OpenAI-CLIP variant checkpoint was found anywhere on this machine "
        f"(would go to {openai_dest}). CFM's own `clip_dinoiser_openai.yaml` config references "
        "one, but its checkpoint file does not exist locally and no public URL for it is known. "
        "If you obtain one, copy it there and pass encoder.clip_backbone=maskclip_openai."
    )
    print(
        "DINOv3 and AnyUp need no manual download: they auto-fetch on first use "
        "(timm from HuggingFace, torch.hub from github.com/wimmerth/anyup)."
    )


if __name__ == "__main__":
    main()
