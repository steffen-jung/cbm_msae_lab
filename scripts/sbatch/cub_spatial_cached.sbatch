#!/usr/bin/env bash
# Selbst einreichen mit: sbatch scripts/sbatch/cub_spatial_cached.sbatch
#
# Run 2 des CFM-vs-Spatial-vs-S2AE-Vergleichs: Spatial-Loss
# STATT Standard-Rekonstruktion (reconstruction.weight=0.0, da scale_spatial's
# s=1-Term bereits die volle Patch-Rekonstruktion enthaelt -- siehe
# scale_spatial.py-Docstring). Nutzt den Roh-Encoder-Cache, der von
# cub_cfm_vanilla_cached.sbatch bereits gebaut wurde (gleicher Encoder,
# gleicher Cache-Key -- hier NICHT nochmal extrahieren, um den teuren
# Encoder-Forward nicht dreifach zu bezahlen). Falls dieser Job als erster
# laeuft: einmal `uv run scripts/extract_raw_features.py --split train` und
# `--split val` vorher von Hand ausfuehren.

#SBATCH --job-name=cub-spatial
#SBATCH --partition=gpu-vram-12gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "$(pwd)"

uv run scripts/train.py \
    train.cache.mode=cache_encoder \
    loss.reconstruction.weight=0.0 \
    loss.scale_spatial.weight=1.0 \
    train.checkpoint_dir=outputs/checkpoints/cub_spatial \
    train.wandb.run_name=cub_spatial
