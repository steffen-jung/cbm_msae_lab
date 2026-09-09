#!/usr/bin/env bash
# Selbst einreichen mit: sbatch scripts/sbatch/cub_k1_cfm_vanilla_cached.sbatch
#
# Baut den Roh-Encoder-Cache einmalig (train+val, nur CLIP-DINOiser-Output,
# VOR der Projektion) und trainiert danach CFM vanilla auf CUB daraus -- der
# teure, eingefrorene Encoder-Forward laeuft dann nur einmal pro Split statt
# einmal pro Epoche. Die 1x1-Projektion selbst laeuft weiterhin live mit
# Gradient jeden Schritt und wird ganz normal mittrainiert.
#
# Partition per --partition unten waehlen: gpu-vram-12gb (RTX 2080 Ti, 12GB)
# oder gpu-vram-48gb (RTX A6000/A40, 48GB). Fuer den 1x1-Kernel-Fall reicht
# ein 12GB-Slot vermutlich locker; auf 48gb ausweichen, falls's eng wird oder
# die 12gb-Queue voll ist.

#SBATCH --job-name=cub-k1-cfm-vanilla
#SBATCH --partition=gpu-vram-12gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
# NOT `dirname "$0"`: SLURM copies the batch script to /var/spool/slurmd/job.../
# before running it, so `$0` points there, not into the repo. `SLURM_SUBMIT_DIR`
# is the directory `sbatch` was invoked from -- always what we actually want.
cd "$SLURM_SUBMIT_DIR"
echo "$(pwd)"

uv run scripts/extract_raw_features.py --split train
uv run scripts/extract_raw_features.py --split val

uv run scripts/train.py \
    projection=k1 \
    train.cache.mode=cache_encoder \
    train.checkpoint_dir=outputs/checkpoints/cub_k1_cfm_vanilla \
    wandb.run_name=cub_k1_cfm_vanilla
