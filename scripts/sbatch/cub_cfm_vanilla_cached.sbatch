#!/usr/bin/env bash
# Selbst einreichen mit: sbatch scripts/sbatch/cub_cfm_vanilla_cached.sbatch
#
# Run 1 des CFM-vs-Spatial-vs-S2AE-Vergleichs (siehe cub_spatial_cached.sbatch,
# cub_s2ae_*_cached.sbatch). Baut den Roh-Encoder-Cache einmalig (train+val,
# CLIP-DINOiser-Output) und trainiert danach CFM vanilla auf CUB daraus -- der
# teure, eingefrorene Encoder-Forward laeuft dann nur einmal pro Split statt
# einmal pro Epoche. Da nichts Trainierbares zwischen Encoder und SAE sitzt,
# ist der Cache exakt das, was die SAE konsumiert. Diesen Job zuerst laufen
# lassen (oder zumindest zuerst durchlaufen lassen bis der Cache steht) --
# Run 2/3 nutzen denselben Roh-Encoder-Cache und extrahieren ihn nicht nochmal.
#
# Partition per --partition unten waehlen: gpu-vram-12gb (RTX 2080 Ti, 12GB)
# oder gpu-vram-48gb (RTX A6000/A40, 48GB); 12GB reicht hier vermutlich locker,
# auf 48gb ausweichen, falls's eng wird oder die 12gb-Queue voll ist.

#SBATCH --job-name=cub-cfm-vanilla
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
    train.cache.mode=cache_encoder \
    train.checkpoint_dir=outputs/checkpoints/cub_cfm_vanilla \
    train.wandb.run_name=cub_cfm_vanilla
