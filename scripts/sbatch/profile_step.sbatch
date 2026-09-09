#!/usr/bin/env bash
# Selbst einreichen mit: sbatch scripts/sbatch/profile_step.sbatch
#
# Zerlegt einen einzelnen Trainings-Schritt in Einzelphasen (Data, Encoder+
# Projection, SAE-Encode-Matmul vs. BatchTopK-Sortierung, Decode, Loss-Terme,
# Backward, Optimizer-Steps) und gibt eine Tabelle mit Zeit pro Phase aus.
# Nutzt den bereits vorhandenen Raw-Encoder-Cache (train.cache.mode=cache_encoder),
# damit die Zahlen die reine SAE/Optimizer-Zeit zeigen, nicht den Encoder-Forward.
#
# Ergebnis steht danach in outputs/slurm/profile-step-<jobid>.out

#SBATCH --job-name=profile-step
#SBATCH --partition=gpu-vram-12gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "$(pwd)"

uv run scripts/profile_step.py --steps 30 --warmup 5 \
    train.cache.mode=cache_encoder \
    train.batch_size=64 \
    sae.warmup_steps=1
