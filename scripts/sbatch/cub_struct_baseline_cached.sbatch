#!/usr/bin/env bash
# Selbst einreichen mit: sbatch scripts/sbatch/cub_struct_baseline_cached.sbatch
#
# Arm 1: BatchTopK-Baseline. Keine Struktur-Overrides -- identisch zu cub_cfm_vanilla, hier aber unter eigenem Namen, damit alle vier Arme dieselben Metriken im selben Lauf-Satz schreiben.
#
# Teil der Struktur-Ablation (Group-TopK / Participation-Ratio-Loss), siehe
# scripts/run_structure_ablation.sh fuer alle vier Arme und deren Begruendung.
# Die vier Arme unterscheiden sich NUR in den Struktur-Overrides -- Encoder,
# Dataset, SAE-Hyperparameter und Eval-Protokoll sind identisch, damit FVU, L0,
# Monosemanticity, Region-Consistency und Aktivierungsfrequenz vergleichbar sind.
#
# Nutzt den gemeinsamen Roh-Encoder-Cache; falls dieser Job als erster laeuft,
# vorher von Hand `uv run scripts/extract_raw_features.py --split train` und
# `--split val`. Region-Consistency wird explizit auf den Attention-Clustern
# gemessen (train.eval.region_grouping=attention), nicht auf 2x2-Kacheln, und
# zwar auf ALLEN vier Armen (auch der Baseline) -- sonst waere val/region_pr
# zwischen den Armen nicht vergleichbar. Braucht deshalb den Attention-Cluster-
# Cache fuer BEIDE Splits, unten idempotent gebaut.

#SBATCH --job-name=cub-struct-base
#SBATCH --partition=gpu-vram-12gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "$(pwd)"

uv run scripts/extract_attention_groups.py --split train --method attention
uv run scripts/extract_attention_groups.py --split val --method attention

uv run scripts/train.py \
    train.cache.mode=cache_encoder \
    train.eval.region_grouping=attention \
    train.checkpoint_dir=outputs/checkpoints/cub_struct_baseline \
    train.wandb.run_name=cub_struct_baseline
