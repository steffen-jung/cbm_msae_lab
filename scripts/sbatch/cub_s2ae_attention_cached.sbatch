#!/usr/bin/env bash
# Selbst einreichen mit: sbatch scripts/sbatch/cub_s2ae_attention_cached.sbatch
#
# Run 3a des CFM-vs-Spatial-vs-S2AE-Vergleichs: Standard-Rekonstruktion PLUS S2AE-Style group-sparsity/exclusivity (additiv, keine
# Alternative -- siehe README), Gruppierung per Attention-Clustering
# (grouping=attention). Siehe cub_s2ae_feature_cached.sbatch fuer dieselbe
# Loss-Kombination mit Feature-Clustering statt Attention -- eigener
# checkpoint_dir/wandb-run, damit beide vergleichbar bleiben.
#
# Baut einmalig den Attention-Gruppen-Cache (nur train-Split noetig, siehe
# attention_grouping.py); nutzt denselben Roh-Encoder-Cache wie die anderen
# Runs (hier nicht nochmal extrahieren -- falls dieser Job als erster laeuft,
# vorher von Hand: `uv run scripts/extract_raw_features.py --split train`
# und `--split val`).

#SBATCH --job-name=cub-s2ae-attn
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

uv run scripts/train.py \
    train.cache.mode=cache_encoder \
    loss.group_sparsity.grouping=attention loss.group_sparsity.weight=0.1 \
    loss.exclusivity.grouping=attention loss.exclusivity.weight=0.1 \
    train.checkpoint_dir=outputs/checkpoints/cub_s2ae_attention \
    train.wandb.run_name=cub_s2ae_attention
