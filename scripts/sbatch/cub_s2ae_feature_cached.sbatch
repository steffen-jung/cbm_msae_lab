#!/usr/bin/env bash
# Selbst einreichen mit: sbatch scripts/sbatch/cub_s2ae_feature_cached.sbatch
#
# Run 3b des CFM-vs-Spatial-vs-S2AE-Vergleichs: identisch zu
# cub_s2ae_attention_cached.sbatch, aber Gruppierung per Feature-Clustering
# (grouping=feature -- cosine-Aehnlichkeit der rohen Encoder-Features statt
# Attention, siehe attention_grouping.py::cluster_patches_by_features).
# Zum Ausprobieren, falls die Attention-Cluster nicht granular genug auf dem
# Objekt selbst sind (siehe notebooks/attention_maps.ipynb).
#
# Baut einmalig den Feature-Gruppen-Cache (nur train-Split noetig, eigener
# Cache-Key als die Attention-Variante -- siehe compute_group_cache_key);
# nutzt denselben Roh-Encoder-Cache wie die anderen Runs (hier nicht nochmal
# extrahieren -- falls dieser Job als erster laeuft, vorher von Hand:
# `uv run scripts/extract_raw_features.py --split train`
# und `--split val`).

#SBATCH --job-name=cub-s2ae-feat
#SBATCH --partition=gpu-vram-12gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "$(pwd)"

uv run scripts/extract_attention_groups.py --split train --method feature

uv run scripts/train.py \
    train.cache.mode=cache_encoder \
    loss.group_sparsity.grouping=feature loss.group_sparsity.weight=0.1 \
    loss.exclusivity.grouping=feature loss.exclusivity.weight=0.1 \
    train.checkpoint_dir=outputs/checkpoints/cub_s2ae_feature \
    train.wandb.run_name=cub_s2ae_feature
