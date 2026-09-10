#!/usr/bin/env bash
# Selbst einreichen mit: sbatch scripts/sbatch/cub_pr_feature_cached.sbatch
#
# CLIP-DINOiser auf CUB mit Standard-Rekonstruktion + AuxK (Dead-Latent-Revival)
# + Participation-Ratio-Loss, gruppiert nach Feature-Clustern.
#
# Encoder (clip_dinoiser), Rekonstruktions-Loss (weight 1.0) und AuxK
# (weight 0.03125 = CFM's auxk_alpha 1/32) sind bereits Defaults und stehen
# unten nur der Nachvollziehbarkeit halber nochmal explizit da -- so ist aus der
# Datei allein ablesbar, welche Loss-Kombination der Lauf hatte.
#
# Gruppierung: `grouping=feature` clustert die Patches pro Bild nach der
# Kosinus-Aehnlichkeit der rohen Encoder-Features (cluster_patches_by_features),
# nicht nach Attention. Das sind min(n_clusters, P) = min(20, 196) = 20 Cluster
# pro Bild, aus loss.attention_grouping.n_clusters (S2AE-Default, zusammen mit
# spatial_coeff=0.02). Die Cluster werden pro Bild neu gezogen -- es sind also
# nicht dieselben 20 Regionen ueber den Datensatz hinweg.
#
# Der PR-Loss ist homogen vom Grad 0 und erzeugt daher keinen
# Shrinkage-Gradienten (siehe src/cbm_msae_lab/participation_ratio.py); sein
# Wertebereich ist [1, 20], also betragsmaessig vergleichbar mit dem
# Rekonstruktions-Loss. weight=0.1 setzt ihn bewusst als Nebenbedingung an.
#
# `train.eval.region_grouping` bleibt auf dem Default `tile`, damit
# val/region_pr gegen eine FESTE 2x2-Partition gemessen wird und mit
# Baseline-Laeufen vergleichbar bleibt. Soll stattdessen auf den Feature-Clustern
# gemessen werden: `train.eval.region_grouping=feature` ergaenzen UND vorher
# zusaetzlich `--split val --method feature` extrahieren.
#
# Alle drei Extraktionsschritte unten sind idempotent (vollstaendiger Cache ->
# skip, --overwrite erzwingt), dieser Job kann also gefahrlos parallel zu
# anderen laufen, die dieselben Caches benutzen.

#SBATCH --job-name=cub-pr-feature
#SBATCH --partition=gpu-vram-12gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "$(pwd)"

uv run scripts/extract_raw_features.py --split train
uv run scripts/extract_raw_features.py --split val
# Nur der train-Split: der PR-Loss wirkt ausschliesslich im Training, und die
# Region-Consistency-Metrik nutzt die Tile-Partition, die keinen Cache braucht.
uv run scripts/extract_attention_groups.py --split train --method feature

uv run scripts/train.py \
    train.cache.mode=cache_encoder \
    encoder=clip_dinoiser \
    loss.reconstruction.weight=1.0 \
    loss.auxk.weight=0.03125 \
    loss.participation_ratio.weight=0.1 \
    loss.participation_ratio.grouping=feature \
    train.checkpoint_dir=outputs/checkpoints/cub_pr_feature \
    train.wandb.run_name=cub_pr_feature
