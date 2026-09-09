#!/usr/bin/env bash
# CUB-Vergleich: CFM vanilla vs. Spatial-Loss vs. S2AE-Losses.
# Fixer Backbone (CLIP-DINOiser, conf/encoder/clip_dinoiser.yaml). Die SAE sieht
# dessen 512-d Patch-Features direkt -- keine Zwischenschicht, genau wie CFM.
# Kein Upsampling (Default upsampler=none -> native 14x14 Low-Res-Grid),
# ReLU (SAE-Default sae.activation=relu). Jeder Lauf bekommt einen
# eigenen checkpoint_dir und wandb run_name, da beide sonst geteilt/überschrieben
# würden (siehe conf/train/default.yaml). Jeder Lauf schreibt automatisch
# checkpoint_dir/best.pt (nach val/fvu) und stösst danach automatisch
# scripts/evaluate_task_accuracy.py darauf an (siehe scripts/train.py).
#
# Läuft sequentiell in EINEM Prozess -- für die eigenständigen SLURM-Jobs
# stattdessen die passenden Skripte in scripts/sbatch/ einzeln einreichen
# (cub_cfm_vanilla_cached.sbatch, cub_spatial_cached.sbatch,
# cub_s2ae_attention_cached.sbatch, cub_s2ae_feature_cached.sbatch).
# Dieses Skript hier baut den Roh-Encoder-Cache selbst vor Run 1 und nutzt
# ihn für die restlichen Runs mit (train.cache.mode=cache_encoder).

set -euo pipefail
cd "$(dirname "$0")/.."

uv run scripts/extract_raw_features.py --split train
uv run scripts/extract_raw_features.py --split val

# --- Run 1: CFM vanilla (Standard-Rekonstruktion + auxk) -----------------
uv run scripts/train.py \
    train.cache.mode=cache_encoder \
    train.checkpoint_dir=outputs/checkpoints/cub_cfm_vanilla \
    train.wandb.run_name=cub_cfm_vanilla

# --- Run 2: Spatial-Loss statt Standard-Rekonstruktion -------------------
# reconstruction.weight=0.0, da scale_spatial's s=1-Term bereits die volle
# Patch-Rekonstruktion enthält (siehe scale_spatial.py-Docstring) -- Ersatz,
# keine Ergänzung.
uv run scripts/train.py \
    train.cache.mode=cache_encoder \
    loss.reconstruction.weight=0.0 \
    loss.scale_spatial.weight=1.0 \
    train.checkpoint_dir=outputs/checkpoints/cub_spatial \
    train.wandb.run_name=cub_spatial

# --- Run 3a: S2AE group-sparsity/exclusivity, Attention-Clustering -------
# (additiv zur Standard-Rekonstruktion). Gruppen-Cache einmalig vorher bauen
# (wird von allen künftigen Attention-S2AE-Läufen mit demselben Encoder +
# Clustering-Hyperparametern wiederverwendet, nicht pro Lauf nötig).
uv run scripts/extract_attention_groups.py --split train --method attention

uv run scripts/train.py \
    train.cache.mode=cache_encoder \
    loss.group_sparsity.grouping=attention loss.group_sparsity.weight=0.1 \
    loss.exclusivity.grouping=attention loss.exclusivity.weight=0.1 \
    train.checkpoint_dir=outputs/checkpoints/cub_s2ae_attention \
    train.wandb.run_name=cub_s2ae_attention

# --- Run 3b: dieselbe Loss-Kombination, aber Feature-Clustering statt -----
# Attention (grouping=feature) -- eigener Gruppen-Cache, eigener Cache-Key.
uv run scripts/extract_attention_groups.py --split train --method feature

uv run scripts/train.py \
    train.cache.mode=cache_encoder \
    loss.group_sparsity.grouping=feature loss.group_sparsity.weight=0.1 \
    loss.exclusivity.grouping=feature loss.exclusivity.weight=0.1 \
    train.checkpoint_dir=outputs/checkpoints/cub_s2ae_feature \
    train.wandb.run_name=cub_s2ae_feature
