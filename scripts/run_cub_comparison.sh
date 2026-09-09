#!/usr/bin/env bash
# CUB-Vergleich: CFM vanilla vs. Spatial-Loss vs. S2AE-Losses.
# Fixer Backbone (CLIP-DINOiser, conf/encoder/clip_dinoiser.yaml), 1x1-Projektion
# (projection=k1), kein Upsampling (Default upsampler=none -> native 14x14
# Low-Res-Grid), ReLU (SAE-Default sae.activation=relu). Jeder Lauf bekommt einen
# eigenen checkpoint_dir und wandb run_name, da beide sonst geteilt/überschrieben
# würden (siehe conf/train/default.yaml).
#
# Nicht als sbatch-Job vorbereitet -- einzeln oder in eigenen sbatch-Skripten
# selbst einreichen. Läuft jedes Kommando sequentiell; zum Parallelisieren auf
# mehrere GPUs am besten einzeln in separaten Terminals/Jobs starten.

set -euo pipefail
cd "$(dirname "$0")/.."

# --- Run 1: CFM vanilla (Standard-Rekonstruktion + auxk) -----------------
uv run scripts/train.py \
    projection=k1 \
    train.checkpoint_dir=outputs/checkpoints/cub_k1_cfm_vanilla \
    train.wandb.run_name=cub_k1_cfm_vanilla

# --- Run 2: Spatial-Loss statt Standard-Rekonstruktion -------------------
# reconstruction.weight=0.0, da scale_spatial's s=1-Term bereits die volle
# Patch-Rekonstruktion enthält (siehe scale_spatial.py-Docstring) -- Ersatz,
# keine Ergänzung.
uv run scripts/train.py \
    projection=k1 \
    loss.reconstruction.weight=0.0 \
    loss.scale_spatial.weight=1.0 \
    train.checkpoint_dir=outputs/checkpoints/cub_k1_spatial \
    train.wandb.run_name=cub_k1_spatial

# --- Run 3: S2AE group-sparsity/exclusivity (additiv zur Rekonstruktion) -
# Einmalig vorher: Attention-Gruppen-Cache bauen (wird von allen künftigen
# S2AE-Läufen mit demselben Encoder wiederverwendet, nicht pro Lauf nötig).
uv run scripts/extract_attention_groups.py --split train

uv run scripts/train.py \
    projection=k1 \
    loss.group_sparsity.grouping=attention loss.group_sparsity.weight=0.1 \
    loss.exclusivity.grouping=attention loss.exclusivity.weight=0.1 \
    train.checkpoint_dir=outputs/checkpoints/cub_k1_s2ae \
    train.wandb.run_name=cub_k1_s2ae
