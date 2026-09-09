#!/usr/bin/env bash
# Selbst einreichen mit: sbatch scripts/sbatch/cub_s2ae_attention_ste_cached.sbatch
#
# Wie cub_s2ae_attention_cached.sbatch, aber nach dem Fix in
# losses/group_sparsity.py / losses/exclusivity.py: group_sparsity/exclusivity
# binarisieren ihr Gruppenprofil jetzt per Straight-Through-Estimator
# (losses/ste.py, S2AE-Paper Sec. 4.2), statt die L1/L2-Strafe direkt auf die
# rohen Aktivierungsmagnituden anzuwenden. Der ursprüngliche
# cub_s2ae_attention-Run (ohne STE) zeigte einen kollabierten
# BatchTopK-Inferenz-Threshold (~5e-6 statt ~0.35-0.39) und eine wachsende
# Zahl toter Features (2806/8192 am Ende) -- eigener checkpoint_dir/wandb-Run
# (Suffix "_ste"), damit der alte Checkpoint zum Vergleich erhalten bleibt.
#
# Baut einmalig den Attention-Gruppen-Cache neu (Cache-Key haengt nicht von
# der Loss-Formel ab, also identisch zum alten Run -- der Schritt ist hier
# trotzdem drin, um das Script eigenstaendig lauffaehig zu halten); nutzt
# denselben Roh-Encoder-Cache wie die anderen Runs (hier nicht nochmal
# extrahieren -- falls dieser Job als erster laeuft, vorher von Hand:
# `uv run scripts/extract_raw_features.py --split train`
# und `--split val`).

#SBATCH --job-name=cub-s2ae-attn-ste
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
    train.checkpoint_dir=outputs/checkpoints/cub_s2ae_attention_ste \
    train.wandb.run_name=cub_s2ae_attention_ste
