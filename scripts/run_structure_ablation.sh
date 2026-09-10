#!/usr/bin/env bash
# Struktur-Ablation: Group-TopK und Participation-Ratio-Loss, einzeln und kombiniert.
#
# Die zentrale Hypothese ist NICHT, dass die Struktur-Terme FVU verbessern,
# sondern dass sie bei kontrollierter Rekonstruktionsqualitaet die raeumliche
# Spezialisierung und die Monosemantizitaet erhoehen. Deshalb schreibt jeder Arm
# denselben Metriksatz und die Arme unterscheiden sich NUR in den
# Struktur-Overrides -- Encoder, Dataset, SAE-Hyperparameter und Eval-Protokoll
# sind identisch:
#
#   val/fvu                          Rekonstruktionsqualitaet
#   val/l0                           mittlere Zahl aktiver Latents pro Patch
#   val/monosemanticity_score        MS, Pach et al. (ueber aktive Latents gemittelt)
#   val/tversky_ms                   TMS, Filus & Pokucinski
#   val/region_pr                    effektive Zahl Regionen pro Feature
#   val/region_dominant_share        Massenanteil der dominanten Region
#   val/activation_frequency_*       Feuerrate pro Feature (Mittel/Median/Tail)
#
# Region-Consistency wird auf den ATTENTION-CLUSTERN gemessen (echte Patch-
# Gruppen, keine 2x2-Kacheln) -- `train.eval.region_grouping=attention` wird
# unten auf allen vier Armen EXPLIZIT gesetzt, auch auf der Baseline, die
# selbst keinen Struktur-Loss hat: Default "auto" wuerde die Baseline sonst auf
# die feste Tile-Partition zurueckfallen lassen, waehrend die anderen drei auf
# den Clustern messen -- zwei verschiedene Partitionen, nicht vergleichbar.
# Mit derselben expliziten Partition auf allen vier Armen bleibt `val/region_pr`
# ueber die ganze Ablation hinweg dieselbe Grosse.
#
# Laeuft sequentiell in EINEM Prozess. Fuer eigenstaendige SLURM-Jobs stattdessen
# die vier Skripte scripts/sbatch/cub_struct_*_cached.sbatch einzeln einreichen;
# alle drei Extraktionsschritte unten sind idempotent, die vier Jobs koennen also
# gemeinsam abgesendet werden.

set -euo pipefail
cd "$(dirname "$0")/.."

uv run scripts/extract_raw_features.py --split train
uv run scripts/extract_raw_features.py --split val
# Beide Splits: Group-TopK/PR-Loss brauchen nur den train-Split (wirken
# ausschliesslich im Training), aber val/region_pr braucht den val-Split-Cache,
# weil alle vier Arme jetzt explizit auf Attention-Clustern messen.
uv run scripts/extract_attention_groups.py --split train --method attention
uv run scripts/extract_attention_groups.py --split val --method attention

# --- Arm 1: BatchTopK-Baseline -------------------------------------------
uv run scripts/train.py \
    train.cache.mode=cache_encoder \
    train.eval.region_grouping=attention \
    train.checkpoint_dir=outputs/checkpoints/cub_struct_baseline \
    train.wandb.run_name=cub_struct_baseline

# --- Arm 2: + Participation-Ratio-Loss -----------------------------------
uv run scripts/train.py \
    train.cache.mode=cache_encoder \
    loss.participation_ratio.weight=0.1 \
    loss.participation_ratio.grouping=attention \
    train.eval.region_grouping=attention \
    train.checkpoint_dir=outputs/checkpoints/cub_struct_pr \
    train.wandb.run_name=cub_struct_pr

# --- Arm 3: + Group-TopK -------------------------------------------------
uv run scripts/train.py \
    train.cache.mode=cache_encoder \
    loss.group_topk.enabled=true \
    loss.group_topk.grouping=attention \
    train.eval.region_grouping=attention \
    train.checkpoint_dir=outputs/checkpoints/cub_struct_grouptopk \
    train.wandb.run_name=cub_struct_grouptopk

# --- Arm 4: + beides -----------------------------------------------------
uv run scripts/train.py \
    train.cache.mode=cache_encoder \
    loss.group_topk.enabled=true \
    loss.group_topk.grouping=attention \
    loss.participation_ratio.weight=0.1 \
    loss.participation_ratio.grouping=attention \
    train.eval.region_grouping=attention \
    train.checkpoint_dir=outputs/checkpoints/cub_struct_both \
    train.wandb.run_name=cub_struct_both
