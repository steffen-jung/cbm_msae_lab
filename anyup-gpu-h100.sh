#!/bin/bash
#SBATCH -J cbm_anyup
#SBATCH -p gpu-h100
#SBATCH -t 23:59:00
#SBATCH -o /BS/TrackOpt/work/cbm_msae_lab/shell/logs/anyup.%A.%a.%x.log
#SBATCH -a 0-0
#SBATCH --gres=gpu:h100:1

# Train the Matryoshka BatchTopK SAE with AnyUp feature-stage upsampling
# (64x64 token grid before the SAE reconstructs). Must run live --
# cache_encoder cannot supply the guidance image AnyUp needs.
#
# Submit from the repo root:
#   sbatch anyup-gpu-h100.sh
#
# Re-submit if needed; early stopping usually finishes sooner.

set -euo pipefail

export HF_HOME=/BS/TrackOpt/work/HF_HOME
export TORCH_HOME=/BS/TrackOpt/work/TORCH_HOME
export PYTHONUTF8=1

REPO=/BS/TrackOpt/work/cbm_msae_lab
ENV=/BS/TrackOpt/work/env-cbm-msae
DATA=/scratch/inf0/user/sjung/CUB200
LOG_DIR="${REPO}/shell/logs"

mkdir -p "${LOG_DIR}" "${HF_HOME}" "${TORCH_HOME}"

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

echo "=== cbm_msae_lab AnyUp MSAE training ==="
echo "host:     $(hostname)"
echo "date:     $(date -Is)"
echo "gpu:      ${CUDA_VISIBLE_DEVICES:-unset}"
echo "job_id:   ${SLURM_JOB_ID:-local}"
echo "repo:     ${REPO}"
echo "data:     ${DATA}"
echo "log:      ${LOG_DIR}/anyup.${SLURM_ARRAY_JOB_ID:-0}.${SLURM_ARRAY_TASK_ID:-0}.${SLURM_JOB_NAME:-cbm_anyup}.log"
echo

eval "$(conda shell.bash hook)"
conda activate "${ENV}"

cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 -c "import multiprocessing as mp; print('cpu_count', mp.cpu_count())"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

echo
echo "--- starting AnyUp feature-stage training (64x64) ---"
python3 scripts/train.py \
    upsampler=anyup \
    upsampler.target_resolution=64 \
    upsampler.stage=features \
    dataset.root="${DATA}" \
    train.cache.mode=live \
    train.wandb.mode=disabled \
    train.checkpoint_dir=outputs/checkpoints/cub_anyup_64 \
    train.wandb.run_name=cub_anyup_64

echo
echo "done: $(date -Is)"
echo "Tip: re-submit this script if the job was cut off by walltime."
