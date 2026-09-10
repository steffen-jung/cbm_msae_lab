#!/bin/bash
#SBATCH -J cbm_anyup_bs
#SBATCH -p gpu-a100
#SBATCH -t 23:59:00
#SBATCH -o /BS/TrackOpt/work/cbm_msae_lab/shell/logs/anyup_bs.%A.%a.%x.log
#SBATCH -a 0-3%1
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=gpu22-a100-01

# Array sweep: AnyUp feature-stage MSAE training at batch sizes 64 / 32 / 16 / 4
# (large -> small). `%1` on the array range runs only one task at a time.
# Task index -> batch size: 0->64, 1->32, 2->16, 3->4.
#
# Submit from the repo root:
#   sbatch anyup-bs-array-gpu22.sh
#
# The gpu22 / A100 partitions cap walltime at 24 hours. Re-submit if needed;
# early stopping usually finishes sooner. Each array task writes its own
# checkpoint dir and wandb run name.

set -euo pipefail

export HF_HOME=/BS/TrackOpt/work/HF_HOME
export TORCH_HOME=/BS/TrackOpt/work/TORCH_HOME
export PYTHONUTF8=1

REPO=/BS/TrackOpt/work/cbm_msae_lab
ENV=/BS/TrackOpt/work/env-cbm-msae
DATA=/BS/databases01/CUB_200_2011/CUB_200_2011
LOG_DIR="${REPO}/shell/logs"

BATCH_SIZES=(64 32 16 4)
BATCH_SIZE="${BATCH_SIZES[${SLURM_ARRAY_TASK_ID}]}"
RUN_NAME="cub_anyup_64_bs${BATCH_SIZE}"

mkdir -p "${LOG_DIR}" "${HF_HOME}" "${TORCH_HOME}"

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

echo "=== cbm_msae_lab AnyUp MSAE training (batch-size sweep) ==="
echo "host:       $(hostname)"
echo "date:       $(date -Is)"
echo "gpu:        ${CUDA_VISIBLE_DEVICES:-unset}"
echo "job_id:     ${SLURM_JOB_ID:-local}"
echo "array_task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "batch_size: ${BATCH_SIZE}"
echo "repo:       ${REPO}"
echo "data:       ${DATA}"
echo "run_name:   ${RUN_NAME}"
echo "log:        ${LOG_DIR}/anyup_bs.${SLURM_ARRAY_JOB_ID:-0}.${SLURM_ARRAY_TASK_ID:-0}.${SLURM_JOB_NAME:-cbm_anyup_bs}.log"
echo

eval "$(conda shell.bash hook)"
conda activate "${ENV}"

cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 -c "import multiprocessing as mp; print('cpu_count', mp.cpu_count())"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

echo
echo "--- starting AnyUp feature-stage training (64x64, batch_size=${BATCH_SIZE}) ---"
python3 scripts/train.py \
    upsampler=anyup \
    upsampler.target_resolution=64 \
    upsampler.stage=features \
    dataset.root="${DATA}" \
    train.cache.mode=live \
    train.batch_size="${BATCH_SIZE}" \
    train.wandb.mode=disabled \
    train.checkpoint_dir="outputs/checkpoints/${RUN_NAME}" \
    train.wandb.run_name="${RUN_NAME}"

echo
echo "done: $(date -Is) (batch_size=${BATCH_SIZE})"
echo "Tip: partition walltime is 24h — re-submit this script if the job was cut off."
