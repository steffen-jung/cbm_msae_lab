#!/bin/bash
#SBATCH -J swb_full
#SBATCH -p gpu22
#SBATCH -t 23:59:00
#SBATCH -o /BS/TrackOpt/work/SpectralWaterBench/shell/logs/full.%A.%a.%x.log
#SBATCH -a 0-0
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=gpu22-a100-01

# SpectralWaterBench full mode: 9 methods × 500 images × full spectral attack grid.
#
# Config: configs/full.yaml (spectral_profile: full, image_save_policy: generated_only)
# Saves watermarked/clean base images + base_image_cache; does NOT save attacked PNGs.
#
# The gpu22 partition caps walltime at 24 hours. Re-submit this script as needed;
# completed conditions are skipped automatically via --resume.

set -euo pipefail

export HF_HOME=/BS/TrackOpt/work/HF_HOME
export PYTHONUTF8=1

REPO=/BS/TrackOpt/work/SpectralWaterBench
LOG_DIR="${REPO}/shell/logs"

mkdir -p "${LOG_DIR}"

trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

echo "=== SpectralWaterBench full benchmark ==="
echo "host:     $(hostname)"
echo "date:     $(date -Is)"
echo "gpu:      ${CUDA_VISIBLE_DEVICES:-unset}"
echo "job_id:   ${SLURM_JOB_ID:-local}"
echo "log:      ${LOG_DIR}/full.${SLURM_ARRAY_JOB_ID:-0}.${SLURM_ARRAY_TASK_ID:-0}.${SLURM_JOB_NAME:-swb_full}.log"
echo

eval "$(conda shell.bash hook)"
conda activate /BS/TrackOpt/work/env-spectralwaterbench

cd "${REPO}"

python3 -c "import multiprocessing as mp; print('cpu_count', mp.cpu_count())"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

echo
echo "--- starting / resuming full run ---"
python benchmark.py --mode full --resume

echo
echo "done: $(date -Is)"
echo "Tip: partition walltime is 24h — re-submit this script to continue."
echo "After completion, run: python benchmark.py --analyze-existing outputs/<run_id>_full"
