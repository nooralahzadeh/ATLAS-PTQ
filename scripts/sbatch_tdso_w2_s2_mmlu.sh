#!/bin/bash
# Gap-fill: firm T-DSO v2-mult 2-bit seed-2 MMLU subsets (extraction never ran for
# these; GSM8k s2 already covered by the rebuild). Sequential extraction on gpu0.
# Submit: sbatch scripts/sbatch_tdso_w2_s2_mmlu.sh
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=tdso_w2s2
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/tdso_w2_s2_mmlu_%j.out
set -euo pipefail
srun --environment=vscode-pytorch --gres=gpu:4 bash -lc '
set -euo pipefail
cd /capstor/scratch/cscs/fnoorala/ATLAS-PTQ
source llama31_venv/bin/activate
set -a; [ -f TACQ/.env ] && source TACQ/.env; set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export HF_HOME=/capstor/scratch/cscs/fnoorala/.cache
export PYTHONUNBUFFERED=1
SEED=2 WBITS=2 COMBINE=mult FORCE_RECOMPUTE=1 \
  TASKS="MMLU_STEM MMLU_humanities MMLU_social_sciences" \
  bash scripts/run_tdso_task_sequential_llama31.sh \
  2>&1 | tee -a tacq_data/results/downstream_firm_w2.log
'
