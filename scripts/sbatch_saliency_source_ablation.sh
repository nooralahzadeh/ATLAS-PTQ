#!/bin/bash
# Saliency-source ablation: transcoder arms (ce/align/mult) + non-circuit baselines
# (weight/magnitude/random). Submit: sbatch scripts/sbatch_saliency_source_ablation.sh
# RECONSTRUCTED after the 2026-06-10 scratch incident.
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=abl_src
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/saliency_source_ablation_%j.out
set -euo pipefail
srun --environment=vscode-pytorch --gres=gpu:4 bash -lc '
set -euo pipefail
cd /capstor/scratch/cscs/fnoorala/ATLAS-PTQ
source llama31_venv/bin/activate
set -a; [ -f TACQ/.env ] && source TACQ/.env; set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export HF_HOME=/capstor/scratch/cscs/fnoorala/.cache
export PYTHONUNBUFFERED=1
ARMS="${ARMS:-ce align mult weight magnitude random}" \
TASKS="${TASKS:-GSM8k MMLU_STEM}" WBITS="${WBITS:-2}" SEED="${SEED:-0}" \
  bash scripts/run_saliency_source_ablation.sh \
  2>&1 | tee -a tacq_data/results/saliency_source_ablation_w${WBITS:-2}.log
'
