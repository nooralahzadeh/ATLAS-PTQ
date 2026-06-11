#!/bin/bash
# Qwen2.5-7B-Instruct: ATLAS-N (mult_free) vs TaCQ (ce) full grid.
# 4 tasks x 3 seeds, 2-bit. Dictionary-free only — no Qwen transcoders exist.
# Submit ONLY after qwen_probe succeeds end-to-end.
# Usage: SEED=<0|1|2> sbatch scripts/sbatch_qwen_grid.sh
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=qwen_grid
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/qwen_grid_s%x_%j.out
set -euo pipefail
srun --environment=vscode-pytorch --gres=gpu:4 bash -lc '
set -euo pipefail
cd /capstor/scratch/cscs/fnoorala/ATLAS-PTQ
source llama31_venv/bin/activate
set -a; [ -f TACQ/.env ] && source TACQ/.env; set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export HF_HOME=/capstor/scratch/cscs/fnoorala/.cache
export PYTHONUNBUFFERED=1
MODEL_LOAD="Qwen/Qwen2.5-7B-Instruct" MODEL_BASE="Qwen2.5-7B-Instruct" \
  ARMS="ce mult_free" TASKS="GSM8k MMLU_STEM MMLU_humanities MMLU_social_sciences" \
  WBITS=2 SEED="${SEED:-0}" FORCE_RECOMPUTE=1 EXTRACT_BS=1 \
  bash scripts/run_saliency_source_ablation.sh \
  2>&1 | tee -a tacq_data/results/qwen_secondmodel_w2.log
'
