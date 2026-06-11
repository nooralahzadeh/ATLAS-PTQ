#!/bin/bash
# SECOND-MODEL PROBE: ATLAS-N (mult_free, dict-free) vs TaCQ (ce) on
# Qwen2.5-7B-Instruct, GSM8k 2-bit, seed 0. Validates the full chain
# (dict-free extraction -> GPTQ -> eval) on a non-Llama architecture for which
# NO transcoder exists -- the core selling point of the dictionary-free method.
# Scoped to one task / 2h so we catch arch breakage cheap before the full grid.
# Submit: sbatch scripts/sbatch_qwen_probe.sh
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=qwen_probe
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/qwen_probe_%j.out
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
  ARMS="ce mult_free" TASKS="GSM8k" WBITS=2 SEED=0 FORCE_RECOMPUTE=1 EXTRACT_BS=1 \
  bash scripts/run_saliency_source_ablation.sh \
  2>&1 | tee -a tacq_data/results/qwen_secondmodel_w2.log
'
