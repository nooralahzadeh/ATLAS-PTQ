#!/bin/bash
# Downstream firm numbers (2-bit), resumable. Submit: sbatch scripts/sbatch_downstream_firm_resume.sh
# RECONSTRUCTED after the 2026-06-10 scratch incident (convention from
# sbatch_tacq_spider_llama31_replication.sh; --time capped to the normal 12h limit).
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=ds_firm_w2
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/downstream_firm_w2_%j.out
set -euo pipefail
srun --environment=vscode-pytorch --gres=gpu:4 bash -lc '
set -euo pipefail
cd /capstor/scratch/cscs/fnoorala/ATLAS-PTQ
source llama31_venv/bin/activate
set -a; [ -f TACQ/.env ] && source TACQ/.env; set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export HF_HOME=/capstor/scratch/cscs/fnoorala/.cache
export PYTHONUNBUFFERED=1
python - <<PY
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
print("torch perf flags set; visible GPUs:", torch.cuda.device_count())
PY
SEEDS="${SEEDS:-0 1 2}" WBITS=2 FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-1}" \
  bash scripts/run_downstream_firm_resume.sh \
  2>&1 | tee -a tacq_data/results/downstream_firm_w2.log
'
