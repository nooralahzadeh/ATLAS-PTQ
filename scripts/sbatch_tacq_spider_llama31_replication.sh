#!/bin/bash
# Full TaCQ Spider replication for Meta-Llama-3.1-8B-Instruct on 4× H200.
#
# Uses llama31_venv (transformers 4.45+ for Llama-3.1). Does NOT touch tacq_venv,
# Llama-3-8B artifacts, tacq_msg files, or llama31_corrupt_2bit.pt.
#
# Submit with:  sbatch scripts/sbatch_tacq_spider_llama31_replication.sh
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=tacq_llama31
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/repro_llama31_%j.out

set -euo pipefail

srun --environment=vscode-pytorch --gres=gpu:4 bash -lc '
set -euo pipefail
cd /capstor/scratch/cscs/fnoorala/ATLAS-PTQ
bash scripts/setup_llama31_venv.sh

cd TACQ
source ../llama31_venv/bin/activate
set -a; source .env; set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export HF_HOME=/capstor/scratch/cscs/fnoorala/.cache
export PYTHONUNBUFFERED=1
export NUM_GPUS=4
export TACQ_VENV=../llama31_venv

python - <<'"'"'PY'"'"'
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
print("torch perf flags set; visible GPUs:", torch.cuda.device_count())
PY

UNQUANT_LOG="/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/results/Spider/unquantized_Meta-Llama-3.1-8B-Instruct/eval_exec.log"
if [ ! -f "$UNQUANT_LOG" ]; then
  echo "=== Step 1: Unquantized Spider baseline (Llama-3.1-8B, GPU 0) ==="
  CUDA_VISIBLE_DEVICES=0 MODEL_NAME=Meta-Llama-3.1-8B-Instruct TACQ_VENV=../llama31_venv \
    bash scripts/run_unquantized_baseline.sh
else
  echo "=== Step 1: Unquantized baseline already present, skipping ==="
fi

echo "=== Step 2: Full TaCQ q2 + q3 on 4 GPUs (contrastive, ratio .0035, top_p_sparse) ==="
TESTING=0 KEEP_CORRUPT=1 TACQ_VENV=../llama31_venv bash scripts/examples/evaluate_llama31_8b_spider.sh

echo "=== Step 3: Collect Llama-3.1 results ==="
python scripts/collect_results_llama31.py
'
