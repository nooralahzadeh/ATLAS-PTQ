#!/bin/bash
# Gate 1b (a0125 normal): GSM8K 8-shot CoT eval of the boost_rank engine built
# by fixval2 on a135 debug. Single GPU, ~2.5h. This is the only a0125 spend
# before the Gate-1 go/no-go decision.
# Submit: sbatch --dependency=afterok:<fixval2job> scripts/sbatch_a0125_gate1b_gsm8k_eval.sh
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=gate1b_gsm8k
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/gate1b_gsm8k_%j.out
set -euo pipefail
srun --environment=vscode-pytorch --gres=gpu:1 bash -lc '
set -euo pipefail
cd /capstor/scratch/cscs/fnoorala/ATLAS-PTQ
source llama31_venv/bin/activate
set -a; [ -f TACQ/.env ] && source TACQ/.env; set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export HF_HOME=/capstor/scratch/cscs/fnoorala/.cache
export PYTHONUNBUFFERED=1
export CKPT_DIR=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data
export RESULTS_DIR=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/results
source scripts/pretrain_eval_helpers.sh

ENGINE=Meta-Llama-3.1-8B-Instruct_fixv2_boostrank_free_GSM8k_2bit_quantized_model
[[ -f "$CKPT_DIR/$ENGINE.pt" ]] || { echo "FATAL: engine missing: $CKPT_DIR/$ENGINE.pt"; exit 1; }

(cd TACQ && TASK=GSM8k ENGINE="$ENGINE" SEED=0 run_pretrain_eval_profile same)
echo "=== gate1b complete. Compare against firm TaCQ GSM8k s0 = 28.3 (mean 29.6) ==="
'
