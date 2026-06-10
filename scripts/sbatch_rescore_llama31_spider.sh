#!/bin/bash
# Rescore Spider exec accuracy for Llama-3.1 TaCQ job 2468963 (no GPU inference).
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=rescore_llama31
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/rescore_llama31_%j.out

set -euo pipefail

srun --environment=vscode-pytorch bash -lc '
set -euo pipefail
cd /capstor/scratch/cscs/fnoorala/ATLAS-PTQ/TACQ
source ../llama31_venv/bin/activate
pip install -q sqlparse==0.5.3

ROOT=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/results/Spider
UNQ="${ROOT}/unquantized_Meta-Llama-3.1-8B-Instruct"
Q2="${ROOT}/Meta-Llama-3.1-8B-Instruct+0+Spider+sample_abs_weight_prod_contrastive_sm16bit+2bit+implementation_test+gptq_on_Spider+q2+top_p_sparse+.0035+quantized_model"
Q3="${ROOT}/Meta-Llama-3.1-8B-Instruct+0+Spider+sample_abs_weight_prod_contrastive_sm16bit+3bit+implementation_test+gptq_on_Spider+q3+top_p_sparse+.0035+quantized_model"

rescore_one() {
  local label="$1" pred="$2" log="$3"
  echo "=== Rescore ${label} ==="
  python scripts/rescore_spider_exec.py "$pred" --log "$log"
}

rescore_one unquantized "${UNQ}/predictions.txt" "${UNQ}/eval_exec.log"
rescore_one q2 "${Q2}/predictions.txt" "${Q2}/eval_exec.log"
rescore_one q3 "${Q3}/predictions.txt" "${Q3}/eval_exec.log"

python scripts/collect_results_llama31.py
'
