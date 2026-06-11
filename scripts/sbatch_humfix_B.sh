#!/bin/bash
# Humanities truncation-bug fix, part B (gated on all in-flight writers):
#  - ablation w3 s0: all heavy arms; w2/w3 s1,s2: dict-free arms
#  - firm ATLAS-T (tdsoV2mult) humanities w3 seeds 0,1,2
# Submit: sbatch --dependency=afterany:<jobids> scripts/sbatch_humfix_B.sh
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=humfix_B
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/humfix_B_%j.out
set -euo pipefail
srun --environment=vscode-pytorch --gres=gpu:4 bash -lc '
set -euo pipefail
cd /capstor/scratch/cscs/fnoorala/ATLAS-PTQ
source llama31_venv/bin/activate
set -a; [ -f TACQ/.env ] && source TACQ/.env; set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export HF_HOME=/capstor/scratch/cscs/fnoorala/.cache
export PYTHONUNBUFFERED=1

ARMS="ce align mult align_free mult_free" TASKS="MMLU_humanities" \
  WBITS=3 SEED=0 FORCE_RECOMPUTE=1 EXTRACT_BS=1 \
  bash scripts/run_saliency_source_ablation.sh \
  2>&1 | tee -a tacq_data/results/saliency_source_ablation_w3.log

for combo in "2 1" "2 2" "3 1" "3 2"; do
  set -- $combo
  ARMS="align_free mult_free" TASKS="MMLU_humanities" \
    WBITS=$1 SEED=$2 FORCE_RECOMPUTE=1 EXTRACT_BS=1 \
    bash scripts/run_saliency_source_ablation.sh \
    2>&1 | tee -a tacq_data/results/saliency_source_ablation_w$1.log
done

for SEED in 0 1 2; do
  SEED=$SEED WBITS=3 COMBINE=mult FORCE_RECOMPUTE=1 EXTRACT_BS=1 \
    TASKS="MMLU_humanities" \
    bash scripts/run_tdso_task_sequential_llama31.sh \
    2>&1 | tee -a tacq_data/results/downstream_firm_w3.log
done
'
