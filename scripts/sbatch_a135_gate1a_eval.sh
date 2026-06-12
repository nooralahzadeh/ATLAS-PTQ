#!/bin/bash
# Gate 1a/window B (a135 debug): evaluate the 3 MMLU boost_rank engines built
# by window A. 3 evals in parallel on GPUs 0-2 (~0.5-1h each).
# Submit: sbatch --dependency=afterok:<buildjob> scripts/sbatch_a135_gate1a_eval.sh
#SBATCH --account=a135
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=gate1a_eval
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/gate1a_eval_%j.out
set -euo pipefail
srun --environment=vscode-pytorch --gres=gpu:4 bash -lc '
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

eval_one() {
  local task="$1" gpu="$2"
  local engine="Meta-Llama-3.1-8B-Instruct_fixv2_boostrank_free_${task}_2bit_quantized_model"
  [[ -f "$CKPT_DIR/$engine.pt" ]] || { echo "FATAL [$task]: engine missing"; exit 1; }
  (cd TACQ && CUDA_VISIBLE_DEVICES=$gpu TASK="$task" ENGINE="$engine" SEED=0 \
    run_pretrain_eval_profile same)
}

pids=(); gpu=0
for task in MMLU_STEM MMLU_humanities MMLU_social_sciences; do
  eval_one "$task" "$gpu" & pids+=($!); gpu=$((gpu + 1))
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit $fail
'
