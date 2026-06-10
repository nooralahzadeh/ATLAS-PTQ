#!/bin/bash
# Task-conditioned quantization across 4 GPUs in parallel (one task per GPU).
# Default METHOD=tacq (each task is lightweight enough to run concurrently);
# T-DSO is run sequentially instead (see run_tdso_task_sequential_llama31.sh)
# because its fp32 gradient buffers OOM the node when run 4-up.
#
# RECONSTRUCTED after the 2026-06-10 scratch incident — VERIFY before production.
# Grounded on the surviving log markers:
#   [parallel] gpu=N method=run_<method>_task_conditioned_llama31.sh task=<TASK>
# Env: METHOD TASKS SEED WBITS FORCE_RECOMPUTE
set -euo pipefail
ROOT="${ROOT:-/capstor/scratch/cscs/fnoorala/ATLAS-PTQ}"
METHOD="${METHOD:-tacq}"
SEED="${SEED:-0}"
WBITS="${WBITS:-2}"
TASKS=(${TASKS:-GSM8k MMLU_STEM MMLU_humanities MMLU_social_sciences})
export FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"
log() { echo "[$(date +%H:%M:%S)] $*"; }

runner="run_${METHOD}_task_conditioned_llama31.sh"
log "[parallel] METHOD=$METHOD tasks=${TASKS[*]} w$WBITS seed=$SEED"

pids=(); gpu=0
for task in "${TASKS[@]}"; do
  log "[parallel] gpu=$gpu method=$runner task=$task"
  TASK="$task" SEED="$SEED" GPU="$gpu" WBITS="$WBITS" FORCE_RECOMPUTE="$FORCE_RECOMPUTE" \
    LOG="$ROOT/tacq_data/results/${METHOD}_task_conditioned_${WBITS}bit_gpu${gpu}.log" \
    bash "$ROOT/scripts/$runner" \
      >> "$ROOT/tacq_data/results/${METHOD}_task_conditioned_${WBITS}bit_gpu${gpu}.log" 2>&1 &
  pids+=($!)
  gpu=$(((gpu + 1) % 4))
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[[ "$fail" == "0" ]] || log "WARN: one or more parallel workers failed"
exit $fail
