#!/bin/bash
# T-DSO v2 task-conditioned, SEQUENTIAL on a single GPU (avoids the multi-process
# fp32-grad CPU OOM seen with the parallel launcher). Loops tasks on GPU 0.
#
# RECONSTRUCTED after the 2026-06-10 scratch incident — VERIFY before production.
# Env: TASKS SEED WBITS COMBINE FORCE_RECOMPUTE EXTRACT_BS MMLU_MAX_LEN GSM8K_PAIR_FORMAT
set -euo pipefail
ROOT="${ROOT:-/capstor/scratch/cscs/fnoorala/ATLAS-PTQ}"
SEED="${SEED:-0}"
WBITS="${WBITS:-2}"
COMBINE="${COMBINE:-mult}"
GPU="${GPU:-0}"
TASKS=(${TASKS:-GSM8k MMLU_STEM MMLU_humanities MMLU_social_sciences})
export FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-1}"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "T-DSO v2 $COMBINE sequential (gpu=$GPU) w$WBITS seed=$SEED tasks=${TASKS[*]}"
for task in "${TASKS[@]}"; do
  if [[ "$task" == "Spider" && "${SKIP_SPIDER_IF_EXISTS:-0}" == "1" ]]; then
    log "skip Spider (SKIP_SPIDER_IF_EXISTS=1)"; continue
  fi
  TASK="$task" SEED="$SEED" GPU="$GPU" WBITS="$WBITS" COMBINE="$COMBINE" \
    EXTRACT_BS="${EXTRACT_BS:-2}" MMLU_MAX_LEN="${MMLU_MAX_LEN:-4096}" \
    FORCE_RECOMPUTE="$FORCE_RECOMPUTE" \
    bash "$ROOT/scripts/run_tdso_v2_task_conditioned_llama31.sh"
done
