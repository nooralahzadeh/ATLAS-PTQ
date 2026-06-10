#!/bin/bash
# One downstream seed: build pairs -> TaCQ (parallel 4-GPU) -> T-DSO v2 mult (seq gpu0).
#
# RECONSTRUCTED after the 2026-06-10 scratch incident — VERIFY before production.
# Grounded on the surviving downstream_firm_w2.log step markers:
#   Step 1/3: contrastive pairs (seed, mmlu=test75 gsm8k=tacq)
#   Step 2/3: TaCQ task-conditioned (parallel, 4 GPU)
#   Step 3/3: T-DSO v2 mult task-conditioned (sequential, gpu0; avoids node OOM)
# Env: SEED WBITS FORCE_RECOMPUTE TACQ_ONLY EXTRACT_BS MMLU_MAX_LEN MMLU_CALIB_MODE GSM8K_PAIR_FORMAT
set -euo pipefail
ROOT="${ROOT:-/capstor/scratch/cscs/fnoorala/ATLAS-PTQ}"
cd "$ROOT"
SEED="${SEED:-0}"
WBITS="${WBITS:-2}"
export FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-1}"
TASKS=(${TASKS:-GSM8k MMLU_STEM MMLU_humanities MMLU_social_sciences})
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Downstream seed=$SEED w$WBITS FORCE_RECOMPUTE=$FORCE_RECOMPUTE ==="

log "Step 1/3: contrastive pairs (seed=$SEED, mmlu=${MMLU_CALIB_MODE:-test75} gsm8k=${GSM8K_PAIR_FORMAT:-tacq})"
# NOTE: scripts/data_prep_contrastive.py is recovery/recon/ blueprint — clean it into
# the tree before forcing a rebuild; pairs from the original run survived in data/contrastive/.
python scripts/data_prep_contrastive.py \
  --tasks "${TASKS[@]}" --seed "$SEED" \
  --mmlu-calib-mode "${MMLU_CALIB_MODE:-test75}" \
  --gsm8k-format "${GSM8K_PAIR_FORMAT:-tacq}" \
  --n-calib 128 || log "WARN: pair build skipped/failed (pairs may already exist)"

log "Step 2/3: TaCQ task-conditioned (parallel, 4 GPU)"
METHOD=tacq SEED="$SEED" WBITS="$WBITS" TASKS="${TASKS[*]}" FORCE_RECOMPUTE="$FORCE_RECOMPUTE" \
  bash "$ROOT/scripts/run_task_conditioned_parallel_llama31.sh"

if [[ "${TACQ_ONLY:-0}" != "1" ]]; then
  log "Step 3/3: T-DSO v2 mult task-conditioned (sequential, gpu=0; avoids node OOM)"
  SKIP_SPIDER_IF_EXISTS=1 COMBINE=mult SEED="$SEED" WBITS="$WBITS" TASKS="${TASKS[*]}" \
    EXTRACT_BS="${EXTRACT_BS:-2}" MMLU_MAX_LEN="${MMLU_MAX_LEN:-2048}" \
    FORCE_RECOMPUTE="$FORCE_RECOMPUTE" \
    bash "$ROOT/scripts/run_tdso_task_sequential_llama31.sh"
else
  log "SKIP T-DSO (TACQ_ONLY=1)"
fi

log "=== Downstream seed=$SEED w$WBITS done ==="
