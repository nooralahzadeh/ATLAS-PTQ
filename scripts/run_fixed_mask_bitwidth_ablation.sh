#!/bin/bash
# Fixed-mask bitwidth ablation: hold a method's 0.35% FP16 outlier mask FIXED and
# quantize the rest at --wbits, to separate "which weights are protected" (the
# mask/circuit) from "the bitwidth of everything else". Compares tacq vs tdso masks.
#
# RECONSTRUCTED after the 2026-06-10 scratch incident — LOWEST-confidence runner
# (session-authored, no bytecode). VERIFY the ablation semantics against the
# fixed_mask_*.log results before trusting numbers.
# Env: TASK METHOD SEED GPU WBITS FORCE_RECOMPUTE RESUME
set -euo pipefail
ROOT="${ROOT:-/capstor/scratch/cscs/fnoorala/ATLAS-PTQ}"
TASK="${TASK:?TASK required}"
METHOD="${METHOD:-tacq}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
WBITS="${WBITS:-2}"
MODEL_LOAD="${MODEL_LOAD:-unsloth/Meta-Llama-3.1-8B-Instruct}"
MODEL_BASE="${MODEL_BASE:-Meta-Llama-3.1-8B-Instruct}"
CKPT_DIR="${CKPT_DIR:-$ROOT/tacq_data}"
export RESULTS_DIR="${RESULTS_DIR:-$ROOT/tacq_data/results}"
export CKPT_DIR
source "$ROOT/scripts/pretrain_eval_helpers.sh"
log() { echo "[$(date +%H:%M:%S)] $*"; }

seed_suffix=""; [[ "$SEED" != "0" ]] && seed_suffix="_s${SEED}"
ENGINE="${MODEL_BASE}_fixedmask_${METHOD}_${TASK}_${WBITS}bit${seed_suffix}_quantized_model"

# Locate the method's flat important_mask produced by the task-conditioned runners.
case "$METHOD" in
  tacq)
    src_run="${MODEL_BASE}+${SEED}+${TASK}+sample_abs_weight_prod_contrastive+${WBITS}bit+conditioned"
    MASK_FLAT="$CKPT_DIR/importances/$src_run/important_mask_q${WBITS}+top_p_sparse+.0035.pt" ;;
  tdso)
    src_engine="${MODEL_BASE}_tdsoV2mult_${TASK}_${WBITS}bit${seed_suffix}_quantized_model"
    MASK_FLAT="$CKPT_DIR/${src_engine}_important_mask.pt" ;;
  *) echo "unknown METHOD=$METHOD" >&2; exit 1 ;;
esac

log "=== FIXED-MASK ablation METHOD=$METHOD TASK=$TASK w$WBITS gpu=$GPU ==="
if [[ ! -f "$MASK_FLAT" ]]; then
  log "ERROR: source mask not found: $MASK_FLAT (run the $METHOD task-conditioned step first)"; exit 2
fi

if [[ -f "$CKPT_DIR/$ENGINE.pt" && "${RESUME:-0}" == "1" && "${FORCE_RECOMPUTE:-0}" != "1" ]]; then
  log "RESUME: $ENGINE.pt exists, skipping to eval"
else
  log "gptq.llama masked $TASK (fixed $METHOD mask, uniform wbits=$WBITS)"
  cd "$ROOT/TACQ"
  CUDA_VISIBLE_DEVICES=$GPU python -m gptq.llama "$MODEL_LOAD" "$TASK" \
    --true-sequential --wbits "$WBITS" \
    --save_in_16bits "$CKPT_DIR/$ENGINE.pt" --no-eval --seed "$SEED" \
    --important_mask "$MASK_FLAT"
fi

cd "$ROOT/TACQ"
CUDA_VISIBLE_DEVICES=$GPU TASK="$TASK" ENGINE="$ENGINE" SEED="$SEED" \
  run_pretrain_eval_profile same
