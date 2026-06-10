#!/bin/bash
# TaCQ task-conditioned quantization for ONE task (Llama-3.1-8B-Instruct).
#
# RECONSTRUCTED after the 2026-06-10 scratch incident — VERIFY with a --testing
# smoke run before production. Grounded on the canonical (surviving) TaCQ pipeline
# TACQ/scripts/examples/evaluate_llama3_8b.sh, the eval modules, the log step
# markers, and the engine names of surviving tacq_data/*.pt.
#
# Pipeline: build per-bit corrupt -> measure_importances (contrastive, full grads)
#   -> make_quantization_configs (mask @ ratio) -> gptq masked -> eval -> RESULT.
#
# Env (set by caller): TASK SEED GPU WBITS FORCE_RECOMPUTE RESUME
set -euo pipefail
ROOT="${ROOT:-/capstor/scratch/cscs/fnoorala/ATLAS-PTQ}"
cd "$ROOT/TACQ"

MODEL_LOAD="${MODEL_LOAD:-unsloth/Meta-Llama-3.1-8B-Instruct}"
MODEL_BASE="${MODEL_BASE:-Meta-Llama-3.1-8B-Instruct}"
TASK="${TASK:?TASK required}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
WBITS="${WBITS:-2}"
RATIO="${RATIO:-.0035}"
QTYPE="q${WBITS}"
RANKING="${RANKING:-top_p_sparse}"
SELECTOR="${SELECTOR:-sample_abs_weight_prod_contrastive}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"

CKPT_DIR="${CKPT_DIR:-$ROOT/tacq_data}"
IMP_DIR="${IMP_DIR:-$ROOT/tacq_data/importances}"
export RESULTS_DIR="${RESULTS_DIR:-$ROOT/tacq_data/results}"
export CKPT_DIR
source "$ROOT/scripts/pretrain_eval_helpers.sh"

seed_suffix=""; [[ "$SEED" != "0" ]] && seed_suffix="_s${SEED}"
ENGINE="${MODEL_BASE}_tacq_${TASK}_${WBITS}bit${seed_suffix}_quantized_model"
RUN_NAME="${MODEL_BASE}+${SEED}+${TASK}+${SELECTOR}+${WBITS}bit+conditioned"
CORRUPT_NAME="${MODEL_BASE}+${SEED}+${TASK}+${WBITS}bit+quantized_model"
QUANT_ID="${QTYPE}+${RANKING}+${RATIO}"
MASK_PT="$IMP_DIR/$RUN_NAME/important_mask_${QUANT_ID}.pt"
CFG_YAML="$IMP_DIR/$RUN_NAME/quantization_configs_${QUANT_ID}.yaml"

log() { echo "[$(date +%H:%M:%S)] $*"; }
log "=== TASK=$TASK w$WBITS gpu=$GPU seed=$SEED ==="

if [[ "$FORCE_RECOMPUTE" == "1" ]]; then
  log "FORCE_RECOMPUTE: clearing $TASK w$WBITS artifacts"
  rm -f "$CKPT_DIR/$ENGINE.pt" "$MASK_PT" "$CFG_YAML" "$IMP_DIR/$RUN_NAME/importances.pt"
fi

if [[ -f "$CKPT_DIR/$ENGINE.pt" && "${RESUME:-0}" == "1" ]]; then
  log "RESUME: $ENGINE.pt exists, skipping to eval"
else
  if [[ ! -f "$CKPT_DIR/$CORRUPT_NAME.pt" ]]; then
    log "[gpu=$GPU] building corrupt $TASK w$WBITS"
    CUDA_VISIBLE_DEVICES=$GPU python -m gptq.llama "$MODEL_LOAD" "$TASK" \
      --true-sequential --save_in_16bits "$CKPT_DIR/$CORRUPT_NAME.pt" \
      --wbits "$WBITS" --seed "$SEED" --no-eval
  fi

  log "measure_importances $TASK (save_full_gradients)"
  CUDA_VISIBLE_DEVICES=$GPU python -m measure_importances \
    --model "$MODEL_BASE" --corrupt_model "$CORRUPT_NAME" --dataset "$TASK" \
    --run_name "$RUN_NAME" --checkpoints_dir "$CKPT_DIR" --results_dir "$IMP_DIR" \
    --selector_type "$SELECTOR" --serial_number "$SEED" \
    --save_full_gradients --save_importances_pt_path "$IMP_DIR/$RUN_NAME/importances.pt" \
    --override_args_yaml

  log "make_quantization_configs $TASK"
  CUDA_VISIBLE_DEVICES=$GPU python -m make_quantization_configs \
    --run_name "$RUN_NAME" --checkpoints_dir "$CKPT_DIR" --results_dir "$IMP_DIR" \
    --serial_number "$SEED" --importances_pt_path "$IMP_DIR/$RUN_NAME/importances.pt" \
    --mask_save_path "$MASK_PT" --model "$MODEL_BASE" \
    --quantization_type "$QTYPE" --ranking_type "$RANKING" \
    --configs_save_path "$CFG_YAML" --mask_fraction "$RATIO" \
    --proportional_total_params --force_recompute

  log "gptq.llama masked $TASK"
  CUDA_VISIBLE_DEVICES=$GPU python -m gptq.llama "$MODEL_LOAD" "$TASK" \
    --true-sequential --fine-wbits-yaml "$CFG_YAML" \
    --save_in_16bits "$CKPT_DIR/$ENGINE.pt" --no-eval --seed "$SEED" \
    --important_mask "$MASK_PT"
fi

CUDA_VISIBLE_DEVICES=$GPU TASK="$TASK" ENGINE="$ENGINE" SEED="$SEED" \
  run_pretrain_eval_profile same
