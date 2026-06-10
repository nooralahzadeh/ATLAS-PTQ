#!/bin/bash
# T-DSO v2 (mult) task-conditioned quantization for ONE task (Llama-3.1-8B-Instruct).
#
# RECONSTRUCTED after the 2026-06-10 scratch incident — VERIFY with a smoke run.
# Grounded on the surviving [cfg] log lines (combine=mult lam=1.0 frac=0.0035
# budget=global q=0.95 bs=2 max_len=2048), engine names of tacq_data/*.pt
# (…_tdsoV2mult_<TASK>_<bits>bit…), the rebuilt extractor/convert, and the
# TACQ gptq --important_mask contract.
#
# Pipeline: extract_tdso_v2_h200.py --combine mult -> convert_tdso_mask.py
#   -> gptq masked (uniform --wbits, fp16 outliers via mask) -> eval -> RESULT.
#
# Env: TASK SEED GPU WBITS COMBINE FORCE_RECOMPUTE RESUME EXTRACT_BS MMLU_MAX_LEN
set -euo pipefail
ROOT="${ROOT:-/capstor/scratch/cscs/fnoorala/ATLAS-PTQ}"

MODEL_LOAD="${MODEL_LOAD:-unsloth/Meta-Llama-3.1-8B-Instruct}"
MODEL_BASE="${MODEL_BASE:-Meta-Llama-3.1-8B-Instruct}"
TASK="${TASK:?TASK required}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
WBITS="${WBITS:-2}"
COMBINE="${COMBINE:-mult}"
LAM="${LAM:-1.0}"
FRAC="${FRAC:-0.0035}"
QUANTILE="${QUANTILE:-0.95}"
EXTRACT_BS="${EXTRACT_BS:-2}"
MAX_LEN="${MMLU_MAX_LEN:-2048}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"

CKPT_DIR="${CKPT_DIR:-$ROOT/tacq_data}"
export RESULTS_DIR="${RESULTS_DIR:-$ROOT/tacq_data/results}"
export CKPT_DIR
source "$ROOT/scripts/pretrain_eval_helpers.sh"
log() { echo "[$(date +%H:%M:%S)] $*"; }

seed_suffix=""; [[ "$SEED" != "0" ]] && seed_suffix="_s${SEED}"

# Per-task calibration pairs (train/calib splits only — see calib_split_policy.py)
case "$TASK" in
  GSM8k)                PAIRS="data/contrastive/gsm8k_contrastive_tacq${seed_suffix}.jsonl" ;;
  MMLU_STEM)            PAIRS="data/contrastive/mmlu_stem_contrastive_test75${seed_suffix}.jsonl" ;;
  MMLU_humanities)      PAIRS="data/contrastive/mmlu_humanities_contrastive_test75${seed_suffix}.jsonl" ;;
  MMLU_social_sciences) PAIRS="data/contrastive/mmlu_social_sciences_contrastive_test75${seed_suffix}.jsonl" ;;
  Spider)               PAIRS="data/contrastive/spider_contrastive_train1360${seed_suffix}.jsonl" ;;
  *) echo "unknown TASK=$TASK" >&2; exit 1 ;;
esac
PAIRS="${PAIRS_OVERRIDE:-$PAIRS}"

tag="tdsoV2$([[ "$COMBINE" == "mult" ]] && echo mult || echo "$COMBINE")"
ENGINE="${MODEL_BASE}_${tag}_${TASK}_${WBITS}bit${seed_suffix}_quantized_model"
MASK_RAW="$CKPT_DIR/tdso_v2_${TASK,,}_${WBITS}bit_task${seed_suffix}_mask.pt"
MASK_FLAT="$CKPT_DIR/${ENGINE}_important_mask.pt"
CORRUPT_PT="$CKPT_DIR/${MODEL_BASE}+${SEED}+${TASK}+${WBITS}bit+quantized_model.pt"

log "=== T-DSO v2 $COMBINE TASK=$TASK w$WBITS gpu=$GPU seed=$SEED ==="

if [[ "$FORCE_RECOMPUTE" == "1" ]]; then
  log "FORCE_RECOMPUTE: clearing $TASK w$WBITS T-DSO artifacts"
  rm -f "$MASK_RAW" "$MASK_FLAT" "$CKPT_DIR/$ENGINE.pt"
fi

if [[ -f "$CKPT_DIR/$ENGINE.pt" && "${RESUME:-0}" == "1" ]]; then
  log "RESUME: $ENGINE.pt exists, skipping to eval"
else
  corrupt_arg=()
  [[ -f "$CORRUPT_PT" ]] && corrupt_arg=(--corrupt-model "$CORRUPT_PT")

  if [[ ! -f "$MASK_RAW" || "$FORCE_RECOMPUTE" == "1" ]]; then
    log "extract_tdso_v2_h200 $TASK combine=$COMBINE bits=$WBITS"
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES=$GPU python scripts/extraction/extract_tdso_v2_h200.py \
      --pairs "$PAIRS" --model "$MODEL_LOAD" --bits "$WBITS" \
      --mask-fraction "$FRAC" --quantile "$QUANTILE" --combine "$COMBINE" --lam "$LAM" \
      --batch-size "$EXTRACT_BS" --max-len "$MAX_LEN" --seed "$SEED" \
      --mask-budget global --out "$MASK_RAW" "${corrupt_arg[@]}"
  fi

  log "convert_tdso_mask -> flat important_mask"
  python "$ROOT/scripts/extraction/convert_tdso_mask.py" --in "$MASK_RAW" --out "$MASK_FLAT"

  log "gptq.llama masked $TASK (uniform wbits=$WBITS, fp16 outliers via mask)"
  cd "$ROOT/TACQ"
  CUDA_VISIBLE_DEVICES=$GPU python -m gptq.llama "$MODEL_LOAD" "$TASK" \
    --true-sequential --wbits "$WBITS" \
    --save_in_16bits "$CKPT_DIR/$ENGINE.pt" --no-eval --seed "$SEED" \
    --important_mask "$MASK_FLAT"
fi

cd "$ROOT/TACQ"
CUDA_VISIBLE_DEVICES=$GPU TASK="$TASK" ENGINE="$ENGINE" SEED="$SEED" \
  run_pretrain_eval_profile same
