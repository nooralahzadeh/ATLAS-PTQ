#!/bin/bash
# Run GPTQ on Llama-3-8B-Instruct with TaCQ saliency masks → mixed-precision checkpoint.
#
# Usage:
#   TESTING=1 bash scripts/run_gptq_tacq_base.sh   # 3 cal samples (smoke)
#   bash scripts/run_gptq_tacq_base.sh               # full 128 cal samples
#
# Outputs (kept on disk):
#   /home/ubuntu/tacq_data/checkpoints/Meta-Llama-3-8B-Instruct+0+Spider+tacq_saliency_w2+quantized_model.pt
#   /home/ubuntu/tacq_data/checkpoints/gptq_inputs/important_mask_w2.pt
#   /home/ubuntu/tacq_data/checkpoints/gptq_inputs/wbits_q2_uniform.yaml

set -euo pipefail

REPO=/home/ubuntu/ATLAS/TACQ
CHECKPOINTS=/home/ubuntu/tacq_data/checkpoints
GPTQ_INPUTS="$CHECKPOINTS/gptq_inputs"
LOG="$CHECKPOINTS/gptq_w2.log"
WBITS="${WBITS:-2}"
NSAMPLES="${NSAMPLES:-128}"
DEVICE="${DEVICE:-0}"

if [ "${TESTING:-0}" = "1" ]; then
  NSAMPLES=3
  echo "TESTING=1 → nsamples=$NSAMPLES"
fi

MODEL="Meta-Llama-3-8B-Instruct"
LOADSTRING="meta-llama/${MODEL}"
DATASET="Spider"
SEED=0
QUANTIZED_NAME="${MODEL}+${SEED}+${DATASET}+tacq_saliency_w${WBITS}+quantized_model"
OUT_PT="$CHECKPOINTS/${QUANTIZED_NAME}.pt"

cd "$REPO"
source tacq_venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -z "${HUGGINGFACE_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: Set HUGGINGFACE_TOKEN in TACQ/.env"
  exit 1
fi

mkdir -p "$CHECKPOINTS" "$GPTQ_INPUTS"

echo "=== Step 1: merge layer masks for GPTQ ==="
python scripts/merge_layer_masks_for_gptq.py \
  --mask-dir /home/ubuntu/tacq_data/importances \
  --wbits "$WBITS" \
  --output-dir "$GPTQ_INPUTS"

MASK_PT="$GPTQ_INPUTS/important_mask_w${WBITS}.pt"
YAML_PT="$GPTQ_INPUTS/wbits_q${WBITS}_uniform.yaml"

if [ -f "$OUT_PT" ] && [ "${FORCE_RECOMPUTE:-0}" != "1" ]; then
  echo "Checkpoint already exists: $OUT_PT"
  exit 0
fi

echo "=== Step 2: GPTQ (${WBITS}-bit base + FP16 outliers) ==="
echo "Started $(date -Is)" | tee "$LOG"
echo "Output: $OUT_PT" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$DEVICE python -m gptq.llama \
  "$LOADSTRING" \
  "$DATASET" \
  --true-sequential \
  --fine-wbits-yaml "$YAML_PT" \
  --important_mask "$MASK_PT" \
  --save_in_16bits "$OUT_PT" \
  --save_in_dtype_float16 \
  --nsamples "$NSAMPLES" \
  --seed "$SEED" \
  --no-eval \
  2>&1 | tee -a "$LOG"

echo "=== Done $(date -Is) ===" | tee -a "$LOG"
echo "Checkpoint: $OUT_PT"
ls -lh "$OUT_PT"
