#!/bin/bash
# Evaluate GPTQ+TaCQ checkpoint on Spider (exec accuracy via test-suite-sql-eval).
#
# Prerequisite: bash scripts/run_gptq_tacq_base.sh
#
# Usage:
#   TESTING=1 bash scripts/run_tacq_gptq_spider_eval.sh   # 4 dev examples
#   bash scripts/run_tacq_gptq_spider_eval.sh             # full dev set

set -euo pipefail

REPO=/home/ubuntu/ATLAS/TACQ
CHECKPOINTS=/home/ubuntu/tacq_data/checkpoints
RESULTS=/home/ubuntu/tacq_data/results
WBITS="${WBITS:-2}"
DEVICE="${DEVICE:-0}"

MODEL="Meta-Llama-3-8B-Instruct"
QUANTIZED_NAME="${MODEL}+0+Spider+tacq_saliency_w${WBITS}+quantized_model"
OUT_PT="$CHECKPOINTS/${QUANTIZED_NAME}.pt"

if [ ! -f "$OUT_PT" ]; then
  echo "Missing checkpoint: $OUT_PT"
  echo "Run: bash scripts/run_gptq_tacq_base.sh"
  exit 1
fi

cd "$REPO"
source tacq_venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi

tables="datasets_directory/Spider/data/spider/tables.json"
dataset_path="datasets_directory/Spider/data/spider/dev.json"
dev_gold_path="datasets_directory/Spider/data/spider/dev_gold.sql"
db_dir="datasets_directory/Spider/database"
output_savedir="$RESULTS/Spider/${QUANTIZED_NAME}"
mkdir -p "$output_savedir"

TESTING_FLAG=""
if [ "${TESTING:-0}" = "1" ]; then
  TESTING_FLAG="--testing"
fi

echo "=== Spider inference: $QUANTIZED_NAME ==="
CUDA_VISIBLE_DEVICES=$DEVICE python -m datasets_directory.Spider.Spider_eval \
  --model "$QUANTIZED_NAME" \
  --input "$dataset_path" \
  --tables "$tables" \
  --predictions_filename predictions.txt \
  --output "${output_savedir}/debug.txt" \
  --output_savedir "$output_savedir" \
  --addition_dir "$CHECKPOINTS" \
  $TESTING_FLAG

export NLTK_DATA="datasets_directory/Spider/third_party/nltk_data"
python datasets_directory/Spider/third_party/test-suite-sql-eval/evaluation.py \
  --gold "$dev_gold_path" \
  --pred "${output_savedir}/predictions.txt" \
  --db "$db_dir" \
  --table "$tables" \
  --etype exec \
  2>&1 | tee "${output_savedir}/eval_exec.log"

echo "Results: $output_savedir"
