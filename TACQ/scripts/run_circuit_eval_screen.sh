#!/bin/bash
# Assemble 32-layer TaCQ circuit + Spider eval. Requires all layer_*_saliency.pt files.
set -euo pipefail

REPO=/home/ubuntu/ATLAS/TACQ
MASK_DIR=/home/ubuntu/tacq_data/importances
LOG=/home/ubuntu/tacq_data/results/circuit_eval.log
OUT=/home/ubuntu/tacq_data/results/circuit_spider_sample.json
WBITS="${WBITS:-2}"
REQUIRED=32

cd "$REPO"
source tacq_venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi

COUNT=$(ls "$MASK_DIR"/layer_*_saliency.pt 2>/dev/null | wc -l)
if [ "$COUNT" -lt "$REQUIRED" ]; then
  echo "ERROR: found $COUNT / $REQUIRED layer mask files in $MASK_DIR"
  echo "Run full extraction first: bash scripts/run_full_extraction_screen.sh"
  exit 1
fi

echo "All $COUNT layer masks present. Starting assembly + eval …"
mkdir -p "$(dirname "$LOG")"

python scripts/assemble_and_evaluate_circuit.py \
  --mask-dir "$MASK_DIR" \
  --wbits "$WBITS" \
  --output "$OUT" \
  2>&1 | tee "$LOG"

echo "Done. Result: $OUT"
