#!/bin/bash
# Full 32-layer TaCQ saliency extraction (L4-safe, one layer at a time).
# Writes /home/ubuntu/tacq_data/importances/layer_0_saliency.pt … layer_31_saliency.pt
#
# Launch detached:
#   bash scripts/run_full_extraction_screen.sh
#
# Monitor:
#   tail -f /home/ubuntu/tacq_data/importances/extraction.log
#   watch -n 10 'ls /home/ubuntu/tacq_data/importances/layer_*_saliency.pt 2>/dev/null | wc -l'
#   screen -r tacq_extract

set -euo pipefail

REPO=/home/ubuntu/ATLAS/TACQ
LOG=/home/ubuntu/tacq_data/importances/extraction.log
OUT_DIR=/home/ubuntu/tacq_data/importances
WBITS="${WBITS:-2}"
FRACTION="${FRACTION:-0.015}"
NUM_EX="${NUM_EX:-128}"

cd "$REPO"
source tacq_venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi

mkdir -p "$OUT_DIR"
echo "=== TaCQ full extraction started $(date -Is) ===" | tee "$LOG"
echo "Output dir: $OUT_DIR" | tee -a "$LOG"
echo "wbits=$WBITS  mask_fraction=$FRACTION  num_examples=$NUM_EX" | tee -a "$LOG"

python scripts/extract_tacq_saliency.py \
  --wbits "$WBITS" \
  --mask-fraction "$FRACTION" \
  --num-examples "$NUM_EX" \
  --layer-output-dir "$OUT_DIR" \
  --output "$OUT_DIR/spider_tacq_saliency_w${WBITS}.pt" \
  2>&1 | tee -a "$LOG"

COUNT=$(ls "$OUT_DIR"/layer_*_saliency.pt 2>/dev/null | wc -l)
echo "=== Finished $(date -Is): $COUNT / 32 layer files ===" | tee -a "$LOG"
