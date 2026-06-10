#!/bin/bash
# In-memory FP16 circuit boost/suppression counter-test (Spider + English probes).
#
# Loads FP16 Llama-3.1, scales masked weights: W *= 1 + (alpha-1)*Mask
#   boost:     alpha=1.2  (+20% on B0 heavy 0.35% circuit)
#   suppress:  alpha=0.8  (-20% on circuit; Spider should crash, English OK)
#
# Baseline reference (unmodified FP16): 0.678 exec (see OBSERVATIONS.md)
#
# Smoke:
#   TESTING=1 bash scripts/run_circuit_scale_ablation.sh
# Full:
#   srun -A a0125 -p normal -t 03:00:00 -N1 -n1 --gres=gpu:1 \
#     --environment=vscode-pytorch bash scripts/run_circuit_scale_ablation.sh
set -euo pipefail

ROOT=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ
MASK="${MASK:-$ROOT/masks/tdso_b0_heavy_0.35.pt}"
export HF_HOME=/capstor/scratch/cscs/fnoorala/.cache
export PYTHONUNBUFFERED=1
export NLTK_DATA=$ROOT/TACQ/datasets_directory/Spider/third_party/nltk_data
LOG=$ROOT/tacq_data/results/circuit_scale_ablation.log
mkdir -p "$(dirname "$LOG")"

TESTING_FLAG=""
if [ "${TESTING:-0}" = "1" ]; then
  TESTING_FLAG="--testing"
fi

cd "$ROOT"
source tdso_venv/bin/activate 2>/dev/null || source llama31_venv/bin/activate
set -a; [ -f TACQ/.env ] && source TACQ/.env; set +a

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_one() {
  local alpha="$1" tag="$2"
  log "=== alpha=$alpha ($tag) ==="
  python scripts/evaluation/circuit_scale_ablation.py \
    --mask "$MASK" \
    --alpha "$alpha" \
    --tag "$tag" \
    $TESTING_FLAG \
    2>&1 | tee -a "$LOG"
}

log "circuit scale ablation | mask=$MASK"
run_one 1.2 "boost_1p2"
run_one 0.8 "suppress_0p8"

log "=== SUMMARY (compare to FP16 baseline 0.678) ==="
for d in "$ROOT/tacq_data/results/Spider/circuit_scale_boost_1p2" \
         "$ROOT/tacq_data/results/Spider/circuit_scale_suppress_0p8"; do
  if [ -f "$d/summary.json" ]; then
    python -c "import json; s=json.load(open('$d/summary.json')); print(s.get('tag'), 'exec=', s.get('spider_exec_acc'))"
  fi
done
log "DONE"
