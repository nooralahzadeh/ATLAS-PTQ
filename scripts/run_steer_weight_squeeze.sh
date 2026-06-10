#!/bin/bash
# Contrastive SNR squeeze — leak-free evaluation pipeline.
#
# Phase grid:       train-only hyperparameter search (default subset 150)
# Phase dev-final:  single pass on dev with frozen_alphas.json
# Phase pipeline:   grid → freeze → one dev pass (recommended)
#
# Smoke (4 train grid configs + dev pass):
#   TESTING=1 bash scripts/run_steer_weight_squeeze.sh
#
# Full pipeline (150-train grid + dev final):
#   bash scripts/run_steer_weight_squeeze.sh
#
# Grid only (no dev touch):
#   PHASE=grid bash scripts/run_steer_weight_squeeze.sh
#
# Dev final only (after grid):
#   PHASE=dev-final bash scripts/run_steer_weight_squeeze.sh
#
# Cluster:
#   sbatch scripts/sbatch_steer_weight_squeeze.sh
set -euo pipefail

ROOT=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ
export HF_HOME=/capstor/scratch/cscs/fnoorala/.cache
export PYTHONUNBUFFERED=1
export NLTK_DATA=$ROOT/TACQ/datasets_directory/Spider/third_party/nltk_data

MASK="${MASK:-$ROOT/masks/tdso_b0_heavy_0.35.pt}"
PHASE="${PHASE:-pipeline}"
SUBSET_SIZE="${SUBSET_SIZE:-150}"
LOG=$ROOT/tacq_data/results/steer_weight_squeeze.log
OUT=$ROOT/tacq_data/results/Spider/steer_squeeze
mkdir -p "$(dirname "$LOG")"

TESTING_FLAG=""
if [ "${TESTING:-0}" = "1" ]; then
  TESTING_FLAG="--testing"
fi

FULL_POOL_FLAG=""
if [ "${FULL_POOL:-0}" = "1" ]; then
  FULL_POOL_FLAG="--full-pool"
fi

FROZEN_FLAG=""
if [ -n "${FROZEN_ALPHAS:-}" ]; then
  FROZEN_FLAG="--frozen-alphas $FROZEN_ALPHAS"
fi

cd "$ROOT"
source tdso_venv/bin/activate 2>/dev/null || source llama31_venv/bin/activate
set -a; [ -f TACQ/.env ] && source TACQ/.env; set +a

echo "[$(date +%H:%M:%S)] steer_weight_squeeze phase=$PHASE subset=$SUBSET_SIZE" | tee "$LOG"
python scripts/evaluation/steer_weight_squeeze.py \
  --phase "$PHASE" \
  --mask "$MASK" \
  --subset-size "$SUBSET_SIZE" \
  --output-root "$OUT" \
  $TESTING_FLAG \
  $FULL_POOL_FLAG \
  $FROZEN_FLAG \
  2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] DONE phase=$PHASE" | tee -a "$LOG"
echo "  grid:      $OUT/grid_summary.json" | tee -a "$LOG"
echo "  frozen:    $OUT/frozen_alphas.json" | tee -a "$LOG"
echo "  dev final: $OUT/dev_final_summary.json (paper metric)" | tee -a "$LOG"
