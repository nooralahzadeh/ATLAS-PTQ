#!/bin/bash
# Contemporaneous FP16 Spider dev baseline (same harness as steer squeeze dev-final).
#
# Run: sbatch scripts/sbatch_evaluate_base_fp16.sh
# Smoke: TESTING=1 bash scripts/run_evaluate_base_fp16.sh
set -euo pipefail

ROOT=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ
export HF_HOME=/capstor/scratch/cscs/fnoorala/.cache
export PYTHONUNBUFFERED=1
export NLTK_DATA=$ROOT/TACQ/datasets_directory/Spider/third_party/nltk_data

SPLIT="${SPLIT:-dev}"
LOG=$ROOT/tacq_data/results/fp16_baseline_${SPLIT}.log
mkdir -p "$(dirname "$LOG")"

TESTING_FLAG=""
if [ "${TESTING:-0}" = "1" ]; then
  TESTING_FLAG="--testing"
fi

cd "$ROOT"
source tdso_venv/bin/activate 2>/dev/null || source llama31_venv/bin/activate
set -a; [ -f TACQ/.env ] && source TACQ/.env; set +a

echo "[$(date +%H:%M:%S)] evaluate_base_fp16 split=$SPLIT" | tee "$LOG"
python scripts/evaluation/evaluate_base_fp16.py \
  --split "$SPLIT" \
  $TESTING_FLAG \
  2>&1 | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] DONE -> tacq_data/results/Spider/fp16_baseline/$SPLIT/summary.json" | tee -a "$LOG"
