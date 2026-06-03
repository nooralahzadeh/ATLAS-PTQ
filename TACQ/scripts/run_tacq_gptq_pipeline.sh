#!/bin/bash
# Full TaCQ GPTQ pipeline: build checkpoint → Spider eval.
# Launch in screen for long runs:
#   screen -S tacq_gptq
#   bash scripts/run_tacq_gptq_pipeline.sh
#   # Ctrl+A D

set -euo pipefail
REPO=/home/ubuntu/ATLAS/TACQ
cd "$REPO"

echo "=== Phase 1: GPTQ base checkpoint ==="
bash scripts/run_gptq_tacq_base.sh

echo "=== Phase 2: Spider exec eval ==="
bash scripts/run_tacq_gptq_spider_eval.sh

echo "=== Pipeline complete ==="
