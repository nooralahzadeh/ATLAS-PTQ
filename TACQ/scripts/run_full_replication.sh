#!/bin/bash
# Full TaCQ Spider replication: unquantized baseline, smoke test, then full q2+q3 run.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
source tacq_venv/bin/activate
if [ -f .env ]; then set -a; source .env; set +a; fi

if [ -z "${HUGGINGFACE_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "Set HUGGINGFACE_TOKEN in TACQ/.env then re-run."
    exit 1
fi

echo "=== Step 1: Unquantized Spider baseline ==="
bash scripts/run_unquantized_baseline.sh

echo "=== Step 2: Smoke test (TESTING=1, q2 only) ==="
TESTING=1 bash scripts/examples/evaluate_llama3_8b_spider_l4.sh

echo "=== Step 3: Full Spider TaCQ (q2 + q3) ==="
TESTING=0 bash scripts/examples/evaluate_llama3_8b_spider_l4.sh

echo "=== Done. Results under /home/ubuntu/tacq_data/results/Spider/ ==="
python scripts/collect_results.py
