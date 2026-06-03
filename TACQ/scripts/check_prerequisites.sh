#!/bin/bash
# Validate TaCQ Spider replication prerequisites (no GPU model download).
cd "$(dirname "$0")/.." || exit 1
source tacq_venv/bin/activate

set -e
python scripts/validate_setup.py
