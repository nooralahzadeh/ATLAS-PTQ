#!/bin/bash
# Create llama31_venv: TaCQ + Spider eval for Meta-Llama-3.1-8B-Instruct.
#
# Bootstraps virtualenv via tdso_venv (container-only), then installs TaCQ deps.
# Run inside vscode-pytorch:
#   srun --environment=vscode-pytorch bash scripts/setup_llama31_venv.sh
#
# Llama-3-8B reproduction continues to use TACQ/tacq_venv (transformers 4.40).

set -euo pipefail

ROOT="/capstor/scratch/cscs/fnoorala/ATLAS-PTQ"
VENV="${ROOT}/llama31_venv"
REQ="${ROOT}/TACQ/requirements-llama31.txt"
TDSO="${ROOT}/tdso_venv"

cd "$ROOT"

if [ ! -f "$REQ" ]; then
    echo "Missing $REQ"
    exit 1
fi

verify_venv() {
    # shellcheck disable=SC1090
    source "${VENV}/bin/activate"
    python - <<'PY'
import torch
import transformers
v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
assert v >= (4, 43), transformers.__version__
print("llama31_venv OK")
print("  torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("  transformers:", transformers.__version__)
PY
}

if [ -f "${VENV}/bin/activate" ]; then
    if verify_venv 2>/dev/null; then
        exit 0
    fi
    echo "llama31_venv exists but incomplete; finishing pip install..."
    # shellcheck disable=SC1090
    source "${VENV}/bin/activate"
    pip install -U pip wheel
    pip install -r "$REQ"
    verify_venv
    exit 0
fi

if [ ! -f "${TDSO}/bin/activate" ]; then
    echo "ERROR: tdso_venv required to bootstrap (missing ${TDSO})"
    exit 1
fi

echo "Bootstrapping virtualenv from tdso_venv..."
# shellcheck disable=SC1090
source "${TDSO}/bin/activate"
pip install -q virtualenv
python -m virtualenv --system-site-packages "$VENV"
deactivate

echo "Installing Llama-3.1 TaCQ requirements into llama31_venv..."
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
pip install -U pip wheel
pip install -r "$REQ"

verify_venv
