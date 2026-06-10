#!/bin/bash
# Paper 2 — B1 FP16 DIN-SQL multistep smoke on Spider dev.
#
# Env knobs:
#   MODEL, PROMPT_PROFILE=(full|lite|minimal), PROMPT_FORMAT=(raw|chat),
#   MAX_EXAMPLES, MAX_PROMPT_TOKENS, ICE_SHOTS
set -euo pipefail

ROOT=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAX_EXAMPLES="${MAX_EXAMPLES:-4}"
MODEL="${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
PROMPT_PROFILE="${PROMPT_PROFILE:-lite}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
ICE_SHOTS="${ICE_SHOTS:-2}"

cd "$ROOT"
source llama31_venv/bin/activate
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NLTK_DATA="$ROOT/TACQ/datasets_directory/Spider/third_party/nltk_data"
export HF_HOME="${HF_HOME:-/capstor/scratch/cscs/fnoorala/.cache}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

EXTRA_ARGS=()
if [[ -n "${MAX_PROMPT_TOKENS:-}" ]]; then
  EXTRA_ARGS+=(--max-prompt-tokens "$MAX_PROMPT_TOKENS")
fi
if [[ "${NO_SKIP_CLASSIFY_EASY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-skip-classify-easy)
fi
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--trust-remote-code)
fi

log "=== Paper2 B1 smoke ==="
log "model=$MODEL gpu=$GPU max_examples=$MAX_EXAMPLES profile=$PROMPT_PROFILE format=$PROMPT_FORMAT ice_shots=$ICE_SHOTS"

CUDA_VISIBLE_DEVICES="$GPU" python scripts/agents/spider_multistep_eval.py \
  --agent dinsql \
  --mode fp16 \
  --model "$MODEL" \
  --device "cuda:$GPU" \
  --max-examples "$MAX_EXAMPLES" \
  --prompt-profile "$PROMPT_PROFILE" \
  --prompt-format "$PROMPT_FORMAT" \
  --ice-shots "$ICE_SHOTS" \
  --verbose \
  "${EXTRA_ARGS[@]}"

log "DONE"
