#!/bin/bash
# Paper 2 — B1 FP16 DIN-SQL multistep on full Spider dev (4-GPU shard parallel).
#
# Baseline config (locked from smoke job 2507824):
#   MODEL=unsloth/Meta-Llama-3.1-8B-Instruct
#   PROMPT_PROFILE=lite  PROMPT_FORMAT=raw  ICE_SHOTS=2
#
# Env knobs:
#   DEV_TOTAL, NUM_GPUS, MODEL, PROMPT_PROFILE, PROMPT_FORMAT, ICE_SHOTS,
#   MAX_PROMPT_TOKENS, RUN_DIR (optional fixed output root)
set -euo pipefail

ROOT=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ
DEV_TOTAL="${DEV_TOTAL:-1034}"
NUM_GPUS="${NUM_GPUS:-4}"
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

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$ROOT/tacq_data/results/paper2/b1_${PROMPT_PROFILE}_${PROMPT_FORMAT}_dev_${STAMP}}"
mkdir -p "$RUN_DIR"

PER_SHARD=$(( (DEV_TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

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

log "=== Paper2 B1 full Spider dev ==="
log "model=$MODEL dev_total=$DEV_TOTAL num_gpus=$NUM_GPUS per_shard=$PER_SHARD"
log "profile=$PROMPT_PROFILE format=$PROMPT_FORMAT ice_shots=$ICE_SHOTS"
log "run_dir=$RUN_DIR"

run_shard() {
  local gpu="$1"
  local offset="$2"
  local count="$3"
  local shard_dir="$RUN_DIR/shard_${gpu}"
  mkdir -p "$shard_dir"

  log "shard gpu=$gpu offset=$offset count=$count -> $shard_dir"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/agents/spider_multistep_eval.py \
    --agent dinsql \
    --mode fp16 \
    --model "$MODEL" \
    --device "cuda:0" \
    --offset "$offset" \
    --max-examples "$count" \
    --output-dir "$shard_dir" \
    --prompt-profile "$PROMPT_PROFILE" \
    --prompt-format "$PROMPT_FORMAT" \
    --ice-shots "$ICE_SHOTS" \
    "${EXTRA_ARGS[@]}"
}

PIDS=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
  offset=$((gpu * PER_SHARD))
  if ((offset >= DEV_TOTAL)); then
    break
  fi
  remain=$((DEV_TOTAL - offset))
  count=$((remain < PER_SHARD ? remain : PER_SHARD))
  run_shard "$gpu" "$offset" "$count" &
  PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done
if ((FAIL != 0)); then
  log "ERROR: one or more shards failed"
  exit 1
fi

MERGE_ARGS=(--expected "$DEV_TOTAL" --out "$RUN_DIR/predictions_clean_merged.txt" --run-exec-eval)
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
  offset=$((gpu * PER_SHARD))
  if ((offset >= DEV_TOTAL)); then
    break
  fi
  remain=$((DEV_TOTAL - offset))
  count=$((remain < PER_SHARD ? remain : PER_SHARD))
  MERGE_ARGS+=(--shard "$RUN_DIR/shard_${gpu}/predictions_clean.txt")
  MERGE_ARGS+=(--shard-count "$count")
done

python scripts/merge_spider_shard_preds.py "${MERGE_ARGS[@]}"

CONFIG_JSON="$RUN_DIR/b1_config.json"
python - <<PY
import json
from pathlib import Path
cfg = {
    "baseline": "B1",
    "model": "$MODEL",
    "prompt_profile": "$PROMPT_PROFILE",
    "prompt_format": "$PROMPT_FORMAT",
    "ice_shots": int("$ICE_SHOTS"),
    "dev_total": int("$DEV_TOTAL"),
    "num_gpus": int("$NUM_GPUS"),
    "per_shard": int("$PER_SHARD"),
    "run_dir": "$RUN_DIR",
    "merged_predictions": "$RUN_DIR/predictions_clean_merged.txt",
    "b0_reference_exec": 67.8,
}
Path("$CONFIG_JSON").write_text(json.dumps(cfg, indent=2) + "\\n")
PY

log "DONE run_dir=$RUN_DIR"
log "Merged predictions: $RUN_DIR/predictions_clean_merged.txt"
log "Config: $CONFIG_JSON"
