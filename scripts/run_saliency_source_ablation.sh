#!/bin/bash
# Saliency-source ablation: everything fixed except the saliency signal g.
# Arms: ce | align | mult (transcoder) ; align_free | mult_free (dictionary-free) ;
#       weight | magnitude | random (non-circuit controls). Matched 0.35% budget,
#       same corrupt ΔW, same calib pairs, same GPTQ, same eval profile.
#
# RECONSTRUCTED after the 2026-06-10 scratch incident (session-authored) — VERIFY.
# Phase 1 builds masks (heavy arms SEQUENTIAL to avoid the ~56 GB/proc fp32-grad
# CPU OOM; dumb baselines parallel). Phase 2 runs GPTQ+eval round-robin on 4 GPUs.
# Env: ARMS TASKS WBITS SEED FORCE_RECOMPUTE
set -euo pipefail
ROOT="${ROOT:-/capstor/scratch/cscs/fnoorala/ATLAS-PTQ}"
cd "$ROOT"
MODEL_LOAD="${MODEL_LOAD:-unsloth/Meta-Llama-3.1-8B-Instruct}"
MODEL_BASE="${MODEL_BASE:-Meta-Llama-3.1-8B-Instruct}"
WBITS="${WBITS:-2}"
SEED="${SEED:-0}"
FRAC="${FRAC:-0.0035}"
MAX_LEN="${MAX_LEN:-4096}"
EXTRACT_BS="${EXTRACT_BS:-2}"
ARMS=(${ARMS:-ce align mult align_free mult_free weight magnitude random})
TASKS=(${TASKS:-GSM8k MMLU_STEM})
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"
CKPT_DIR="${CKPT_DIR:-$ROOT/tacq_data}"
export RESULTS_DIR="${RESULTS_DIR:-$ROOT/tacq_data/results}"
export CKPT_DIR
source "$ROOT/scripts/pretrain_eval_helpers.sh"
log() { echo "[$(date +%H:%M:%S)] $*"; }

pairs_for() {
  local task="$1" suf=""; [[ "$SEED" != "0" ]] && suf="_s${SEED}"
  case "$task" in
    GSM8k)                echo "data/contrastive/gsm8k_contrastive_tacq${suf}.jsonl" ;;
    MMLU_STEM)            echo "data/contrastive/mmlu_stem_contrastive_test75${suf}.jsonl" ;;
    MMLU_humanities)      echo "data/contrastive/mmlu_humanities_contrastive_test75${suf}.jsonl" ;;
    MMLU_social_sciences) echo "data/contrastive/mmlu_social_sciences_contrastive_test75${suf}.jsonl" ;;
    Spider)               echo "data/contrastive/spider_contrastive_train1360${suf}.jsonl" ;;
  esac
}
corrupt_for() { echo "$CKPT_DIR/${MODEL_BASE}+${SEED}+${1}+${WBITS}bit+quantized_model.pt"; }
raw_path_for()  { echo "$CKPT_DIR/ablsrc_${2}_${1,,}_${WBITS}bit${3}_maskraw.pt"; }
flat_path_for() { echo "$CKPT_DIR/ablsrc_${2}_${1,,}_${WBITS}bit${3}_maskflat.pt"; }
engine_for()    { local s=""; [[ "$SEED" != "0" ]] && s="_s${SEED}"; echo "${MODEL_BASE}_ablsrc_${2}_${1}_${WBITS}bit${s}_quantized_model"; }

build_mask_flat() {
  local task="$1" arm="$2" gpu="$3"
  local seed_suffix=""; [[ "$SEED" != "0" ]] && seed_suffix="_s${SEED}"
  local raw flat pairs corrupt
  raw="$(raw_path_for "$task" "$arm" "$seed_suffix")"
  flat="$(flat_path_for "$task" "$arm" "$seed_suffix")"
  pairs="$(pairs_for "$task")"
  corrupt="$(corrupt_for "$task")"
  local corrupt_arg=(); [[ -f "$corrupt" ]] && corrupt_arg=(--corrupt-model "$corrupt")
  if [[ -f "$flat" && "$FORCE_RECOMPUTE" != "1" ]]; then log "[mask] exists: $flat"; return 0; fi

  case "$arm" in
    ce|align|mult)
      CUDA_VISIBLE_DEVICES=$gpu python scripts/extraction/extract_tdso_v2_h200.py \
        --pairs "$pairs" --model "$MODEL_LOAD" --bits "$WBITS" --mask-fraction "$FRAC" \
        --combine "$arm" --batch-size "$EXTRACT_BS" --max-len "$MAX_LEN" --seed "$SEED" \
        --mask-budget global --out "$raw" "${corrupt_arg[@]}" ;;
    align_free|mult_free)
      local c="${arm%_free}"
      CUDA_VISIBLE_DEVICES=$gpu python scripts/extraction/extract_dictfree_saliency.py \
        --pairs "$pairs" --model "$MODEL_LOAD" --bits "$WBITS" --mask-fraction "$FRAC" \
        --combine "$c" --feature-source mlp_neurons --batch-size "$EXTRACT_BS" \
        --max-len "$MAX_LEN" --seed "$SEED" --out "$raw" "${corrupt_arg[@]}" ;;
    weight|magnitude|random)
      CUDA_VISIBLE_DEVICES=$gpu python scripts/extraction/build_baseline_mask.py \
        --model "$MODEL_LOAD" --mode "$arm" --bits "$WBITS" --mask-fraction "$FRAC" \
        --seed "$SEED" --out "$raw" $([[ -f "$corrupt" ]] && echo --corrupt-model "$corrupt") ;;
    *) echo "unknown arm=$arm" >&2; return 1 ;;
  esac
  python scripts/extraction/convert_tdso_mask.py --in "$raw" --out "$flat"
}

gptq_eval_arm() {
  local task="$1" arm="$2" gpu="$3"
  local seed_suffix=""; [[ "$SEED" != "0" ]] && seed_suffix="_s${SEED}"
  local flat engine
  flat="$(flat_path_for "$task" "$arm" "$seed_suffix")"
  engine="$(engine_for "$task" "$arm")"
  if [[ ! -f "$CKPT_DIR/$engine.pt" || "$FORCE_RECOMPUTE" == "1" ]]; then
    (cd "$ROOT/TACQ" && CUDA_VISIBLE_DEVICES=$gpu python -m gptq.llama "$MODEL_LOAD" "$task" \
      --true-sequential --wbits "$WBITS" --save_in_16bits "$CKPT_DIR/$engine.pt" \
      --no-eval --seed "$SEED" --important_mask "$flat")
  fi
  (cd "$ROOT/TACQ" && CUDA_VISIBLE_DEVICES=$gpu TASK="$task" ENGINE="$engine" SEED="$SEED" \
    run_pretrain_eval_profile same)
}

heavy=(); dumb=()
for arm in "${ARMS[@]}"; do
  case "$arm" in ce|align|mult|align_free|mult_free) heavy+=("$arm") ;; *) dumb+=("$arm") ;; esac
done

log "=== Saliency-source ablation w$WBITS seed=$SEED tasks=${TASKS[*]} arms=${ARMS[*]} ==="
log "Phase 1: build masks (heavy seq gpu0, dumb parallel)"
for task in "${TASKS[@]}"; do
  for arm in "${heavy[@]}"; do log "[mask] heavy $arm $task"; build_mask_flat "$task" "$arm" 0; done
  pids=(); gpu=1
  for arm in "${dumb[@]}"; do
    log "[mask] dumb $arm $task (gpu=$gpu)"
    build_mask_flat "$task" "$arm" "$gpu" & pids+=($!); gpu=$(((gpu % 3) + 1))
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
done

log "Phase 2: GPTQ + eval (round-robin 4 GPUs)"
pids=(); gpu=0
for task in "${TASKS[@]}"; do
  for arm in "${ARMS[@]}"; do
    gptq_eval_arm "$task" "$arm" "$gpu" &
    pids+=($!); gpu=$(((gpu + 1) % 4))
    [[ $((${#pids[@]} % 4)) -eq 0 ]] && { for pid in "${pids[@]}"; do wait "$pid" || true; done; pids=(); }
  done
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
log "=== ablation complete ==="
