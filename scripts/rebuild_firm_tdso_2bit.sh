#!/bin/bash
# Rebuild firm T-DSO v2-mult 2-bit engines that the original pipeline built at the
# WRONG bit-width: gptq.llama defaults to --wbits 3, and the pre-incident T-DSO
# runner never passed --wbits, so every "2bit" T-DSO engine was physically 3-bit
# (file size == 3bit engine; scored ~62% instead of ~28%). Confirmed in
# downstream_firm_w2.log: build block shows "224 layers bits = 3".
#
# The MASKS are fine — extract_tdso_v2_h200.py defaults to --bits 2, and the flat
# masks tacq_data/tdso_v2_*_2bit_task*_mask.pt are 0.35% selections (the saliency
# ablation reused tdso_v2_gsm8k_2bit_task_mask.pt and got the correct 28.7%). So
# we only need to re-run GPTQ at a genuine --wbits 2, then eval. No extraction =>
# no CPU OOM, full 4-GPU round-robin.
set -euo pipefail
ROOT="${ROOT:-/capstor/scratch/cscs/fnoorala/ATLAS-PTQ}"
cd "$ROOT"
MODEL_LOAD="${MODEL_LOAD:-unsloth/Meta-Llama-3.1-8B-Instruct}"
MODEL_BASE="${MODEL_BASE:-Meta-Llama-3.1-8B-Instruct}"
WBITS=2
CKPT_DIR="${CKPT_DIR:-$ROOT/tacq_data}"
export RESULTS_DIR="${RESULTS_DIR:-$ROOT/tacq_data/results}"
export CKPT_DIR
source "$ROOT/scripts/pretrain_eval_helpers.sh"
log() { echo "[$(date +%H:%M:%S)] $*"; }

# Map lowercase mask token -> canonical TASK name used by gptq/eval.
canon_task() {
  case "$1" in
    gsm8k) echo "GSM8k" ;;
    mmlu_stem) echo "MMLU_STEM" ;;
    mmlu_humanities) echo "MMLU_humanities" ;;
    mmlu_social_sciences) echo "MMLU_social_sciences" ;;
    *) echo "$1" ;;
  esac
}

rebuild_one() {
  local flat="$1" gpu="$2"
  local base tok ss seed task engine
  base="$(basename "$flat")"                       # tdso_v2_<tok>_2bit_task[_sN]_mask.pt
  tok="$(sed -E 's/^tdso_v2_(.+)_2bit_task.*/\1/' <<<"$base")"
  ss=""; seed=0
  if [[ "$base" =~ _task_s([12])_mask ]]; then ss="_s${BASH_REMATCH[1]}"; seed="${BASH_REMATCH[1]}"; fi
  task="$(canon_task "$tok")"
  engine="${MODEL_BASE}_tdsoV2mult_${task}_2bit${ss}_quantized_model"

  log "[gpu=$gpu] GPTQ $engine  mask=$base  (--wbits 2, seed=$seed)"
  (cd "$ROOT/TACQ" && CUDA_VISIBLE_DEVICES=$gpu python -m gptq.llama "$MODEL_LOAD" "$task" \
      --true-sequential --wbits "$WBITS" --save_in_16bits "$CKPT_DIR/$engine.pt" \
      --no-eval --seed "$seed" --important_mask "$ROOT/$flat")
  (cd "$ROOT/TACQ" && CUDA_VISIBLE_DEVICES=$gpu TASK="$task" ENGINE="$engine" SEED="$seed" \
      run_pretrain_eval_profile same)
}

mapfile -t MASKS < <(ls -1 tacq_data/tdso_v2_*_2bit_task*_mask.pt 2>/dev/null)
log "=== Rebuild firm T-DSO v2-mult 2-bit: ${#MASKS[@]} engines (genuine --wbits 2) ==="
printf '  %s\n' "${MASKS[@]}"

pids=(); gpu=0
for flat in "${MASKS[@]}"; do
  rebuild_one "$flat" "$gpu" &
  pids+=($!); gpu=$(((gpu + 1) % 4))
  [[ $((${#pids[@]} % 4)) -eq 0 ]] && { for pid in "${pids[@]}"; do wait "$pid" || true; done; pids=(); }
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
log "=== rebuild complete ==="
