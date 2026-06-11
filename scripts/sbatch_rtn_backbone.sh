#!/bin/bash
# RTN backbone ablation: same ATLAS-T / ATLAS-N / TaCQ masks, but quantizer is
# RTN (--nearest) instead of GPTQ. Shows that the mask choice, not the quantizer,
# drives the gain. Headline tasks only (GSM8k, MMLU_STEM), 2-bit, seed 0.
# Submit: sbatch scripts/sbatch_rtn_backbone.sh
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=rtn_abl
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/rtn_backbone_%j.out
set -euo pipefail
srun --environment=vscode-pytorch --gres=gpu:4 bash -lc '
set -euo pipefail
cd /capstor/scratch/cscs/fnoorala/ATLAS-PTQ
source llama31_venv/bin/activate
set -a; [ -f TACQ/.env ] && source TACQ/.env; set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export HF_HOME=/capstor/scratch/cscs/fnoorala/.cache
export PYTHONUNBUFFERED=1

ROOT="/capstor/scratch/cscs/fnoorala/ATLAS-PTQ"
MODEL_LOAD="unsloth/Meta-Llama-3.1-8B-Instruct"
MODEL_BASE="Meta-Llama-3.1-8B-Instruct"
WBITS=2
CKPT_DIR="$ROOT/tacq_data"
export RESULTS_DIR="$ROOT/tacq_data/results"
export CKPT_DIR
source "$ROOT/scripts/pretrain_eval_helpers.sh"
log() { echo "[$(date +%H:%M:%S)] $*"; }

rtn_eval() {
  local task="$1" arm="$2" mask="$3" gpu="$4"
  local engine="${MODEL_BASE}_rtn_${arm}_${task}_${WBITS}bit_quantized_model"
  log "RTN $task arm=$arm gpu=$gpu"
  (cd "$ROOT/TACQ" && CUDA_VISIBLE_DEVICES=$gpu python -m gptq.llama "$MODEL_LOAD" "$task" \
    --nearest --wbits $WBITS --save_in_16bits "$CKPT_DIR/$engine.pt" \
    --no-eval --seed 0 --important_mask "$mask")
  (cd "$ROOT/TACQ" && CUDA_VISIBLE_DEVICES=$gpu TASK="$task" ENGINE="$engine" SEED=0 \
    run_pretrain_eval_profile same)
}

# Use existing masks from the ablation (already built, no extraction needed)
for task in GSM8k MMLU_STEM; do
  tk=$(echo $task | tr "[:upper:]" "[:lower:]")
  pids=(); gpu=0
  for arm_mask in \
    "ce $CKPT_DIR/ablsrc_ce_${tk}_2bit_maskflat.pt" \
    "mult $CKPT_DIR/ablsrc_mult_${tk}_2bit_maskflat.pt" \
    "mult_free $CKPT_DIR/ablsrc_mult_free_${tk}_2bit_maskflat.pt" \
    "random $CKPT_DIR/ablsrc_random_${tk}_2bit_maskflat.pt"; do
    arm="${arm_mask%% *}"
    mask="${arm_mask#* }"
    if [[ ! -f "$mask" ]]; then log "SKIP $arm $task (no mask $mask)"; continue; fi
    rtn_eval "$task" "$arm" "$mask" "$gpu" &
    pids+=($!); gpu=$(((gpu + 1) % 4))
  done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
done

# Also run RTN WITHOUT any mask (uniform 2-bit) as the rock-bottom baseline
for task in GSM8k MMLU_STEM; do
  engine="${MODEL_BASE}_rtn_nomask_${task}_${WBITS}bit_quantized_model"
  log "RTN no-mask $task gpu=0"
  (cd "$ROOT/TACQ" && CUDA_VISIBLE_DEVICES=0 python -m gptq.llama "$MODEL_LOAD" "$task" \
    --nearest --wbits $WBITS --save_in_16bits "$CKPT_DIR/$engine.pt" \
    --no-eval --seed 0)
  (cd "$ROOT/TACQ" && CUDA_VISIBLE_DEVICES=0 TASK="$task" ENGINE="$engine" SEED=0 \
    run_pretrain_eval_profile same)
done

log "=== RTN backbone ablation complete ==="
' 2>&1 | tee -a tacq_data/results/rtn_backbone_w2.log
