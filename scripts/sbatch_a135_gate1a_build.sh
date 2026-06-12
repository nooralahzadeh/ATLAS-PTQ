#!/bin/bash
# Gate 1a/window A (a135 debug): build boost_rank engines for the 3 MMLU splits,
# seed 0, TaCQ-parity extraction (fp16, bs=1, raw text, GPTQ corrupt dW).
# 3 tasks in parallel on GPUs 0-2: extract (~20min) -> preflight -> GPTQ (~40min).
# Evals happen in window B (sbatch_a135_gate1a_eval.sh, submitted afterok).
# Submit: sbatch scripts/sbatch_a135_gate1a_build.sh
#SBATCH --account=a135
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=gate1a_build
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/gate1a_build_%j.out
set -euo pipefail
srun --environment=vscode-pytorch --gres=gpu:4 bash -lc '
set -euo pipefail
cd /capstor/scratch/cscs/fnoorala/ATLAS-PTQ
source llama31_venv/bin/activate
set -a; [ -f TACQ/.env ] && source TACQ/.env; set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export HF_HOME=/capstor/scratch/cscs/fnoorala/.cache
export PYTHONUNBUFFERED=1

declare -A PAIRS=(
  [MMLU_STEM]=data/contrastive/mmlu_stem_contrastive_test75.jsonl
  [MMLU_humanities]=data/contrastive/mmlu_humanities_contrastive_test75.jsonl
  [MMLU_social_sciences]=data/contrastive/mmlu_social_sciences_contrastive_test75.jsonl
)

build_one() {
  local task="$1" gpu="$2"
  local corrupt="tacq_data/checkpoints_llama31/Meta-Llama-3.1-8B-Instruct+0+${task}+2bit+quantized_model.pt"
  local raw="tacq_data/fixv2_llama31_boostrank_free_${task,,}_2bit_maskraw.pt"
  local flat="tacq_data/fixv2_llama31_boostrank_free_${task,,}_2bit_maskflat.pt"
  local engine="Meta-Llama-3.1-8B-Instruct_fixv2_boostrank_free_${task}_2bit_quantized_model"

  # Guard 1: corrupt checkpoint must exist (no silent RTN fallback, ever again)
  [[ -f "$corrupt" ]] || { echo "FATAL [$task]: corrupt model missing: $corrupt"; exit 1; }
  [[ -f "${PAIRS[$task]}" ]] || { echo "FATAL [$task]: pairs missing"; exit 1; }

  CUDA_VISIBLE_DEVICES=$gpu python scripts/extraction/extract_dictfree_saliency.py \
    --pairs "${PAIRS[$task]}" --model unsloth/Meta-Llama-3.1-8B-Instruct \
    --combine boost_rank --lam 0.25 --feature-source mlp_neurons \
    --bits 2 --mask-fraction 0.0035 --batch-size 1 --max-len 4096 --seed 0 \
    --model-dtype float16 --raw-text --corrupt-model "$corrupt" --out "$raw"

  # Guard 2: meta must say corrupt_model + sane kept fraction before any GPTQ
  python - "$raw" <<EOF
import sys, torch
m = torch.load(sys.argv[1], map_location="cpu")["meta"]
assert m["delta_w"] == "corrupt_model", f"delta_w={m['delta_w']} (RTN fallback!)"
frac = m["kept_params"] / m["total_params"]
assert 0.003 <= frac <= 0.004, f"kept fraction {frac:.5f} out of band"
print(f"[preflight OK] {sys.argv[1]}: delta_w=corrupt_model kept={frac:.5f}")
EOF

  python scripts/extraction/convert_tdso_mask.py --in "$raw" --out "$flat"
  (cd TACQ && CUDA_VISIBLE_DEVICES=$gpu python -m gptq.llama \
    unsloth/Meta-Llama-3.1-8B-Instruct "$task" \
    --true-sequential --wbits 2 \
    --save_in_16bits "../tacq_data/$engine.pt" \
    --no-eval --seed 0 --important_mask "../$flat")
  echo "[done $task] engine -> tacq_data/$engine.pt"
}

pids=(); gpu=0
for task in MMLU_STEM MMLU_humanities MMLU_social_sciences; do
  build_one "$task" "$gpu" & pids+=($!); gpu=$((gpu + 1))
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit $fail
'
