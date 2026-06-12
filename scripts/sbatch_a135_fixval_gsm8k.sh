#!/bin/bash
# Pipeline-parity validation on the a135 debug partition (NO a0125 hours).
#
# Goal: prove (cheaply) that the ~4pp ce-vs-TaCQ gap is caused by the four
# extraction divergences we found, by rebuilding the GSM8k seed-0 ce mask with
# TaCQ-parity settings and measuring Jaccard against TaCQ's actual mask:
#   1. fp16 gradients   (TaCQ sm16bit selector)   [was bf16]
#   2. batch size 1     (per-sample |grad| sum)   [was 2]
#   3. raw-text tokens  (no chat template)        [was chat-template]
#   4. GPTQ corrupt dW  (checkpoints_llama31)     [was RTN fallback]
# Also builds the proposed boost_rank dict-free mask (lam=0.25) for inspection.
#
# NO GPTQ, NO eval — masks + Jaccard only. Fits the 1h30 debug limit.
# Submit: sbatch scripts/sbatch_a135_fixval_gsm8k.sh
#SBATCH --account=a135
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=fixval_gsm8k
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/fixval_gsm8k_%j.out
set -euo pipefail
srun --environment=vscode-pytorch --gres=gpu:1 bash -lc '
set -euo pipefail
cd /capstor/scratch/cscs/fnoorala/ATLAS-PTQ
source llama31_venv/bin/activate
set -a; [ -f TACQ/.env ] && source TACQ/.env; set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export HF_HOME=/capstor/scratch/cscs/fnoorala/.cache
export PYTHONUNBUFFERED=1

PAIRS=data/contrastive/gsm8k_contrastive_tacq.jsonl
CORRUPT="tacq_data/checkpoints_llama31/Meta-Llama-3.1-8B-Instruct+0+GSM8k+2bit+quantized_model.pt"
TACQ_MASK="tacq_data/importances/Meta-Llama-3.1-8B-Instruct+0+GSM8k+sample_abs_weight_prod_contrastive_sm16bit+2bit+implementation_test/important_mask_q2+top_p_sparse+.0035.pt"
OUT_CE=tacq_data/fixv1_llama31_ce_gsm8k_2bit_maskraw.pt
OUT_BR=tacq_data/fixv1_llama31_boostrank_free_gsm8k_2bit_maskraw.pt

echo "=== [1/4] TaCQ-parity ce extraction (fp16, bs=1, raw text, GPTQ dW) ==="
python scripts/extraction/extract_tdso_v2_h200.py \
  --pairs "$PAIRS" --model unsloth/Meta-Llama-3.1-8B-Instruct \
  --combine ce --bits 2 --mask-fraction 0.0035 --mask-budget global \
  --batch-size 1 --max-len 4096 --seed 0 \
  --model-dtype float16 --raw-text \
  --corrupt-model "$CORRUPT" \
  --out "$OUT_CE"

echo "=== [2/4] Jaccard: fixed ce vs actual TaCQ mask (success: >=0.8) ==="
python scripts/analysis/mask_jaccard.py --a "$TACQ_MASK" --b "$OUT_CE" --per-layer

echo "=== [3/4] boost_rank dict-free extraction (lam=0.25, same parity fixes) ==="
python scripts/extraction/extract_dictfree_saliency.py \
  --pairs "$PAIRS" --model unsloth/Meta-Llama-3.1-8B-Instruct \
  --combine boost_rank --lam 0.25 --feature-source mlp_neurons \
  --bits 2 --mask-fraction 0.0035 \
  --batch-size 1 --max-len 4096 --seed 0 \
  --model-dtype float16 --raw-text \
  --corrupt-model "$CORRUPT" \
  --out "$OUT_BR"

echo "=== [4/4] Jaccard: boost_rank vs TaCQ mask (expect high, < ce-fixed) ==="
python scripts/analysis/mask_jaccard.py --a "$TACQ_MASK" --b "$OUT_BR"

echo "=== fixval complete ==="
'
