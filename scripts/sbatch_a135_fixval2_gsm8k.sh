#!/bin/bash
# fixval window 2 (a135 debug): rebuild boost_rank mask with the fp32-contrib
# overflow fix (fp16 run had align=-inf -> nan grads), then GPTQ the engine.
# Eval happens in a later window (GSM8k eval alone is ~2.5h, exceeds debug).
# Submit: sbatch scripts/sbatch_a135_fixval2_gsm8k.sh
#SBATCH --account=a135
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=fixval2_gsm8k
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/fixval2_gsm8k_%j.out
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
OUT_BR=tacq_data/fixv2_llama31_boostrank_free_gsm8k_2bit_maskraw.pt
FLAT_BR=tacq_data/fixv2_llama31_boostrank_free_gsm8k_2bit_maskflat.pt
ENGINE=Meta-Llama-3.1-8B-Instruct_fixv2_boostrank_free_GSM8k_2bit_quantized_model

echo "=== [1/3] boost_rank dict-free extraction (fp32 contrib fix) ==="
python scripts/extraction/extract_dictfree_saliency.py \
  --pairs "$PAIRS" --model unsloth/Meta-Llama-3.1-8B-Instruct \
  --combine boost_rank --lam 0.25 --feature-source mlp_neurons \
  --bits 2 --mask-fraction 0.0035 \
  --batch-size 1 --max-len 4096 --seed 0 \
  --model-dtype float16 --raw-text \
  --corrupt-model "$CORRUPT" \
  --out "$OUT_BR"

echo "=== [2/3] Jaccard vs TaCQ mask + vs fixed-ce mask ==="
python scripts/analysis/mask_jaccard.py --a "$TACQ_MASK" --b "$OUT_BR"
python scripts/analysis/mask_jaccard.py \
  --a tacq_data/fixv1_llama31_ce_gsm8k_2bit_maskraw.pt --b "$OUT_BR"

echo "=== [3/3] GPTQ engine (boost_rank mask) ==="
python scripts/extraction/convert_tdso_mask.py --in "$OUT_BR" --out "$FLAT_BR"
(cd TACQ && python -m gptq.llama unsloth/Meta-Llama-3.1-8B-Instruct GSM8k \
  --true-sequential --wbits 2 \
  --save_in_16bits "../tacq_data/$ENGINE.pt" \
  --no-eval --seed 0 --important_mask "../$FLAT_BR")

echo "=== fixval2 complete; engine at tacq_data/$ENGINE.pt ==="
'
