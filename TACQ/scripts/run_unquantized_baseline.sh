#!/bin/bash
# Run unquantized Spider baseline (paper target: 67.6% exec accuracy).
cd "$(dirname "$0")/.." || exit 1
source tacq_venv/bin/activate

checkpoints_dir="/home/ubuntu/tacq_data/checkpoints"
results_dir="/home/ubuntu/tacq_data/results"
eval_device="${eval_device:-0}"

model_name="Meta-Llama-3-8B-Instruct"
tables="datasets_directory/Spider/data/spider/tables.json"
dataset_path="datasets_directory/Spider/data/spider/dev.json"
dev_gold_path="datasets_directory/Spider/data/spider/dev_gold.sql"
db_dir="datasets_directory/Spider/database"
output_savedir="$results_dir/Spider/unquantized_${model_name}"
mkdir -p "$output_savedir"

TESTING_FLAG=""
if [ "${TESTING:-0}" = "1" ]; then
    TESTING_FLAG="--testing"
fi

set -e
CUDA_VISIBLE_DEVICES=$eval_device python -m datasets_directory.Spider.Spider_eval \
    --model "$model_name" \
    --input "$dataset_path" \
    --tables "$tables" \
    --predictions_filename predictions.txt \
    --output "${output_savedir}/debug.txt" \
    --output_savedir "$output_savedir" \
    --addition_dir "$checkpoints_dir" \
    $TESTING_FLAG

export NLTK_DATA="datasets_directory/Spider/third_party/nltk_data"
python datasets_directory/Spider/third_party/test-suite-sql-eval/evaluation.py \
    --gold "$dev_gold_path" \
    --pred "${output_savedir}/predictions.txt" \
    --db "$db_dir" \
    --table "$tables" \
    --etype exec \
    2>&1 | tee "${output_savedir}/eval_exec.log"

echo "Unquantized baseline results: ${output_savedir}/eval_exec.log"
