#!/bin/bash
# Spider-only TaCQ pipeline for single L4 GPU (23GB VRAM).
# Usage:
#   TESTING=1 source scripts/examples/evaluate_llama3_8b_spider_l4.sh   # smoke test (3 cal points, q2 only)
#   source scripts/examples/evaluate_llama3_8b_spider_l4.sh               # full run (q2 + q3)

echo "Script Starting..."
cd "$(dirname "$0")/../.." || exit 1
source tacq_venv/bin/activate
set -euo pipefail

if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -z "${HUGGINGFACE_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Set HUGGINGFACE_TOKEN in TACQ/.env before running (Llama-3 is gated)."
    exit 1
fi

bash scripts/check_prerequisites.sh || true
set -x

importances_dir="/home/ubuntu/tacq_data/importances"
results_dir="/home/ubuntu/tacq_data/results"
checkpoints_dir="/home/ubuntu/tacq_data/checkpoints"
device="0"
eval_device="0"

model_name="Meta-Llama-3-8B-Instruct"
loadstring="meta-llama"
datasets=("Spider")
serial_numbers=(0)
selector_types=("sample_abs_weight_prod_contrastive_sm16bit")
ranking_types=("top_p_sparse")
ratios=(".0035")

# Smoke test: 3 calibration points, q2 only. Full run: all bit widths.
if [ "${TESTING:-0}" = "1" ]; then
    quantization_types=("q2")
    TESTING_FLAG="--testing"
    SPIDER_EVAL_TESTING="--testing"
else
    quantization_types=("q2" "q3")
    TESTING_FLAG=""
    SPIDER_EVAL_TESTING=""
fi

# Optional memory mitigations (set before sourcing if needed):
#   N_CALIB_POINTS=16  MAX_LENGTH=1024
N_CALIB_POINTS="${N_CALIB_POINTS:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
EXTRA_IMPORTANCE_ARGS=""
if [ -n "$N_CALIB_POINTS" ] && [ "$N_CALIB_POINTS" != "128" ]; then
    EXTRA_IMPORTANCE_ARGS="$EXTRA_IMPORTANCE_ARGS --n_calibration_points $N_CALIB_POINTS"
fi
if [ -n "$MAX_LENGTH" ] && [ "$MAX_LENGTH" != "2048" ]; then
    EXTRA_IMPORTANCE_ARGS="$EXTRA_IMPORTANCE_ARGS --max_length $MAX_LENGTH"
fi

declare -a quantized_model_names

# Phase 1: gradient capture (corrupt model + measure_importances)
for serial_number in "${serial_numbers[@]}"
do
    for cur_dataset in "${datasets[@]}"
    do
        dataset=$cur_dataset
        calibration_dataset=$cur_dataset
        for ratio in "${ratios[@]}"
        do
            for ranking_type in "${ranking_types[@]}"
            do
                for quantization_type in "${quantization_types[@]}"
                do
                    wbits=${quantization_type%%_*}
                    wbits=${wbits#q}
                    echo "wbits: $wbits"
                    for selector_type in "${selector_types[@]}"
                    do
                        corrupt_model_name="${model_name}+${serial_number}+${dataset}+${wbits}bit+quantized_model"
                        if [ ! -f "$checkpoints_dir/${corrupt_model_name}.pt" ]; then
                            echo -e "\n\nRunning gptq corrupt model: ${corrupt_model_name}"
                            CUDA_VISIBLE_DEVICES=$device python -m gptq.llama \
                                $loadstring/${model_name} \
                                $dataset \
                                --true-sequential \
                                --save_in_16bits $checkpoints_dir/${corrupt_model_name}.pt \
                                --wbits $wbits \
                                --seed $serial_number \
                                --no-eval
                        fi

                        run_name="${model_name}+${serial_number}+${dataset}+${selector_type}+${wbits}bit+implementation_test"
                        if [ -f "$importances_dir/$run_name/importances.pt" ] && [ "${FORCE_RECOMPUTE:-0}" != "1" ]; then
                            echo "Importances already exist at $importances_dir/$run_name/importances.pt, skipping"
                        else
                            echo -e "\n\nRunning measure_importances: ${run_name}"
                            CUDA_VISIBLE_DEVICES=$device python -m measure_importances \
                                --model $model_name \
                                --corrupt_model $corrupt_model_name \
                                --dataset $dataset \
                                --run_name $run_name \
                                --checkpoints_dir $checkpoints_dir \
                                --results_dir $importances_dir \
                                --selector_type $selector_type \
                                --serial_number $serial_number \
                                --save_full_gradients \
                                --save_importances_pt_path $importances_dir/$run_name/importances.pt \
                                --override_args_yaml \
                                $TESTING_FLAG \
                                $EXTRA_IMPORTANCE_ARGS
                        fi

                        rm -rf $checkpoints_dir/${corrupt_model_name}.pt
                    done
                done
            done
        done
    done
done

# Phase 2: quantize + Spider eval
for serial_number in "${serial_numbers[@]}"
do
    for cur_dataset in "${datasets[@]}"
    do
        dataset=$cur_dataset
        calibration_dataset=$cur_dataset
        for ratio in "${ratios[@]}"
        do
            for ranking_type in "${ranking_types[@]}"
            do
                for quantization_type in "${quantization_types[@]}"
                do
                    wbits=${quantization_type%%_*}
                    wbits=${wbits#q}
                    echo "wbits: $wbits"
                    for selector_type in "${selector_types[@]}"
                    do
                        run_name="${model_name}+${serial_number}+${dataset}+${selector_type}+${wbits}bit+implementation_test"
                        quant_identifier="${quantization_type}+${ranking_type}+${ratio}"

                        echo -e "\n\nRunning make_quantization_configs: ${run_name}"
                        CUDA_VISIBLE_DEVICES=$eval_device python -m make_quantization_configs \
                            --run_name $run_name \
                            --checkpoints_dir $checkpoints_dir \
                            --results_dir $importances_dir \
                            --serial_number $serial_number \
                            --importances_pt_path $importances_dir/$run_name/importances.pt \
                            --mask_save_path $importances_dir/$run_name/important_mask_${quant_identifier}.pt \
                            --model $model_name \
                            --quantization_type $quantization_type \
                            --ranking_type $ranking_type \
                            --configs_save_path $importances_dir/$run_name/quantization_configs_${quant_identifier}.yaml \
                            --mask_fraction $ratio \
                            --proportional_total_params \
                            --force_recompute

                        quantized_model_name="${run_name}+gptq_on_${dataset}+${quant_identifier}+quantized_model"
                        echo -e "\n\nRunning final gptq: ${quantized_model_name}"
                        CUDA_VISIBLE_DEVICES=$eval_device python -m gptq.llama \
                            $loadstring/${model_name} \
                            $dataset \
                            --true-sequential \
                            --fine-wbits-yaml $importances_dir/$run_name/quantization_configs_${quant_identifier}.yaml \
                            --save_in_16bits $checkpoints_dir/${quantized_model_name}.pt \
                            --no-eval \
                            --seed $serial_number \
                            --important_mask $importances_dir/$run_name/important_mask_${quant_identifier}.pt

                        if [ "$calibration_dataset" = "Spider" ]; then
                            tables="datasets_directory/Spider/data/spider/tables.json"
                            dataset_path="datasets_directory/Spider/data/spider/dev.json"
                            dev_gold_path="datasets_directory/Spider/data/spider/dev_gold.sql"
                            db_dir="datasets_directory/Spider/database"
                            output_savedir="$results_dir/Spider/${quantized_model_name}"
                            mkdir -p "$output_savedir"

                            CUDA_VISIBLE_DEVICES=$eval_device python -m datasets_directory.Spider.Spider_eval \
                                --model ${quantized_model_name} \
                                --input $dataset_path \
                                --tables $tables \
                                --predictions_filename predictions.txt \
                                --output "${output_savedir}/debug.txt" \
                                --output_savedir $output_savedir \
                                --addition_dir $checkpoints_dir \
                                $SPIDER_EVAL_TESTING

                            export NLTK_DATA="datasets_directory/Spider/third_party/nltk_data"
                            python datasets_directory/Spider/third_party/test-suite-sql-eval/evaluation.py \
                                --gold $dev_gold_path \
                                --pred ${output_savedir}/predictions.txt \
                                --db $db_dir \
                                --table $tables \
                                --etype exec \
                                2>&1 | tee "${output_savedir}/eval_exec.log"
                        fi

                        quantized_model_names+=("$quantized_model_name")
                        rm -rf $checkpoints_dir/${quantized_model_name}.pt
                        rm -rf $importances_dir/$run_name/important_mask_${quant_identifier}.pt
                    done
                done
            done
        done
    done
done

echo "Quantized model names: ${quantized_model_names[@]}"
echo "Results directory: $results_dir/Spider/"
