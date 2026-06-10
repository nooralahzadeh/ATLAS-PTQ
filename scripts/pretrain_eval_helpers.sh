#!/bin/bash
# Downstream eval helpers for the Llama-3.1 task-conditioned pipeline.
#
# RECONSTRUCTED after the 2026-06-10 scratch incident. Every primitive here is
# pinned to *surviving* code/artifacts, so fidelity is high:
#   - GSM8k  : TACQ/datasets_directory/GSM8k/GSM8k_eval.py -> writes scores.txt
#              "Accuracy: <f>." ; metric=accuracy
#   - MMLU_* : TACQ/datasets_directory/MMLU/MMLU_eval.py  -> writes
#              results_<engine>/0_evaluation_results.json {"Average accuracy": f}
#              eval rows = test 75-100% (--eval_start_p 0.75) ; metric=avg_acc
#   - Spider : datasets_directory.Spider.Spider_eval + test-suite-sql-eval ; metric=exec
# RESULT line format matches the surviving logs:
#   RESULT calib=<T> eval=<T> metric=<M> acc=<f> engine=<engine>
#
# Contract (set by callers, run from the TACQ/ directory with the venv active):
#   TASK        e.g. GSM8k | MMLU_STEM | MMLU_humanities | MMLU_social_sciences | Spider
#   ENGINE      quantized-model basename in $CKPT_DIR (…_quantized_model)
#   SEED        serial_number
#   CKPT_DIR    checkpoints dir (default: <repo>/tacq_data)
#   RESULTS_DIR eval output root (default: <repo>/tacq_data/results)

pe_ts() { date +%H:%M:%S; }

# run_pretrain_eval_profile <profile>   (profile is informational; "same" => calib task == eval task)
run_pretrain_eval_profile() {
  local profile="${1:-same}"
  local task="${TASK:?TASK required}" engine="${ENGINE:?ENGINE required}" seed="${SEED:-0}"
  local ckpt="${CKPT_DIR:-$PWD/tacq_data}" results="${RESULTS_DIR:-$PWD/tacq_data/results}"
  local metric acc

  echo "[$(pe_ts)] eval $task profile=$profile engine=$engine"

  case "$task" in
    MMLU_*|MMLU)
      local save_dir="$results/$task"
      python3 -m datasets_directory.MMLU.MMLU_eval \
        --engine "$engine" \
        --ntrain 5 \
        --data_dir "datasets_directory/MMLU/data" \
        --save_dir "$save_dir" \
        --addition_dir "$ckpt" \
        --device cuda \
        --MMLU_split "$task" \
        --serial_number "$seed" \
        --eval_start_p 0.75 \
        --eval_end_p 1.00
      metric=avg_acc
      acc=$(python3 -c "import json,sys; d=json.load(open('$save_dir/results_$engine/0_evaluation_results.json')); print(d['Average accuracy'])")
      ;;
    GSM8k|GSM8K)
      local out_dir="$results/GSM8k/$engine"
      python3 -m datasets_directory.GSM8k.GSM8k_eval \
        --model_name_or_path "$engine" \
        --output_dir "$out_dir" \
        --data_root "datasets_directory/GSM8k/data" \
        --seed "$seed" \
        --checkpoints_dir "$ckpt"
      metric=accuracy
      acc=$(grep -oE "Accuracy: [0-9.]+" "$out_dir/scores.txt" | tail -1 | grep -oE "[0-9.]+")
      ;;
    Spider)
      local tables="datasets_directory/Spider/data/spider/tables.json"
      local dev_json="datasets_directory/Spider/data/spider/dev.json"
      local dev_gold="datasets_directory/Spider/data/spider/dev_gold.sql"
      local db_dir="datasets_directory/Spider/database"
      local out_dir="$results/Spider/$engine"
      python3 -m datasets_directory.Spider.Spider_eval \
        --model "$engine" \
        --input "$dev_json" \
        --tables "$tables" \
        --predictions_filename predictions.txt \
        --output "$out_dir/debug.txt" \
        --output_savedir "$out_dir"
      export NLTK_DATA="datasets_directory/Spider/third_party/nltk_data"
      acc=$(python datasets_directory/Spider/third_party/test-suite-sql-eval/evaluation.py \
              --gold "$dev_gold" --pred "$out_dir/predictions.txt" --db "$db_dir" \
              --table "$tables" --etype exec --output_savedir "$out_dir" \
            | grep -oE "execution[^0-9]*[0-9.]+" | grep -oE "[0-9.]+" | tail -1)
      metric=exec
      ;;
    *)
      echo "[pretrain_eval] unknown TASK=$task" >&2
      return 1
      ;;
  esac

  echo "RESULT calib=$task eval=$task metric=$metric acc=$acc engine=$engine"
}
