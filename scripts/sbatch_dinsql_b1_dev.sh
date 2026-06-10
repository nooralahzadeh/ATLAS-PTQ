#!/bin/bash
# Paper 2 B1 baseline — full Spider dev, 8B lite+raw, 4-GPU parallel.
#
# Submit:
#   sbatch scripts/sbatch_dinsql_b1_dev.sh
#
# Smoke subset (e.g. 50 examples, 1 GPU):
#   DEV_TOTAL=50 NUM_GPUS=1 sbatch scripts/sbatch_dinsql_b1_dev.sh
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=p2_b1dev
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/paper2_b1_dev_%j.out

set -euo pipefail
ROOT=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ
export DEV_TOTAL="${DEV_TOTAL:-1034}"
export NUM_GPUS="${NUM_GPUS:-4}"
export MODEL="${MODEL:-unsloth/Meta-Llama-3.1-8B-Instruct}"
export PROMPT_PROFILE="${PROMPT_PROFILE:-lite}"
export PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
export ICE_SHOTS="${ICE_SHOTS:-2}"
export MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
srun --environment=vscode-pytorch bash "$ROOT/scripts/run_dinsql_b1_dev.sh"
