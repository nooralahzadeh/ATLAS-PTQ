#!/bin/bash
# Paper 2 B1 smoke — Qwen3-32B full DIN-SQL ICE (needs ~80GB+ VRAM; H200 OK).
#
# Submit:
#   sbatch scripts/sbatch_dinsql_b1_smoke_32b_debug.sh
#SBATCH --account=a135
#SBATCH --partition=debug
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=p2_b1_32b
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/paper2_b1_32b_%j.out

set -euo pipefail
ROOT=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ
export MAX_EXAMPLES="${MAX_EXAMPLES:-4}"
export MODEL="${MODEL:-Qwen/Qwen3-32B}"
export PROMPT_PROFILE="${PROMPT_PROFILE:-full}"
export PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
export MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-24576}"
export TRUST_REMOTE_CODE=1
srun --environment=vscode-pytorch bash "$ROOT/scripts/run_dinsql_b1_smoke.sh"
