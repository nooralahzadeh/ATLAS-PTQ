#!/bin/bash
# Paper 2 B1 smoke — 8B lite+raw (default) on a135 debug.
#
# Submit:
#   sbatch scripts/sbatch_dinsql_b1_smoke_debug.sh
#
# Full DIN-SQL ICE on 8B (may still truncate):
#   PROMPT_PROFILE=full sbatch scripts/sbatch_dinsql_b1_smoke_debug.sh
#SBATCH --account=a135
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=p2_b1lite
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/paper2_b1_smoke_%j.out

set -euo pipefail
ROOT=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ
export MAX_EXAMPLES="${MAX_EXAMPLES:-4}"
export PROMPT_PROFILE="${PROMPT_PROFILE:-lite}"
export PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
export ICE_SHOTS="${ICE_SHOTS:-2}"
srun --environment=vscode-pytorch bash "$ROOT/scripts/run_dinsql_b1_smoke.sh"
