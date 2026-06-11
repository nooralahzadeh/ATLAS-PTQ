#!/bin/bash
# Rebuild firm T-DSO v2-mult 2-bit engines at genuine --wbits 2 (the originals were
# 3-bit in disguise). GPTQ-only over existing valid masks => fast, 4-GPU, no OOM.
# Submit: sbatch scripts/sbatch_rebuild_firm_tdso_2bit.sh
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=tdso2b_fix
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/rebuild_firm_tdso_2bit_%j.out
set -euo pipefail
srun --environment=vscode-pytorch --gres=gpu:4 bash -lc '
set -euo pipefail
cd /capstor/scratch/cscs/fnoorala/ATLAS-PTQ
source llama31_venv/bin/activate
set -a; [ -f TACQ/.env ] && source TACQ/.env; set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
export HF_HOME=/capstor/scratch/cscs/fnoorala/.cache
export PYTHONUNBUFFERED=1
bash scripts/rebuild_firm_tdso_2bit.sh \
  2>&1 | tee -a tacq_data/results/downstream_firm_w2_REBUILD.log
'
