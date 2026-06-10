#!/bin/bash
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=circuit_scale
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/circuit_scale_%j.out

set -euo pipefail
srun --environment=vscode-pytorch bash -lc '
cd /capstor/scratch/cscs/fnoorala/ATLAS-PTQ
TESTING=${TESTING:-0} bash scripts/run_circuit_scale_ablation.sh
'
