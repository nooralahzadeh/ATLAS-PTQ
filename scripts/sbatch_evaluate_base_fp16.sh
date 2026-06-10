#!/bin/bash
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --job-name=fp16_baseline
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/fp16_baseline_%j.out

ROOT=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ
srun --environment=vscode-pytorch bash "$ROOT/scripts/run_evaluate_base_fp16.sh"
