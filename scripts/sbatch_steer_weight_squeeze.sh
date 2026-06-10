#!/bin/bash
#SBATCH --account=a0125
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --job-name=steer_squeeze
#SBATCH --output=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ/tacq_data/logs/steer_squeeze_%j.out

ROOT=/capstor/scratch/cscs/fnoorala/ATLAS-PTQ
srun --environment=vscode-pytorch bash "$ROOT/scripts/run_steer_weight_squeeze.sh"
