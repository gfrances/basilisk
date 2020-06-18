#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --partition=infai_1
#

CONFIG=$1


srun --job-name=generate-data-$CONFIG \
     singularity run --bind /infai/blaas/projects/basilisk/benchmarks/:/code/basilisk/benchmarks \
     --bind .:/workspace ./basilisk.sif /code/basilisk/experiments/run.py \
     --workspace /workspace $CONFIG
