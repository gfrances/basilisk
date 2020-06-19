#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --partition=infai_1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=3GB

CONFIG=$1


srun --job-name=generate-data-$CONFIG \
     singularity run --bind /infai/blaas/projects/basilisk/benchmarks/:/code/basilisk/benchmarks \
     --bind /infai/blaas/projects/basilisk/pyperplan-benchmarks/:/code/basilisk/pyperplan-benchmarks \
     --bind ../workspace:/workspace \
     ./basilisk.sif /code/basilisk/experiments/run.py --workspace /workspace $CONFIG
