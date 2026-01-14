#!/bin/bash
#SBATCH -J Pilot_Test
#SBATCH -p gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=8
#SBATCH -t 0-04:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname) ($(date))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Pilot test: phi, llama, gemma on GSM8K with minimal settings
python run_parallel.py --config-name=pilot_test #resume_from=/mnt/shared-scratch/Shakkottai_S/ujwald36/repos/PITA/outputs/2026-01-12/15-04-29
