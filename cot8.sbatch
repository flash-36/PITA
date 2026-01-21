#!/bin/bash
#SBATCH -J CoT8_PITA
#SBATCH -p gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 4-00:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname) ($(date))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python run_parallel.py --config-name=cot8_eval

# To resume: python run_parallel.py --config-name=llama resume_from=/path/to/your/llama/checkpoint
