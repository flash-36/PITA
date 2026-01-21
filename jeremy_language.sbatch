#!/bin/bash
#SBATCH -J Phi_PITA
#SBATCH -p gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH -t 4-00:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname) ($(date))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# (Optional) activate your env/modules here
# module load cuda/12.1
# source ~/.bashrc && conda activate pita

python run_parallel.py --config-name=jeremy_language #resume_from=