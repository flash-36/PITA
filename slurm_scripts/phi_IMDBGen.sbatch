#!/bin/bash
#SBATCH -J Phi_IMDBGen
#SBATCH -p gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=8
#SBATCH -t 4-00:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

CONFIG=phi_IMDBGen
STATE_FILE="run_state/${CONFIG}.path"

set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname) ($(date))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [ -f "$STATE_FILE" ]; then
  RESUME_FROM=$(cat "$STATE_FILE")
  echo "Auto-resuming from $RESUME_FROM"
  python run_parallel.py --config-name=$CONFIG resume_from="$RESUME_FROM"
else
  echo "Starting fresh run"
  python run_parallel.py --config-name=$CONFIG
fi
