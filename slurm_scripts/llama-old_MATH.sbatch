#!/bin/bash
#SBATCH -J LlamaOld_MATH
#SBATCH -p gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-task=8
#SBATCH -t 4-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

CONFIG=llama-old_MATH
STATE_FILE="run_state/${CONFIG}.path"

handle_timeout() {
  echo "Timeout approaching, requeuing job $SLURM_JOB_ID..."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap handle_timeout USR1

echo "Job $SLURM_JOB_ID on $(hostname) ($(date))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [ -f "$STATE_FILE" ]; then
  RESUME_FROM=$(cat "$STATE_FILE")
  echo "Auto-resuming from $RESUME_FROM"
  python run_parallel.py --config-name=$CONFIG resume_from="$RESUME_FROM" &
else
  echo "Starting fresh run"
  python run_parallel.py --config-name=$CONFIG &
fi
wait $!
