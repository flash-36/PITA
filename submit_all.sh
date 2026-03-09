#!/bin/bash
# Submit all experiments as a Slurm dependency chain.
#
# Each job auto-requeues on timeout (up to 5x per MaxBatchRequeue)
# and auto-resumes from checkpoint via run_state/{config}.path.
# The afterany chain waits for full completion across requeues.
#
#   bash submit_all.sh           # submit all 25 experiments
#   bash submit_all.sh 8         # start from experiment #8
#   bash submit_all.sh 8 12      # submit experiments #8 through #12
#
# Fresh start: rm -rf run_state/ && bash submit_all.sh

set -euo pipefail

SCRIPTS=(
  # -- 1-5: Qwen3-sm (1.7B ref + 0.6B cls, 2xA100) --
  slurm_scripts/qwen3-sm_GSM8K.sbatch
  slurm_scripts/qwen3-sm_AIME22to24.sbatch
  slurm_scripts/qwen3-sm_MATH.sbatch
  slurm_scripts/qwen3-sm_IMDBGen.sbatch
  slurm_scripts/qwen3-sm_TLDR.sbatch
  # -- 6-10: Gemma (4B ref + 1B cls, 3xA100) --
  slurm_scripts/gemma_GSM8K.sbatch
  slurm_scripts/gemma_AIME22to24.sbatch
  slurm_scripts/gemma_MATH.sbatch
  slurm_scripts/gemma_IMDBGen.sbatch
  slurm_scripts/gemma_TLDR.sbatch
  # -- 11-15: Qwen3 (8B ref + 1.7B cls, 3xA100) --
  slurm_scripts/qwen3_GSM8K.sbatch
  slurm_scripts/qwen3_AIME22to24.sbatch
  slurm_scripts/qwen3_MATH.sbatch
  slurm_scripts/qwen3_IMDBGen.sbatch
  slurm_scripts/qwen3_TLDR.sbatch
  # -- 16-20: Phi (14B ref + 3.8B cls, 3xA100) --
  slurm_scripts/phi_GSM8K.sbatch
  slurm_scripts/phi_AIME22to24.sbatch
  slurm_scripts/phi_MATH.sbatch
  slurm_scripts/phi_IMDBGen.sbatch
  slurm_scripts/phi_TLDR.sbatch
  # -- 21-25: Llama-old (8B ref + 1B cls, 3xA100) --
  slurm_scripts/llama-old_GSM8K.sbatch
  slurm_scripts/llama-old_AIME22to24.sbatch
  slurm_scripts/llama-old_MATH.sbatch
  slurm_scripts/llama-old_IMDBGen.sbatch
  slurm_scripts/llama-old_TLDR.sbatch
)

START=${1:-1}
END=${2:-${#SCRIPTS[@]}}
PREV_JOB=""
SUBMITTED=0

echo "Submitting experiments #${START}-${END} of ${#SCRIPTS[@]} total"
echo "------------------------------------------------------"

for i in "${!SCRIPTS[@]}"; do
  NUM=$((i + 1))
  [[ $NUM -lt $START ]] && continue
  [[ $NUM -gt $END ]]   && break

  SCRIPT="${SCRIPTS[$i]}"

  if [ -z "$PREV_JOB" ]; then
    RESULT=$(sbatch "$SCRIPT")
  else
    RESULT=$(sbatch --dependency=afterany:$PREV_JOB "$SCRIPT")
  fi

  JOB_ID=$(echo "$RESULT" | awk '{print $4}')
  PREV_JOB=$JOB_ID
  SUBMITTED=$((SUBMITTED + 1))
  printf "  %2d. %-42s  ->  %s\n" "$NUM" "$SCRIPT" "$JOB_ID"
done

echo "------------------------------------------------------"
echo "Submitted $SUBMITTED jobs. Monitor with: squeue -u \$USER"
