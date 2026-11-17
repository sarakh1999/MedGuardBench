#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=grpo_lg7
#SBATCH --clusters=ascend
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/grpo_smoke/slurm_grpo_lg7_%j.log
#
# Resume the Llama-Guard-7B (Llama-2) guided GRPO v3 run. The chain is
# idempotent: merge/mine/v2-select/guided-build are already cached, and
# training auto-resumes from the newest checkpoint (checkpoint-400 at submit),
# then runs checkpoint-select on validation and eval vs the merged SFT policy.
#
#   sbatch Claude/GRPO/run_grpo_llamaguard7b.slurm
set -uo pipefail
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-?}  $(date) ==="
nvidia-smi -L || true
export MGB_MODEL=llamaguard-7b
export MGB_DATA_CONDITION=legacy_v1
export MGB_GRPO_RECIPE=v3
export MGB_LOAD_IN_4BIT=1
export VLLM_GPU_MEMORY_UTILIZATION=0.40
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
exec bash /users/PCS0289/sarakhosravi/grpo_smoke/model_guided_chain_v3.sh grpo-guided-v3 \
  --epochs 1 --per-device-batch 4 --grpo-mini-batch 32 --vllm-gpu-util 0.40 --load-in-4bit
