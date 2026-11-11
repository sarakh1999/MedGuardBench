#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=wg7b_eval
#SBATCH --clusters=ascend
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/Claude/SFT/new_outputs/WildGuard-7B/eval_ascend_%j.log

set -uo pipefail

ROOT=/users/PCS0289/sarakhosravi/Guardrail
SFT_OUT=$ROOT/Claude/SFT/new_outputs/WildGuard-7B
TEST=$ROOT/Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl
PREDS=$SFT_OUT/test_predictions_sft.jsonl

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-?}  time: $(date) ==="
nvidia-smi -L || true

source "$ROOT/Claude/SFT/blind/gpu_env.sh"
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MGB_DATA_CONDITION=legacy_v1
export MGB_MODEL=wildguard-7b

cd "$ROOT"

echo "=== Evaluating WildGuard-7B SFT (final_adapter = best ckpt-900) ==="
python Claude/SFT/blind/eval_sft.py \
    --model wildguard-7b \
    --adapter "$SFT_OUT/final_adapter" \
    --test "$TEST" \
    --out "$PREDS"

echo "=== Computing metrics ==="
python Claude/SFT/compute_metrics.py \
    --pred "$PREDS" \
    --gt "$TEST" \
    --out_dir "$SFT_OUT"

echo "=== Done: $(date) ==="
