#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=dpo_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --partition=gpudebug
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/Claude/DPO/logs/slurm_smoke_%j.log

set -euo pipefail

ROOT=/users/PCS0289/sarakhosravi/Guardrail
DPO="$ROOT/Claude/DPO"
POLICY_INIT="$ROOT/Claude/SFT/new_outputs/Qwen3-4B-Instruct/checkpoint-950-merged"
PAIRS="$DPO/data/dpo_partial_miss_audited.jsonl"
OUT="$DPO/outputs/smoke_test_4b"

echo "=== Smoke Test on $(hostname) (Job ${SLURM_JOB_ID:-?}) ==="
source "$ROOT/Claude/SFT/blind/gpu_env.sh"
mkdir -p "$OUT"

python "$DPO/train_dpo.py" \
  --pairs "$PAIRS" \
  --policy-init "$POLICY_INIT" \
  --out "$OUT" \
  --max-steps 5 \
  --batch-size 2 \
  --grad-accum 2 \
  --beta 0.1 \
  --rpo-alpha 0.0 \
  --lr 5e-6 \
  --mask-audit-only

echo "=== Smoke test completed successfully ==="
