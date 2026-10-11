#!/usr/bin/env bash
set -euo pipefail

ROOT=/users/PCS0289/sarakhosravi/Guardrail
DPO="$ROOT/Claude/DPO"
MODEL_KEY="qwen3-4b"
SFT_RUN="Qwen3-4B-Instruct"
POLICY_INIT="$ROOT/Claude/SFT/new_outputs/$SFT_RUN/checkpoint-950-merged"
PAIRS="$DPO/data/dpo_partial_miss_audited.jsonl"
OUT="$DPO/outputs/partial_miss_audited/$MODEL_KEY"
TEST="$ROOT/Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl"

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-?}  model: $MODEL_KEY ==="
echo "pairs: $PAIRS"
nvidia-smi -L || true

source "$ROOT/Claude/SFT/blind/gpu_env.sh"
mkdir -p "$OUT"

echo "=== Starting Token-Masked DPO Training ==="
python "$DPO/train_dpo.py" \
  --pairs "$PAIRS" \
  --policy-init "$POLICY_INIT" \
  --out "$OUT" \
  --beta 0.1 \
  --rpo-alpha 1.0 \
  --lr 5e-6 \
  --epochs 2 \
  --save-all \
  --mask-audit-only

echo "=== Starting Test Evaluation ==="
python "$ROOT/Claude/SFT/blind/eval_sft.py" \
  --model "$MODEL_KEY" \
  --adapter "$OUT/final_adapter" \
  --test "$TEST" \
  --schema-in-prompt \
  --out "$OUT/test_predictions_dpo.jsonl"

echo "=== Computing Metrics ==="
python "$ROOT/Claude/SFT/compute_metrics.py" \
  --pred "$OUT/test_predictions_dpo.jsonl" \
  --gt "$TEST"

echo "=== Done ==="
