#!/usr/bin/env bash
set -euo pipefail

ROOT=/users/PCS0289/sarakhosravi/Guardrail
DPO="$ROOT/Claude/DPO"
PAIRS="$DPO/data/dpo_partial_miss_audited.jsonl"
TEST="$ROOT/Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl"

source "$ROOT/Claude/SFT/blind/gpu_env.sh"

run_experiment() {
  local MODEL_KEY="$1"
  local SEED="$2"

  case "$MODEL_KEY" in
    qwen3-4b) SFT_RUN="Qwen3-4B-Instruct" ;;
    qwen3-8b) SFT_RUN="Qwen3-8B-Instruct" ;;
    *) echo "Unknown model $MODEL_KEY" >&2; exit 1 ;;
  esac

  local POLICY_INIT="$ROOT/Claude/SFT/new_outputs/$SFT_RUN/checkpoint-950-merged"
  local OUT="$DPO/outputs/partial_miss_audited/${MODEL_KEY}_seed${SEED}"

  echo "=========================================================================="
  echo "STARTING: Model=$MODEL_KEY | Seed=$SEED | Time=$(date)"
  echo "Output dir: $OUT"
  echo "=========================================================================="

  mkdir -p "$OUT"

  # 1. Train Token-Masked DPO
  python "$DPO/train_dpo.py" \
    --pairs "$PAIRS" \
    --policy-init "$POLICY_INIT" \
    --out "$OUT" \
    --beta 0.1 \
    --rpo-alpha 1.0 \
    --lr 5e-6 \
    --epochs 2 \
    --seed "$SEED" \
    --save-all \
    --mask-audit-only

  # 2. Test Set Inference
  python "$ROOT/Claude/SFT/blind/eval_sft.py" \
    --model "$MODEL_KEY" \
    --adapter "$OUT/final_adapter" \
    --test "$TEST" \
    --schema-in-prompt \
    --out "$OUT/test_predictions_dpo.jsonl"

  # 3. Compute Metrics
  python "$ROOT/Claude/SFT/compute_metrics.py" \
    --pred "$OUT/test_predictions_dpo.jsonl" \
    --gt "$TEST"

  echo "=========================================================================="
  echo "FINISHED: Model=$MODEL_KEY | Seed=$SEED | Time=$(date)"
  echo "=========================================================================="
}

# Run 4B seeds 43 and 44
run_experiment "qwen3-4b" 43
run_experiment "qwen3-4b" 44

# Run 8B seeds 43 and 44
run_experiment "qwen3-8b" 43
run_experiment "qwen3-8b" 44

echo "=== ALL SEEDS COMPLETED ==="
