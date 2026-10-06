#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=dpo_medguard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/Claude/DPO/logs/slurm_%x_%j.log
#
# DPO train + test eval + per-category metrics in one batch job.
#
#   sbatch --export=ALL,MODEL_KEY=qwen3-8b,PAIRS=...,OUT=... Claude/DPO/run_dpo.slurm
#   sbatch -M ascend -p nextgen --export=ALL,... Claude/DPO/run_dpo.slurm   # A100
#
# Env vars (defaults in parentheses):
#   MODEL_KEY    qwen3-8b | qwen3-4b            (qwen3-8b)
#   PAIRS        pairs jsonl                    (data/mixed_balanced_v2/dpo_pairs.jsonl)
#   OUT          output dir                     (outputs/mixed_balanced_v2/$MODEL_KEY)
#   BETA (0.1)  RPO_ALPHA (1.0)  LR (5e-6)  EPOCHS (2)  SAVE_ALL (1)

set -euo pipefail

ROOT=/users/PCS0289/sarakhosravi/Guardrail
DPO="$ROOT/Claude/DPO"
MODEL_KEY="${MODEL_KEY:-qwen3-8b}"
PAIRS="${PAIRS:-$DPO/data/mixed_balanced_v2/dpo_pairs.jsonl}"
OUT="${OUT:-$DPO/outputs/mixed_balanced_v2/$MODEL_KEY}"
case "$MODEL_KEY" in
  qwen3-8b) SFT_RUN=Qwen3-8B-Instruct ;;
  qwen3-4b) SFT_RUN=Qwen3-4B-Instruct ;;
  *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
POLICY_INIT="${POLICY_INIT:-$ROOT/Claude/SFT/new_outputs/$SFT_RUN/checkpoint-950-merged}"
TEST="$ROOT/Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl"

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-?}  model: $MODEL_KEY ==="
echo "pairs: $PAIRS"
nvidia-smi -L || true

# shellcheck disable=SC1091
source "$ROOT/Claude/SFT/blind/gpu_env.sh"
mkdir -p "$OUT"

SAVE_FLAG=""
[ "${SAVE_ALL:-1}" = "1" ] && SAVE_FLAG="--save-all"

python "$DPO/train_dpo.py" \
  --pairs "$PAIRS" \
  --policy-init "$POLICY_INIT" \
  --out "$OUT" \
  --beta "${BETA:-0.1}" --rpo-alpha "${RPO_ALPHA:-1.0}" \
  --lr "${LR:-5e-6}" --epochs "${EPOCHS:-2}" $SAVE_FLAG

python "$ROOT/Claude/SFT/blind/eval_sft.py" --model "$MODEL_KEY" \
  --adapter "$OUT/final_adapter" \
  --test "$TEST" \
  --schema-in-prompt \
  --out "$OUT/test_predictions_dpo.jsonl"

python "$ROOT/Claude/SFT/compute_metrics.py" \
  --pred "$OUT/test_predictions_dpo.jsonl" \
  --gt "$TEST"

echo "=== done ==="
