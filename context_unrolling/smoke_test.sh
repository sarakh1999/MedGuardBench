#!/usr/bin/env bash
# Smoke test for patient-profile unrolling: one arm of the ablation ladder.
#
# Trains Qwen3-4B-Instruct with the standard SFT recipe on a capped subset of
# one ChatML mode, then evaluates the best adapter on the first N test rows
# and writes compute_metrics.py output. Every arm uses the same seed, the same
# train/val/test rows and the same hyperparameters, so the arms are directly
# comparable.
#
#   bash context_unrolling/smoke_test.sh <mode_tag> [n_train] [n_val] [n_test]
#
#   mode_tag  a folder name under context_unrolling/data/chatml, e.g.
#             direct | long | unrolled__assistant__patient+prescription |
#             unrolled_long__assistant__patient+prescription
#
# Run on a GPU node (sources Claude/SFT/blind/gpu_env.sh).

set -euo pipefail

ARM="${1:?mode_tag required}"
N_TRAIN="${2:-1000}"
N_VAL="${3:-100}"
N_TEST="${4:-150}"
SEED="${SEED:-3407}"
MAX_SEQ="${MAX_SEQ:-4096}"
EPOCHS="${EPOCHS:-1}"

ROOT=/users/PCS0289/sarakhosravi/Guardrail
DATA="$ROOT/context_unrolling/data/chatml/$ARM"
OUT="$ROOT/context_unrolling/outputs/smoke/$ARM"
PRED="$OUT/test_predictions.jsonl"

[ -d "$DATA" ] || { echo "no data dir: $DATA"; exit 1; }
mkdir -p "$OUT"

# shellcheck disable=SC1091
source "$ROOT/Claude/SFT/blind/gpu_env.sh"
cd "$ROOT"

echo "=== smoke arm: $ARM  node: $(hostname)  $(date) ==="
echo "train=$N_TRAIN val=$N_VAL test=$N_TEST seed=$SEED max_seq=$MAX_SEQ epochs=$EPOCHS"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader

# ---------------------------------------------------------------- train
if [ ! -f "$OUT/final_adapter/adapter_config.json" ]; then
  python Claude/SFT/blind/sft_train.py --model qwen3-4b \
      --data-dir "$DATA" --output-dir "$OUT" \
      --max-train "$N_TRAIN" --max-val "$N_VAL" \
      --epochs "$EPOCHS" --seed "$SEED" --max-seq "$MAX_SEQ" \
      --eval-steps 25 --patience 5 --save-total-limit 2
else
  echo "adapter exists, skipping training"
fi

# ---------------------------------------------------------------- eval
python context_unrolling/unroll_at_inference.py --mode direct \
    --checkpoint "$OUT/final_adapter" \
    --test-jsonl "$DATA/test.jsonl" \
    --out "$PRED" --limit "$N_TEST" \
    --max-seq-length "$MAX_SEQ" --max-new-tokens 2048

python Claude/SFT/compute_metrics.py --pred "$PRED" --gt "$DATA/test.jsonl"

echo "=== done: $ARM  $(date) ==="
