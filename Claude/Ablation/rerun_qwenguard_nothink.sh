#!/usr/bin/env bash
# Root-cause fix for Qwen3Guard-4B/8B base unparsable rows: the custom ChatML
# template does not suppress reasoning, so these Qwen3-based guard models emit
# <think> and loop past the token budget without ever committing to the JSON
# verdict. --no-think pre-fills an empty <think></think> block (enable_thinking
# =False) so they answer directly. Re-runs ONLY the unparsable rows (resume).
set -uo pipefail
ROOT=/users/PCS0289/sarakhosravi/Guardrail
cd "$ROOT"
# shellcheck disable=SC1091
source Claude/SFT/blind/gpu_env.sh

SB=/fs/scratch/PCS0289/sarakhosravi/rerun4k_b
TEST="$ROOT/Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl"
E="python Claude/SFT/blind/eval_sft.py --test $TEST --max-new-tokens 2048 --max-seq-length 4096 --batch-size 8 --no-think --no-repeat-ngram-size 3 --base --schema-in-prompt"

echo "=== $(hostname) $(date) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
run() { echo "--- $* $(date) ---"; "$@"; echo "exit=$? $(date)"; }

for KEY in qwen3guard-4b qwen3guard-8b; do
  run $E --model "$KEY" --out "$SB/$KEY/test_predictions.jsonl"
done
echo "=== ALL DONE $(date) ==="
