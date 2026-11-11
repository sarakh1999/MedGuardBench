#!/usr/bin/env bash
# Re-run ONLY the unparsable rows of four arms at the new budget (4096/8192),
# on the booked node a0176. Uses eval_sft.py resume: parsed rows are pre-seeded
# into the scratch out file, so the generator regenerates only the missing idx.
set -uo pipefail
ROOT=/users/PCS0289/sarakhosravi/Guardrail
cd "$ROOT"
# shellcheck disable=SC1091
source Claude/SFT/blind/gpu_env.sh

SB=/fs/scratch/PCS0289/sarakhosravi/rerun4k_b
TEST="$ROOT/Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl"
CU_ADAPTER="$ROOT/context_unrolling/outputs/ontop_14b/sft+1ep_unrolled_v4pro/final_adapter"
E="python Claude/SFT/blind/eval_sft.py --test $TEST --max-new-tokens 4096 --max-seq-length 8192 --batch-size 8"

echo "=== $(hostname) $(date) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

run() { echo "--- $* $(date) ---"; "$@"; echo "exit=$? $(date)"; }

# three base models (schema in prompt)
for KEY in qwen3guard-4b qwen3guard-8b shieldgemma-2b; do
  run $E --model "$KEY" --base --schema-in-prompt --out "$SB/$KEY/test_predictions.jsonl"
done

# Qwen3-14B context-unrolling arm (trained adapter, no schema-in-prompt)
run $E --model qwen3-14b --adapter "$CU_ADAPTER" --out "$SB/qwen3-14b-cu/test_predictions.jsonl"

echo "=== ALL DONE $(date) ==="
