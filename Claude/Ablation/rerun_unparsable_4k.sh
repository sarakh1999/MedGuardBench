#!/usr/bin/env bash
# Re-run ONLY the previously-unparsable zero-shot base rows for Qwen3-8B / Qwen3-14B
# at the corrected generation budget (max_new_tokens=4096, max_seq=8192), then let
# eval_sft.py's resume logic fill exactly the missing (unparsable) indices into the
# parsed-row-seeded output. Identical model / prompt / parsing as the original run;
# only the token budget changed.
#
# Output lives on SCRATCH (stable for compute I/O); home NFS threw stale-file-handle
# errors during repeated writes on the first attempt.
set -euo pipefail

ROOT=/users/PCS0289/sarakhosravi/Guardrail
source "$ROOT/Claude/SFT/blind/gpu_env.sh"   # activates vllm_env, cd's to ROOT
cd "$ROOT"

TEST="$ROOT/Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl"
SB=/fs/scratch/PCS0289/sarakhosravi/rerun4k

echo "=== node $(hostname) ==="
nvidia-smi -L || true

for KEY in qwen3-8b qwen3-14b; do
  OUT="$SB/$KEY/base/true/test_predictions.jsonl"
  echo ""
  echo "############ $KEY ############"
  echo "seeded parsed rows: $(wc -l < "$OUT")"
  python "$ROOT/Claude/SFT/blind/eval_sft.py" \
    --model "$KEY" --base --schema-in-prompt \
    --test "$TEST" --out "$OUT" \
    --max-new-tokens 4096 --max-seq-length 8192 --batch-size 8
  echo "after re-run rows: $(wc -l < "$OUT")"
done

echo ""
echo "=== ALL DONE ==="
