#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=dpo_ckpt_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/Claude/DPO/logs/slurm_%x_%j.log
#
# Sweep DPO round-2 checkpoints on a test subsample to find the best-generalizing step.
#   sbatch -M ascend -p nextgen --export=ALL,MODEL_KEY=qwen3-8b,OUTDIR=...,CKPTS="300 650 1320",LIMIT=250 Claude/DPO/run_ckpt_sweep.slurm

set -euo pipefail
ROOT=/users/PCS0289/sarakhosravi/Guardrail
cd "$ROOT"
source "$ROOT/Claude/SFT/blind/gpu_env.sh"

MODEL_KEY="${MODEL_KEY:-qwen3-8b}"
OUTDIR="${OUTDIR:?}"
CKPTS="${CKPTS:-300 500 650 900 1320}"
LIMIT="${LIMIT:-250}"
TEST="$ROOT/Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl"

echo "=== node: $(hostname)  model: $MODEL_KEY  limit: $LIMIT ==="
for S in $CKPTS; do
  CK="$OUTDIR/checkpoint-$S"
  if [ ! -d "$CK" ]; then echo "missing $CK"; continue; fi
  PRED="$OUTDIR/sweep_ckpt${S}_predictions.jsonl"
  MDIR="$OUTDIR/sweep_ckpt${S}_metrics"
  mkdir -p "$MDIR"
  # eval_sft.py resumes from an existing $PRED (skips rows already done), so always
  # call it: a larger LIMIT extends a partial file instead of being ignored.
  python "$ROOT/Claude/SFT/blind/eval_sft.py" --model "$MODEL_KEY" --adapter "$CK" \
    --test "$TEST" --schema-in-prompt --limit "$LIMIT" --out "$PRED"
  echo "===== checkpoint-$S ====="
  python "$ROOT/Claude/SFT/compute_metrics.py" --pred "$PRED" --gt "$TEST" --out_dir "$MDIR"
done
echo "SWEEP COMPLETE $MODEL_KEY"
