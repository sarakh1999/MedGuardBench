#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=unroll_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=v100-32g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/logs/unroll_eval_%x_%A_%a.log
#
# Inference-time patient-profile unrolling on an EXISTING SFT checkpoint.
# Keeps the model fixed and asks only whether injecting the unrolled context
# changes its predictions.
#
#   sbatch --array=0-3 context_unrolling/run_unroll_eval.slurm      # given-context, 4 shards of the test set
#   sbatch --export=ALL,MODE=self,LIMIT=100 context_unrolling/run_unroll_eval.slurm   # self-rollout, subset
#
# Env vars: MODE (given|self|direct), CKPT, TEST, LIMIT, SHARDS (default 4), TAG

set -euo pipefail

ROOT=/users/PCS0289/sarakhosravi/Guardrail
MODE="${MODE:-given}"
CKPT="${CKPT:-$ROOT/Claude/SFT/new_outputs/Qwen3-4B-Instruct/checkpoint-950}"
TEST="${TEST:-$ROOT/Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl}"
SHARDS="${SHARDS:-4}"
TAG="${TAG:-ckpt950}"
OUT_DIR="$ROOT/context_unrolling/outputs/infer_${TAG}"
mkdir -p "$OUT_DIR"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHARD_ARG="--shard ${SLURM_ARRAY_TASK_ID}/${SHARDS}"
  OUT="$OUT_DIR/${MODE}_shard${SLURM_ARRAY_TASK_ID}.jsonl"
else
  SHARD_ARG=""
  OUT="$OUT_DIR/${MODE}.jsonl"
fi
LIMIT_ARG=""
[[ -n "${LIMIT:-}" ]] && LIMIT_ARG="--limit $LIMIT"

echo "=== node $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-none} mode=$MODE ==="
echo "ckpt=$CKPT"; echo "test=$TEST"; echo "out=$OUT"
nvidia-smi -L || true

# shellcheck disable=SC1091
source "$ROOT/Claude/SFT/gpu_env.sh"
cd "$ROOT"

# shellcheck disable=SC2086
python context_unrolling/unroll_at_inference.py --mode "$MODE" \
    --checkpoint "$CKPT" --test-jsonl "$TEST" --out "$OUT" \
    --max-seq-length 4096 --max-new-tokens 2048 $SHARD_ARG $LIMIT_ARG

echo "=== done $(date) ==="
