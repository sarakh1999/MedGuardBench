#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=unroll_full
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/logs/unroll_full_%x_%A_%a.log
#
# Full-scale patient-profile-unrolling arm: same recipe as the Qwen3-4B SFT
# baseline (r=16, alpha=32, lr 1e-4, 3 epochs, effective batch 8, cosine,
# early stopping on eval_loss, best checkpoint), full train/val, 754-row test.
#
#   STAGE=train    train one arm, then submit the eval shards + metrics job
#   STAGE=eval     --array=0-(N_SHARDS-1): generate predictions for one shard
#   STAGE=metrics  merge shards, run compute_metrics.py, print the comparison
#
#   sbatch --export=ALL,STAGE=train,ARM=unrolled_long__user__patient+prescription context_unrolling/run_full.slurm
#
# Also runnable as a plain bash script on an allocated GPU node:
#   STAGE=train ARM=... bash context_unrolling/run_full.slurm

set -euo pipefail

ROOT=/users/PCS0289/sarakhosravi/Guardrail
STAGE="${STAGE:-train}"
ARM="${ARM:?ARM required (folder under context_unrolling/data/chatml)}"
N_SHARDS="${N_SHARDS:-4}"
MAX_SEQ="${MAX_SEQ:-4096}"
EPOCHS="${EPOCHS:-3}"
SEED="${SEED:-3407}"
BS="${BS:-1}"
GA="${GA:-8}"

DATA="$ROOT/context_unrolling/data/chatml/$ARM"
OUT="$ROOT/context_unrolling/outputs/full/$ARM"
SELF="$ROOT/context_unrolling/run_full.slurm"
mkdir -p "$OUT" "$ROOT/logs"

# shellcheck disable=SC1091
source "$ROOT/Claude/SFT/gpu_env.sh"
cd "$ROOT"
echo "=== stage=$STAGE arm=$ARM node=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-none} $(date) ==="

submit_eval_chain() {
  local eval_id metrics_id
  eval_id=$(sbatch --parsable --array=0-$((N_SHARDS - 1)) --time=08:00:00 \
            --export=ALL,STAGE=eval,ARM="$ARM",N_SHARDS="$N_SHARDS",MAX_SEQ="$MAX_SEQ" "$SELF")
  metrics_id=$(sbatch --parsable --dependency=afterok:"$eval_id" --time=00:30:00 --partition=cpu --gpus-per-node=0 \
               --export=ALL,STAGE=metrics,ARM="$ARM",N_SHARDS="$N_SHARDS" "$SELF")
  echo "submitted eval array $eval_id and metrics job $metrics_id"
}

case "$STAGE" in
  train)
    nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
    if [ ! -f "$OUT/final_adapter/adapter_config.json" ]; then
      python Claude/SFT/sft_train.py --model qwen3-4b \
          --data-dir "$DATA" --output-dir "$OUT" \
          --epochs "$EPOCHS" --seed "$SEED" --max-seq "$MAX_SEQ" \
          --batch-size "$BS" --grad-accum "$GA" \
          --eval-steps 50 --patience 3 --save-total-limit 3
    else
      echo "adapter exists, skipping training"
    fi
    submit_eval_chain
    ;;

  eval)
    K="${SLURM_ARRAY_TASK_ID:?run under --array}"
    python context_unrolling/unroll_at_inference.py --mode direct \
        --checkpoint "$OUT/final_adapter" --test-jsonl "$DATA/test.jsonl" \
        --out "$OUT/test_predictions.shard$K.jsonl" --shard "$K/$N_SHARDS" \
        --max-seq-length "$MAX_SEQ" --max-new-tokens 2048
    ;;

  metrics)
    cat "$OUT"/test_predictions.shard*.jsonl > "$OUT/test_predictions.jsonl"
    python Claude/SFT/compute_metrics.py --pred "$OUT/test_predictions.jsonl" --gt "$DATA/test.jsonl"
    python context_unrolling/compare_arms.py --root "$ROOT/context_unrolling/outputs/full"
    ;;

  *) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
echo "=== done stage=$STAGE arm=$ARM $(date) ==="
