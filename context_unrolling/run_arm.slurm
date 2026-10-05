#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=unroll_arm
#SBATCH --partition=gpu-exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/logs/unroll_arm_%x_%A_%a.log
#
# One arm of a context-unrolling comparison: optional training (from the base
# model or on top of an existing SFT adapter), then a sharded 754-row eval and
# compute_metrics.py. Supersedes run_full.slurm.
#
# Environment variables
#   STAGE         train | evalchain | eval | metrics        (default train)
#   ARM           output name under $OUT_ROOT               (required)
#   DATA_ARM      chatml folder under context_unrolling/data/chatml (default $ARM)
#   OUT_ROOT      default context_unrolling/outputs/ontop
#   INIT_ADAPTER  continue training from this adapter dir (omit = from base)
#   EPOCHS LR BS GA MAX_SEQ SEED  training recipe (defaults 3, 1e-4, 1, 8, 4096, 3407)
#   MODE          eval mode for unroll_at_inference.py: direct | given (default direct)
#   EVAL_BACKEND  unroll (default, JSON-output Qwen family) | eval_sft (Llama/LlamaGuard/Gemma)
#   CHECKPOINT    adapter to evaluate (default $OUT/final_adapter)
#   N_SHARDS      eval parallelism (default 4)
#
# Examples
#   # SFT -> +1 epoch on unrolled data, self-rollout
#   sbatch --export=ALL,STAGE=train,ARM=ontop_unrolled_long_assistant,\
#     DATA_ARM=unrolled_long__assistant__patient+prescription,\
#     INIT_ADAPTER=Claude/SFT/new_outputs/Qwen3-4B-Instruct/checkpoint-950,EPOCHS=1 \
#     context_unrolling/run_arm.slurm
#   # inference-time unrolling on the existing SFT checkpoint, no training
#   STAGE=evalchain ARM=ontop_given_only DATA_ARM=long MODE=given \
#     CHECKPOINT=Claude/SFT/new_outputs/Qwen3-4B-Instruct/checkpoint-950 \
#     bash context_unrolling/run_arm.slurm

set -euo pipefail

ROOT=/users/PCS0289/sarakhosravi/Guardrail
STAGE="${STAGE:-train}"
ARM="${ARM:?ARM required}"
DATA_ARM="${DATA_ARM:-$ARM}"
OUT_ROOT="${OUT_ROOT:-$ROOT/context_unrolling/outputs/ontop}"
INIT_ADAPTER="${INIT_ADAPTER:-}"
N_SHARDS="${N_SHARDS:-4}"
MAX_SEQ="${MAX_SEQ:-4096}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
SEED="${SEED:-3407}"
BS="${BS:-1}"
GA="${GA:-8}"
MODE="${MODE:-direct}"
MODEL_KEY="${MODEL_KEY:-qwen3-4b}"
EVAL_BACKEND="${EVAL_BACKEND:-unroll}"   # unroll | eval_sft

DATA="$ROOT/context_unrolling/data/chatml/$DATA_ARM"
OUT="$OUT_ROOT/$ARM"
CHECKPOINT="${CHECKPOINT:-$OUT/final_adapter}"
SELF="$ROOT/context_unrolling/run_arm.slurm"
mkdir -p "$OUT" "$ROOT/logs"

# shellcheck disable=SC1091
source "$ROOT/Claude/SFT/blind/gpu_env.sh"
cd "$ROOT"
echo "=== stage=$STAGE arm=$ARM data=$DATA_ARM mode=$MODE node=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-none} $(date) ==="
[ -d "$DATA" ] || { echo "no data dir: $DATA"; exit 1; }

# No ALL: when the submitter runs on another cluster (Ascend), its environment
# (compiler paths etc.) must not leak into the Pitzer job; gpu_env.sh rebuilds it.
EXPORTS="ARM=$ARM,DATA_ARM=$DATA_ARM,OUT_ROOT=$OUT_ROOT,N_SHARDS=$N_SHARDS,MAX_SEQ=$MAX_SEQ,MODE=$MODE,CHECKPOINT=$CHECKPOINT,EVAL_BACKEND=$EVAL_BACKEND"

submit_eval_chain() {
  local eval_id metrics_id
  # SBATCH_CLUSTERS=pitzer lets a training running elsewhere (e.g. an Ascend
  # OnDemand node) farm its eval out to Pitzer; --parsable then prints
  # "jobid;cluster", so strip the cluster suffix before using it in afterok.
  eval_id=$(sbatch --parsable --array=0-$((N_SHARDS - 1)) --time=06:00:00 \
            --job-name="ev_${ARM:0:10}" --export="$EXPORTS,STAGE=eval,MODEL_KEY=$MODEL_KEY" "$SELF")
  eval_id="${eval_id%%;*}"
  metrics_id=$(sbatch --parsable --dependency=afterok:"$eval_id" --time=00:30:00 \
               --partition=cpu --gpus-per-node=0 --job-name="mt_${ARM:0:10}" \
               --export="$EXPORTS,STAGE=metrics,MODEL_KEY=$MODEL_KEY" "$SELF")
  metrics_id="${metrics_id%%;*}"
  echo "submitted eval array $eval_id and metrics job $metrics_id"
}

case "$STAGE" in
  train)
    nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
    if [ ! -f "$OUT/final_adapter/adapter_config.json" ]; then
      INIT_ARGS=()
      [ -n "$INIT_ADAPTER" ] && INIT_ARGS=(--init-adapter "$ROOT/$INIT_ADAPTER")
      python Claude/SFT/blind/sft_train.py --model "$MODEL_KEY" \
          --data-dir "$DATA" --output-dir "$OUT" \
          --epochs "$EPOCHS" --lr "$LR" --seed "$SEED" --max-seq "$MAX_SEQ" \
          --batch-size "$BS" --grad-accum "$GA" \
          --eval-steps 50 --patience 3 --save-total-limit 3 --resume "${INIT_ARGS[@]}"
    else
      echo "adapter exists, skipping training"
      if ls "$OUT"/test_predictions.shard*.jsonl >/dev/null 2>&1; then
        echo "eval already started for this arm, not resubmitting"; exit 0
      fi
    fi
    submit_eval_chain
    ;;

  evalchain)
    submit_eval_chain
    ;;

  eval)
    K="${SLURM_ARRAY_TASK_ID:?run under --array}"
    if [ "$EVAL_BACKEND" = "eval_sft" ]; then
      # family-aware batched generation (Llama-3 headers, LlamaGuard output
      # format, Gemma); the context is already in the test.jsonl prompts.
      python Claude/SFT/blind/eval_sft.py --model "$MODEL_KEY" --adapter "$CHECKPOINT" \
          --test "$DATA/test.jsonl" --out "$OUT/test_predictions.shard$K.jsonl" \
          --shard "$K/$N_SHARDS" --max-seq-length "$MAX_SEQ" --max-new-tokens 2048
    else
      python context_unrolling/unroll_at_inference.py --mode "$MODE" \
          --checkpoint "$CHECKPOINT" --test-jsonl "$DATA/test.jsonl" \
          --primitives patient_constraints,prescription \
          --out "$OUT/test_predictions.shard$K.jsonl" --shard "$K/$N_SHARDS" \
          --max-seq-length "$MAX_SEQ" --max-new-tokens 2048
    fi
    ;;

  metrics)
    cat "$OUT"/test_predictions.shard*.jsonl > "$OUT/test_predictions.jsonl"
    python Claude/SFT/compute_metrics.py --pred "$OUT/test_predictions.jsonl" --gt "$DATA/test.jsonl"
    python context_unrolling/compare_arms.py --root "$OUT_ROOT"
    ;;

  *) echo "unknown STAGE=$STAGE"; exit 2 ;;
esac
echo "=== done stage=$STAGE arm=$ARM $(date) ==="
