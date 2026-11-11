#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=grpo_medgemma4b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/Claude/GRPO/outputs/slurm_%x_%j.log
#
# Guided GRPO (recipe v3) on top of the MedGemma-4B-it MedGuardBench SFT adapter.
# Every stage is skipped when its output already exists, so resubmitting the
# same script resumes from the last finished stage / latest checkpoint.
#
#   sbatch run_grpo_medgemma4b.slurm
#   sbatch --export=ALL,SEED=1,RUN=grpo-guided-v3-seed1 run_grpo_medgemma4b.slurm
#   sbatch --export=ALL,TRAIN_ARGS="--max-steps 20" run_grpo_medgemma4b.slurm   # smoke test
#
# Stages: merge SFT LoRA -> mine hard examples -> guided set -> GRPO train
#         -> pick best checkpoint on val -> paired test eval vs SFT.

set -euo pipefail

export MGB_MODEL=medgemma-4b
export MGB_DATA_CONDITION=legacy_v1
export MGB_GRPO_RECIPE=v3
export MGB_LOAD_IN_4BIT=0
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
export HF_HOME="${HF_HOME:-/fs/scratch/PCS0289/sarakhosravi/huggingface}"

SEED="${SEED:-42}"
RUN="${RUN:-grpo-guided-v3}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRPO_ENV="${GRPO_ENV:-grpo_env}"

GRPO_DIR=/users/PCS0289/sarakhosravi/Guardrail/Claude/GRPO
cd "$GRPO_DIR"
mkdir -p outputs data

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-?}  run: $RUN  seed: $SEED ==="
nvidia-smi -L || true

# shellcheck disable=SC1091
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  source /apps/python/3.12/etc/profile.d/conda.sh
fi
conda activate "$GRPO_ENV"
python -c "import trl, unsloth, vllm; print('trl', trl.__version__, '| unsloth', unsloth.__version__, '| vllm', vllm.__version__)"

# Same cluster workarounds as run_grpo.slurm (wrong bnb lib, icc host compiler,
# FlashInfer JIT failures).
unset BNB_CUDA_VERSION
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export UNSLOTH_VLLM_NO_FLASHINFER=1

python grpo_config.py
MERGED=$(python -c "import grpo_config as C; print(C.SFT_MERGED_PATH)")
ADAPTER=$(python -c "import grpo_config as C; print(C.SFT_ADAPTER_PATH)")
DATA=$(python -c "import grpo_config as C; print(C.HARD_EXAMPLES_JSONL.parent)")
OUT_ROOT=$(python -c "import grpo_config as C; print(C.GRPO_OUTPUT_DIR)")
OUT="$OUT_ROOT/$RUN"
MINED="$DATA/grpo_train_v2_v3.jsonl"
GUIDED="$DATA/grpo_train_guided_v3.jsonl"
mkdir -p "$DATA"

# ---- 1. Merge the SFT LoRA into bf16 weights (GRPO's init policy and KL ref) ----
# merge_legacy.py (not merge_ckpt.py) also writes <end_of_turn> as eos into
# config/generation_config; without it vLLM never stops and every completion
# is masked as truncated.
if [[ ! -f "$MERGED/config.json" ]]; then
  echo "=== merge $ADAPTER -> $MERGED ==="
  python merge_legacy.py
fi
# Unsloth loads MedGemma through its processor even in text_only mode; a merged
# dir without the image-processor jsons fails at load time.
BASE_SNAP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('google/medgemma-4b-it', local_files_only=True))")
for f in preprocessor_config.json processor_config.json tokenizer.model; do
  if [[ ! -f "$MERGED/$f" && -f "$BASE_SNAP/$f" ]]; then
    cp "$BASE_SNAP/$f" "$MERGED/$f"
    echo "copied $f into merged dir"
  fi
done

# ---- 2. Hard-example mining on the train split (v3: >= 40% gold-safe) ----
if [[ ! -s "$MINED" ]]; then
  echo "=== mine ==="
  python reward.py
  python mine_hard_examples.py --out "$MINED" ${MINE_ARGS:-}
fi
wc -l "$MINED"

# ---- 3. Guided GRPO set (mined + single-risk counterfactual items) ----
if [[ ! -s "$GUIDED" ]]; then
  echo "=== build guided set ==="
  python build_grpo_from_dpo.py --mined "$MINED" --out "$GUIDED"
fi
wc -l "$GUIDED"

# ---- 4. GRPO training ----
if [[ ! -d "$OUT/final" ]]; then
  RESUME=""
  if compgen -G "$OUT/checkpoint-*" > /dev/null; then
    RESUME="--resume"
  fi
  echo "=== train $RUN ==="
  # TRAIN_ARGS is intentionally unquoted so extra flags split.
  # shellcheck disable=SC2086
  python train_grpo.py \
    --run-name "$RUN" --seed "$SEED" \
    --train-file "$GUIDED" \
    --per-device-batch "$PER_DEVICE_BATCH" \
    --vllm-gpu-util "$VLLM_GPU_MEMORY_UTILIZATION" \
    $RESUME ${TRAIN_ARGS:-}
fi

# ---- 5. Pick the best checkpoint on the VALIDATION split ----
if [[ ! -d "$OUT/final_selected" ]]; then
  echo "=== select checkpoint ==="
  python select_grpo_checkpoint.py --run-dir "$OUT" --every 2 || true
fi
EVAL_MODEL="$OUT/final_selected"
if [[ ! -d "$EVAL_MODEL" ]]; then
  echo "no final_selected (SFT won on val); evaluating $OUT/final instead"
  EVAL_MODEL="$OUT/final"
fi

# ---- 6. Paired test evaluation against the merged SFT model ----
echo "=== eval $EVAL_MODEL ==="
python eval_grpo.py \
  --model "$EVAL_MODEL" \
  --compare-model "$MERGED" \
  --bootstrap 2000
EVAL_DIR="$EVAL_MODEL/eval"
python per_category_report.py "$EVAL_DIR/test_predictions.jsonl" --label GRPO \
  > "$EVAL_DIR/per_category_grpo.txt" || true
python per_category_report.py "$EVAL_DIR/comparison_predictions.jsonl" --label SFT \
  > "$EVAL_DIR/per_category_sft.txt" || true

echo "=== done: $EVAL_MODEL ==="
