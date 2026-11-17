#!/usr/bin/env bash
# Full GRPO pipeline for WildGuard-7B on top of its legacy_v1 SFT run, run on
# the EXISTING a0374 allocation via `srun --jobid=<job> --overlap` (no new GPU
# request). Steps 1-4; each is skipped if its output already exists.
#   1. merge SFT LoRA -> final_adapter-merged
#   2. mine hard examples -> data/legacy_v1-wildguard-7b/grpo_train.jsonl
#   3. train fresh LoRA via GRPO -> outputs/legacy_v1-wildguard-7b/grpo-seed42
#   4. eval final adapter vs merged SFT
set -euo pipefail

GRPO_DIR=/users/PCS0289/sarakhosravi/Guardrail/Claude/GRPO
cd "$GRPO_DIR"
mkdir -p outputs data

export MGB_MODEL=wildguard-7b
export MGB_DATA_CONDITION=legacy_v1
# Proven 40 GB A100 config (matches the successful Qwen3-8B / Qwen3Guard-8B
# GRPO runs): bf16 policy, vLLM util 0.45, and grpo_mini_batch 32 so the
# backward pass is chunked instead of materializing all 64 completions at
# once (that was the OOM in the first attempt at util 0.55 / no mini-batch).
export MGB_LOAD_IN_4BIT=0
export VLLM_GPU_MEMORY_UTILIZATION=0.45
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GRPO_MINI_BATCH=32

echo "=== node: $(hostname)  $(date) ==="
nvidia-smi -L || true

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  source /apps/python/3.12/etc/profile.d/conda.sh
fi
conda activate "${GRPO_ENV:-grpo_env}"
python -c "import trl, unsloth, vllm; print('trl', trl.__version__, '| unsloth', unsloth.__version__, '| vllm', vllm.__version__)"

unset BNB_CUDA_VERSION || true
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export UNSLOTH_VLLM_NO_FLASHINFER=1

TRAINSET=$(python -c "import grpo_config as C; print(C.HARD_EXAMPLES_JSONL)")
OUT_DIR=$(python -c "import grpo_config as C; print(C.GRPO_OUTPUT_DIR)")
RUN=grpo-seed42
echo "=== train set : $TRAINSET"
echo "=== output    : $OUT_DIR/$RUN"

echo "### STEP 1: merge ###"
python merge_legacy.py

echo "### STEP 2: mine ###"
if [[ -s "$TRAINSET" ]]; then
  echo "mined training set already exists -> $TRAINSET  (skipping)"
else
  python reward.py
  python mine_hard_examples.py
fi

echo "### STEP 3: train ###"
python train_grpo.py --seed 42 --run-name "$RUN" \
  --no-4bit --grpo-mini-batch "$GRPO_MINI_BATCH" --vllm-gpu-util 0.45 \
  --epochs 1 --resume

echo "### STEP 4: eval ###"
COMPARE=$(python -c "import grpo_config as C; print(C.SFT_MERGED_PATH)")
python eval_grpo.py \
  --model "${OUT_DIR}/${RUN}/final" \
  --compare-model "$COMPARE" \
  --bootstrap 2000

echo "=== done $(date) ==="
