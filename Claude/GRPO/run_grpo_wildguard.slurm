#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=grpo_wg7b
#SBATCH --clusters=ascend
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/Claude/GRPO/outputs/slurm_%x_%j.log
#
# Full GRPO pipeline for WildGuard-7B on top of its legacy_v1 SFT run:
#   1. merge   SFT LoRA (final_adapter) -> fp16 base  => final_adapter-merged
#   2. mine    hard examples            => data/legacy_v1-wildguard-7b/grpo_train.jsonl
#   3. train   fresh LoRA via GRPO      => outputs/legacy_v1-wildguard-7b/grpo-seed42
#   4. eval    final adapter vs merged SFT
#
# Each step is skipped if its output already exists, so the job is resumable.
#
#   sbatch Claude/GRPO/run_grpo_wildguard.slurm

set -euo pipefail

GRPO_DIR=/users/PCS0289/sarakhosravi/Guardrail/Claude/GRPO
cd "$GRPO_DIR"
mkdir -p outputs data

# ---- GRPO backbone / data selection (read by grpo_config.py) ----
export MGB_MODEL=wildguard-7b
export MGB_DATA_CONDITION=legacy_v1
# 7B policy on a 40 GB A100: 4-bit policy + 0.55 vLLM KV, mirroring the 8B arms.
export MGB_LOAD_IN_4BIT=1
export VLLM_GPU_MEMORY_UTILIZATION=0.55

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-?}  $(date) ==="
nvidia-smi -L || true

# ---- environment ----
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  source /apps/python/3.12/etc/profile.d/conda.sh
fi
conda activate "${GRPO_ENV:-grpo_env}"
python -c "import trl, unsloth, vllm; print('trl', trl.__version__, '| unsloth', unsloth.__version__, '| vllm', vllm.__version__)"

# grpo_env ships torch cu130; forcing BNB_CUDA_VERSION=121 (the SFT env's value)
# would load the wrong bitsandbytes. Make sure it is unset.
unset BNB_CUDA_VERSION || true
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# FlashInfer JIT-compile fails on this cluster (icc host compiler, missing
# curand.h); Unsloth's off switch falls back to FlashAttention + torch sampler.
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export UNSLOTH_VLLM_NO_FLASHINFER=1

MERGED=$(python -c "import grpo_config as C; print(C.SFT_MERGED_PATH)")
TRAINSET=$(python -c "import grpo_config as C; print(C.HARD_EXAMPLES_JSONL)")
OUT_DIR=$(python -c "import grpo_config as C; print(C.GRPO_OUTPUT_DIR)")
RUN=grpo-seed42

echo "=== merged SFT : $MERGED"
echo "=== train set  : $TRAINSET"
echo "=== output dir : $OUT_DIR/$RUN"

# ---- 1. merge SFT adapter onto fp16 base (skips if already present) ----
echo "### STEP 1: merge ###"
python merge_legacy.py

# ---- 2. mine hard examples (skips if the training set already exists) ----
echo "### STEP 2: mine ###"
if [[ -s "$TRAINSET" ]]; then
  echo "mined training set already exists -> $TRAINSET  (skipping)"
else
  python reward.py
  python mine_hard_examples.py
fi

# ---- 3. GRPO training ----
echo "### STEP 3: train ###"
python train_grpo.py --seed 42 --run-name "$RUN" --load-in-4bit

# ---- 4. evaluate the final adapter vs the merged SFT policy ----
echo "### STEP 4: eval ###"
COMPARE=$(python -c "import grpo_config as C; print(C.SFT_MERGED_PATH)")
python eval_grpo.py \
  --model "${OUT_DIR}/${RUN}/final" \
  --compare-model "$COMPARE" \
  --load-in-4bit \
  --bootstrap 2000

echo "=== done $(date) ==="
