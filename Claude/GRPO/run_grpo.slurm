#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=grpo_medguard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/Claude/GRPO/outputs/slurm_%x_%A_%a.log
#
# GRPO on top of the merged Qwen3-4B SFT model.
#
#   sbatch run_grpo.slurm                         # one seed (42)
#   sbatch --array=1-3 run_grpo.slurm             # three seeds, runs grpo-seed{1,2,3}
#   sbatch --export=ALL,STAGE=mine run_grpo.slurm # hard-example mining only
#   sbatch --export=ALL,STAGE=eval,MODEL=outputs/grpo-seed1/final run_grpo.slurm
#   sbatch --export=ALL,STAGE=dry run_grpo.slurm  # 20-step pipeline check
#
# A 32 GB GPU is recommended (vLLM engine at 0.35 of memory + 4-bit policy +
# LoRA optimizer state). On Pitzer add:  --gres=gpu:v100-32g:1
# V100 has no bf16; the scripts fall back to fp16 automatically.
#
# GRPO_ENV must contain unsloth, trl >= 0.15 and vllm in ONE environment.
# The SFT env (unsloth_env) has trl 0.12.2 and no vllm; see setup_grpo_env.sh.

set -euo pipefail

GRPO_DIR=/users/PCS0289/sarakhosravi/Guardrail/Claude/GRPO
GRPO_ENV="${GRPO_ENV:-grpo_env}"
STAGE="${STAGE:-train}"

cd "$GRPO_DIR"
mkdir -p outputs data

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-?}  array: ${SLURM_ARRAY_TASK_ID:-none}  stage: $STAGE ==="
nvidia-smi -L || true

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$GRPO_ENV"
python -c "import trl, unsloth, vllm; print('trl', trl.__version__, '| unsloth', unsloth.__version__, '| vllm', vllm.__version__)"

# The SFT scripts force BNB_CUDA_VERSION=121 for their torch 2.5/cu121 env.
# grpo_env ships torch cu130; forcing 121 loads the wrong bitsandbytes
# library, so make sure it is NOT set here.
unset BNB_CUDA_VERSION
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# FlashInfer JIT-compiles CUDA kernels on first use and that fails on this
# cluster twice over: the default `intel` module exports CC=icc ("nvcc fatal:
# Unsupported host compiler 'ICC'"), and /usr/local/cuda lacks curand.h.
# UNSLOTH_VLLM_NO_FLASHINFER is Unsloth's supported off switch (it overrides
# VLLM_USE_FLASHINFER_SAMPLER itself on sm>=80); vLLM then uses FlashAttention
# + the PyTorch sampler, which is fine.
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export UNSLOTH_VLLM_NO_FLASHINFER=1

case "$STAGE" in
  mine)
    python reward.py
    # MINE_ARGS is intentionally unquoted so e.g. MINE_ARGS="--limit 200" splits
    # shellcheck disable=SC2086
    python mine_hard_examples.py ${MINE_ARGS:-}
    ;;
  dry)
    python train_grpo.py --dry-run --run-name grpo-dry
    ;;
  train)
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      SEED="$SLURM_ARRAY_TASK_ID"
      RUN="grpo-seed${SEED}"
    else
      SEED="${SEED:-42}"
      RUN="${RUN_NAME:-grpo-seed${SEED}}"
    fi
    python train_grpo.py --seed "$SEED" --run-name "$RUN"
    # Evaluate the final adapter against the merged SFT model in the same job.
    python eval_grpo.py \
      --model "outputs/${RUN}/final" \
      --compare-model /users/PCS0289/sarakhosravi/Guardrail/SFT/new_outputs/Qwen3-4B-Instruct/final \
      --bootstrap 2000
    ;;
  eval)
    : "${MODEL:?set MODEL=path/to/adapter}"
    python eval_grpo.py \
      --model "$MODEL" \
      --compare-model /users/PCS0289/sarakhosravi/Guardrail/SFT/new_outputs/Qwen3-4B-Instruct/final \
      --bootstrap 2000
    ;;
  *)
    echo "unknown STAGE=$STAGE (mine|dry|train|eval)" >&2
    exit 2
    ;;
esac

echo "=== done ==="
