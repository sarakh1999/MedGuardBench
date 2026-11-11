#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=repair_cu_parse
#SBATCH --clusters=ascend
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --exclude=a0233
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/Claude/External_Eval/repair_unparsed_%j.log
#
# Regenerate only the unparsable Qwen3-8B CU and Qwen3Guard-8B CU rows.
# Parsed completions are not touched.

set -euo pipefail

ROOT=/users/PCS0289/sarakhosravi/Guardrail
EXT=$ROOT/Claude/External_Eval
GRPO=$ROOT/Claude/GRPO

cd "$GRPO"
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-?} ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
nvidia-smi || true
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  source /apps/python/3.12/etc/profile.d/conda.sh
fi
conda activate "${GRPO_ENV:-grpo_env}"
python -c "import torch; print('torch', torch.__version__, 'devices', torch.cuda.device_count()); x=torch.zeros(1, device='cuda'); print('cuda ok', x.device)"
unset BNB_CUDA_VERSION
export HF_HOME="${HF_HOME:-/fs/scratch/PCS0289/${USER}/huggingface}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export UNSLOTH_VLLM_NO_FLASHINFER=1
export MGB_DATA_CONDITION=legacy_v1

FOURBIT=()
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 || echo 40000)
if [[ ${VRAM_MB%.*} -lt 60000 ]]; then
  echo "VRAM ${VRAM_MB}MB: --load-in-4bit"
  FOURBIT=(--load-in-4bit)
fi

echo "=== Qwen3-8B CU ==="
python "$EXT/repair_unparsed.py" \
  --model "$ROOT/context_unrolling/outputs/ontop_8b/sft+1ep_unrolled_v4pro/final_adapter" \
  "${FOURBIT[@]}" \
  --pair "$EXT/faers/data/faers_test.jsonl" "$EXT/cu8b/outputs/faers/unrolled_self/test_predictions.jsonl" \
  --pair "$EXT/nhanes/data/nhanes_test.jsonl" "$EXT/nhanes/outputs/8b_cu_self/test_predictions.jsonl" \
  --pair "$EXT/mimic_demo/data/mimic_demo_test.jsonl" "$EXT/mimic_demo/outputs/8b_cu_self/test_predictions.jsonl" \
  --pair "$EXT/ddinter/data/ddinter_test.jsonl" "$EXT/cu8b/outputs/ddinter/unrolled_self/test_predictions.jsonl" \
  || echo "Qwen3-8B CU still has unparsed rows"

echo "=== Qwen3Guard-8B CU ==="
# Guard SFT targets start at '{'. The default legacy prefix is the Qwen3
# <think> scaffold, and ChatML cannot accept it. Empty prefix matches training.
export MGB_ASSISTANT_PREFIX=
python "$EXT/repair_unparsed.py" \
  --model "$EXT/guard8b/models/qwen3guard-8b-cu-self" \
  "${FOURBIT[@]}" \
  --pair "$EXT/faers/data/faers_test.jsonl" "$EXT/guard8b/outputs/faers/cu_self/test_predictions.jsonl" \
  --pair "$EXT/nhanes/data/nhanes_test.jsonl" "$EXT/guard8b/outputs/nhanes/cu_self/test_predictions.jsonl" \
  --pair "$EXT/mimic_demo/data/mimic_demo_test.jsonl" "$EXT/guard8b/outputs/mimic/cu_self/test_predictions.jsonl" \
  --pair "$EXT/ddinter/data/ddinter_test.jsonl" "$EXT/guard8b/outputs/ddinter/cu_self/test_predictions.jsonl" \
  || echo "Qwen3Guard-8B CU still has unparsed rows"

echo "=== done ==="
