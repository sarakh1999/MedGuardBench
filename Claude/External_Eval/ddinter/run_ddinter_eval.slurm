#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=ddinter_eval
#SBATCH --clusters=ascend
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/Claude/External_Eval/ddinter/outputs/slurm_%x_%j.log
#
# External DDI evaluation on DDInter for the Qwen3-4B GRPO model (best arm on
# the 754-row in-distribution test) and its SFT init as comparison.
#
# Runs on Ascend (A100): grpo_env's torch/vLLM build has no sm_70 kernels, so
# it fails on Pitzer V100s with "no kernel image is available".
#
#   sbatch run_ddinter_eval.slurm
#   sbatch --export=ALL,MODELS="grpo" run_ddinter_eval.slurm      # one model only

set -euo pipefail

ROOT=/users/PCS0289/sarakhosravi/Guardrail
HERE=$ROOT/Claude/External_Eval/ddinter
GRPO_DIR=$ROOT/Claude/GRPO
TEST=$HERE/data/ddinter_test.jsonl
MODELS="${MODELS:-grpo sft}"

declare -A PATHS=(
  [grpo]=$GRPO_DIR/outputs/legacy_v1/grpo-guided-v2/final
  [sft]=$ROOT/Claude/SFT/new_outputs/Qwen3-4B-Instruct/checkpoint-950-merged
)

cd "$GRPO_DIR"
mkdir -p "$HERE/outputs"
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-?} ==="
nvidia-smi -L || true

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  source /apps/python/3.12/etc/profile.d/conda.sh
fi
conda activate "${GRPO_ENV:-grpo_env}"
unset BNB_CUDA_VERSION
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export UNSLOTH_VLLM_NO_FLASHINFER=1
export MGB_DATA_CONDITION=legacy_v1 MGB_MODEL=qwen3-4b

[[ -f "$TEST" ]] || python "$HERE/build_ddinter_test.py"

PREDS=()
LABELS=()
for m in $MODELS; do
  out="$HERE/outputs/$m/test_predictions.jsonl"
  if [[ ! -s "$out" ]]; then
    echo "=== generating: $m -> ${PATHS[$m]}"
    python eval_grpo.py --model "${PATHS[$m]}" --test-file "$TEST" \
      --predictions-out "$out" --generate-only
  else
    echo "=== reuse existing $out"
  fi
  PREDS+=("$out"); LABELS+=("$m")
done

python "$HERE/score_ddinter.py" --test "$TEST" --predictions "${PREDS[@]}" \
  --label "${LABELS[@]}" --bootstrap 2000 --out "$HERE/outputs/ddinter_metrics.json" \
  | tee "$HERE/outputs/ddinter_report.txt"
echo "=== done ==="
