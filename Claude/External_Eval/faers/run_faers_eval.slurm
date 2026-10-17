#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=faers_eval
#SBATCH --clusters=ascend
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/Claude/External_Eval/faers/outputs/slurm_%x_%j.log
#
# Real-patient (FAERS) evaluation: untrained base Qwen3-4B-Instruct vs SFT vs
# GRPO vs context-unrolling arm, all greedy through Claude/GRPO/eval_grpo.py.
#
#   sbatch run_faers_eval.slurm
#   sbatch --export=ALL,MODELS="base grpo" run_faers_eval.slurm

set -euo pipefail

ROOT=/users/PCS0289/sarakhosravi/Guardrail
HERE=$ROOT/Claude/External_Eval/faers
GRPO_DIR=$ROOT/Claude/GRPO
TEST=$HERE/data/faers_test.jsonl
MODELS="${MODELS:-base sft grpo cu}"

declare -A PATHS=(
  [base]=/fs/scratch/PCS0289/sarakhosravi/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554/
  [sft]=$ROOT/Claude/SFT/new_outputs/Qwen3-4B-Instruct/checkpoint-950-merged
  [grpo]=$GRPO_DIR/outputs/legacy_v1/grpo-guided-v2/final
  [cu]=$ROOT/context_unrolling/outputs/ontop/sft+1ep_unrolled_v4pro/final_adapter
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
export HF_HOME="${HF_HOME:-/fs/scratch/PCS0289/${USER}/huggingface}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export UNSLOTH_VLLM_NO_FLASHINFER=1
export MGB_DATA_CONDITION=legacy_v1 MGB_MODEL=qwen3-4b

[[ -f "$TEST" ]] || { echo "missing $TEST; run build_faers_test.py first"; exit 1; }

PREDS=(); LABELS=()
for m in $MODELS; do
  out="$HERE/outputs/$m/test_predictions.jsonl"
  if [[ ! -s "$out" ]]; then
    echo "=== generating: $m -> ${PATHS[$m]}"
    python eval_grpo.py --model "${PATHS[$m]}" --test-file "$TEST" \
      --predictions-out "$out" --generate-only || { echo "!!! $m failed"; continue; }
  else
    echo "=== reuse existing $out"
  fi
  PREDS+=("$out"); LABELS+=("$m")
done

python "$HERE/score_faers.py" --test "$TEST" --predictions "${PREDS[@]}" \
  --label "${LABELS[@]}" --bootstrap 2000 --out "$HERE/outputs/faers_metrics.json"
echo "=== done ==="
