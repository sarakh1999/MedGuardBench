#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=lg7_eval_sofar
#SBATCH --clusters=ascend
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/grpo_smoke/slurm_lg7_eval_sofar_%j.log
#
# Evaluate the BEST Llama-Guard-7B GRPO checkpoint SO FAR (no further training).
# Runs on a separate node from the live training on a0206. Operates on a
# symlink SNAPSHOT of the completed checkpoints (50..400) so it never touches
# the live run dir and is unaffected by new checkpoints being written.
#   1. select_grpo_checkpoint.py over the snapshot on the VALIDATION split
#      (--force-copy: always materialise the best GRPO checkpoint for eval)
#   2. eval_grpo.py of that checkpoint on the TEST split, compared to merged SFT
set -uo pipefail
echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-?}  $(date) ==="
nvidia-smi -L || true
source /apps/python/3.12/etc/profile.d/conda.sh; conda activate grpo_env
cd /users/PCS0289/sarakhosravi/Guardrail/Claude/GRPO
unset BNB_CUDA_VERSION || true
export MGB_MODEL=llamaguard-7b MGB_DATA_CONDITION=legacy_v1 MGB_GRPO_RECIPE=v3
export MGB_LOAD_IN_4BIT=1 VLLM_GPU_MEMORY_UTILIZATION=0.40
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export CC=/usr/bin/gcc CXX=/usr/bin/g++ UNSLOTH_VLLM_NO_FLASHINFER=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SNAP=outputs/legacy_v1-llamaguard-7b/grpo-guided-v3-evalsnap
SFT=$(python -c "import grpo_config as C; print(C.SFT_MERGED_PATH)")
echo "=== snapshot : $SNAP"
echo "=== SFT      : $SFT"

# 1. pick the best checkpoint on validation (force-copy so we always get a
#    final_selected to evaluate, even if SFT wins the selection score).
python select_grpo_checkpoint.py --run-dir "$SNAP" --every 1 --load-in-4bit --force-copy || true

BEST="$SNAP/final_selected"
if [[ ! -f "$BEST/adapter_config.json" ]]; then
  echo "!!! no final_selected produced; cannot eval"; exit 1
fi
echo "=== best-on-val checkpoint: $(cat "$BEST/SELECTED_FROM" 2>/dev/null | head -1)"

# 2. evaluate that checkpoint on the TEST split vs the merged SFT policy.
python eval_grpo.py --model "$BEST" --compare-model "$SFT" --load-in-4bit --bootstrap 2000 \
  || { echo "=== test eval failed $(date)"; exit 1; }
python per_category_report.py "$BEST/eval/test_predictions.jsonl" --label "GRPO(so-far)" \
  > "$BEST/eval/per_category_grpo.txt" 2>/dev/null || true

echo "=== done $(date) ==="
echo "=== TEST metrics: $BEST/eval/ ==="
