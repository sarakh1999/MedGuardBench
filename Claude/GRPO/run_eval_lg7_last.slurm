#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=lg7_eval_last
#SBATCH --clusters=ascend
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/grpo_smoke/slurm_lg7_eval_last_%j.log
#
# Evaluate the LAST Llama-Guard-7B GRPO checkpoint on the TEST split (no
# further training, no validation selection). Runs on a separate node from the
# live training on a0206. Output goes to a dedicated dir so the live run dir is
# never written to.
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

RUN=outputs/legacy_v1-llamaguard-7b/grpo-guided-v3
CKPT="$RUN/${CKPT_NAME:-checkpoint-450}"
OUTD="$RUN-evalsnap/${CKPT_NAME:-checkpoint-450}_test_eval"
mkdir -p "$OUTD"
echo "=== evaluating: $CKPT"
echo "=== out dir   : $OUTD"
[[ -f "$CKPT/adapter_config.json" ]] || { echo "!!! $CKPT missing"; exit 1; }

python eval_grpo.py --model "$CKPT" --load-in-4bit --bootstrap 2000 --out-dir "$OUTD" \
  || { echo "=== test eval failed $(date)"; exit 1; }
python per_category_report.py "$OUTD/test_predictions.jsonl" --label "GRPO(${CKPT_NAME:-checkpoint-450})" \
  > "$OUTD/per_category_grpo.txt" 2>/dev/null || true

echo "=== done $(date) ==="
echo "=== TEST metrics in: $OUTD ==="
