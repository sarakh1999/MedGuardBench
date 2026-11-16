#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=eval_mg4b_sofar
#SBATCH --clusters=ascend
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/Claude/GRPO/outputs/slurm_%x_%j.log
#
# Evaluate the BEST MedGemma-4B GRPO checkpoint produced SO FAR (while the main
# training job 8608903 is still running on a different A100). Non-destructive:
#   1. select_grpo_checkpoint.py scores every checkpoint-* on the VAL split and
#      picks the best one that meets the recall floor and beats the SFT ref.
#      (predictions are cached under val_select/, so the main run reuses them.)
#   2. eval_grpo.py runs the winner on the TEST split vs the merged SFT policy,
#      writing to grpo-guided-v3/eval_sofar/ (a separate dir, so it never
#      collides with the pipeline's own final_selected/eval step).
#
#   sbatch run_eval_medgemma4b_sofar.slurm

set -euo pipefail

export MGB_MODEL=medgemma-4b
export MGB_DATA_CONDITION=legacy_v1
export MGB_GRPO_RECIPE=v3
export MGB_LOAD_IN_4BIT=0
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
export HF_HOME="${HF_HOME:-/fs/scratch/PCS0289/sarakhosravi/huggingface}"
# MedGemma GRPO adapters have no vLLM load_lora hook, so eval_grpo.py pre-merges
# each adapter to full weights. Keep those (~7 GB each) on scratch, not $HOME.
export MGB_EVAL_MERGE_ROOT="${MGB_EVAL_MERGE_ROOT:-/fs/scratch/PCS0289/sarakhosravi/mgb_eval_merges}"

RUN="${RUN:-grpo-guided-v3}"
GRPO_ENV="${GRPO_ENV:-grpo_env}"

GRPO_DIR=/users/PCS0289/sarakhosravi/Guardrail/Claude/GRPO
cd "$GRPO_DIR"

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-?}  run: $RUN  $(date) ==="
nvidia-smi -L || true

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  source /apps/python/3.12/etc/profile.d/conda.sh
fi
conda activate "$GRPO_ENV"
python -c "import trl, unsloth, vllm; print('trl', trl.__version__, '| unsloth', unsloth.__version__, '| vllm', vllm.__version__)"

unset BNB_CUDA_VERSION || true
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export UNSLOTH_VLLM_NO_FLASHINFER=1

MERGED=$(python -c "import grpo_config as C; print(C.SFT_MERGED_PATH)")
OUT_ROOT=$(python -c "import grpo_config as C; print(C.GRPO_OUTPUT_DIR)")
OUT="$OUT_ROOT/$RUN"
echo "=== run dir   : $OUT"
echo "=== merged SFT: $MERGED"
ls -dt "$OUT"/checkpoint-* | sort -t- -k2 -n

# ---- 1. Score every checkpoint on validation, pick the best ----
# select_grpo_checkpoint.py exits 2 when GRPO does not beat SFT; don't abort.
echo "=== select best checkpoint on val ==="
python select_grpo_checkpoint.py --run-dir "$OUT" --every 2 || true

# ---- 2. Eval target = best GRPO checkpoint (final_selected if shipped) ----
TARGET=$(python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
sel = out / "final_selected"
if (sel / "adapter_config.json").exists():
    print(sel); raise SystemExit
rep = json.loads((out / "val_select" / "selection.json").read_text())
bg = rep.get("best_grpo")
cand = [c for c in rep["candidates"] if c["name"] == bg]
print(cand[0]["path"] if cand else "")
PY
)
if [[ -z "$TARGET" ]]; then
  echo "could not resolve an eval target from selection.json"; exit 1
fi
echo "=== eval target: $TARGET ==="

# ---- 3. Paired test eval vs merged SFT, into a dedicated eval_sofar dir ----
EVAL_DIR="$OUT/eval_sofar"
mkdir -p "$EVAL_DIR"
python eval_grpo.py \
  --model "$TARGET" \
  --compare-model "$MERGED" \
  --out-dir "$EVAL_DIR" \
  --bootstrap 2000

python per_category_report.py "$EVAL_DIR/test_predictions.jsonl" --label GRPO \
  > "$EVAL_DIR/per_category_grpo.txt" || true
python per_category_report.py "$EVAL_DIR/comparison_predictions.jsonl" --label SFT \
  > "$EVAL_DIR/per_category_sft.txt" || true

echo "=== done: $EVAL_DIR  $(date) ==="
