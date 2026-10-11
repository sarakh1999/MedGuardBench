#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=selfplay_sample
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/users/PCS0289/sarakhosravi/Guardrail/Claude/DPO/logs/slurm_%x_%j.log
#
# Stage 1 of self-play DPO (round 3): vLLM-sample N completions per train prompt
# from the merged SFT model, then build chosen=correct-self / rejected=wrong-self pairs.
#   sbatch -M ascend -p nextgen --export=ALL,MODEL_KEY=qwen3-8b Claude/DPO/run_selfplay_sample.slurm
# Env: MODEL_KEY (qwen3-8b|qwen3-4b)  N (6)  TEMP (0.8)  CAP (400)  GOLD_FRAC (0.25)

set -euo pipefail
ROOT=/users/PCS0289/sarakhosravi/Guardrail
DPO="$ROOT/Claude/DPO"
MODEL_KEY="${MODEL_KEY:-qwen3-8b}"
case "$MODEL_KEY" in
  qwen3-8b) SFT_RUN=Qwen3-8B-Instruct; TAG=8b ;;
  qwen3-4b) SFT_RUN=Qwen3-4B-Instruct; TAG=4b ;;
  *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
MERGED="$ROOT/Claude/SFT/new_outputs/$SFT_RUN/checkpoint-950-merged"
DATA="$DPO/data/selfplay_$TAG"
mkdir -p "$DATA"

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-?}  model: $MODEL_KEY ==="
nvidia-smi -L || true
source "$ROOT/Claude/SFT/blind/gpu_env.sh"

python "$DPO/sample_selfplay_vllm.py" \
  --model-dir "$MERGED" \
  --out "$DATA/train_samples.jsonl" \
  --n "${N:-6}" --temperature "${TEMP:-0.8}"

python "$DPO/build_selfplay_pairs.py" \
  --samples "$DATA/train_samples.jsonl" \
  --out "$DATA/dpo_pairs.jsonl" \
  --cap "${CAP:-400}" --gold-frac "${GOLD_FRAC:-0.25}" | tee "$DATA/build_report.txt"

echo "=== sample+build done ==="
