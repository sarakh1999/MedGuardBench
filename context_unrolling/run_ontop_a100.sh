#!/usr/bin/env bash
# "On top of SFT" round for one model, run interactively on a single A100-80GB
# (Ascend OnDemand node). The listed arms train concurrently on the GPU; each
# then farms its 4-shard eval + metrics out to Pitzer (SBATCH_CLUSTERS).
#
#   bash context_unrolling/run_ontop_a100.sh MODEL_KEY INIT_ADAPTER OUT_ROOT ARM=DATA_ARM [ARM=DATA_ARM ...]
# e.g.
#   nohup bash context_unrolling/run_ontop_a100.sh qwen3-8b \
#     Claude/SFT/new_outputs/Qwen3-8B-Instruct/checkpoint-950 context_unrolling/outputs/ontop_8b \
#     sft+1ep_long=long sft+1ep_unrolled_long_user=unrolled_long__user__patient+prescription \
#     > logs/ontop_8b_a100.log 2>&1 &
set -uo pipefail
ROOT=/users/PCS0289/sarakhosravi/Guardrail
cd "$ROOT"
# shellcheck disable=SC1091
source Claude/SFT/blind/gpu_env.sh

export MODEL_KEY="${1:?MODEL_KEY}"
CK="${2:?INIT_ADAPTER}"
export OUT_ROOT="$ROOT/${3:?OUT_ROOT}"
shift 3
export SBATCH_CLUSTERS=pitzer
export SBATCH_ACCOUNT=PCS0289
mkdir -p "$OUT_ROOT" logs
tag=$(basename "$OUT_ROOT")
echo "=== $(hostname) $(date) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) model=$MODEL_KEY init=$CK arms=$*"

for spec in "$@"; do
  arm="${spec%%=*}"; data="${spec#*=}"
  (
    STAGE=train ARM="$arm" DATA_ARM="$data" INIT_ADAPTER="$CK" EPOCHS=1 \
      bash context_unrolling/run_arm.slurm > "logs/${tag}_train_${arm}.log" 2>&1
    echo "train $arm exit=$? $(date)"
  ) &
  sleep 90   # stagger model loads
done
wait
echo "=== all done $(date)"
