#!/usr/bin/env bash
# Create a conda env that has unsloth + trl (>= 0.15, for GRPOTrainer) + vllm
# together. None of the existing envs do:
#   unsloth_env : trl 0.12.2 (no GRPOTrainer), no vllm, torch 2.5.1
#   vllm_env    : vllm 0.19 but no unsloth / trl
#
# Run from a login node with internet access:
#   bash setup_grpo_env.sh            # creates ~/.conda/envs/grpo_env
#   GRPO_ENV=my_env bash setup_grpo_env.sh
#
# Unsloth's recommended install is `pip install unsloth vllm` on CUDA 12.x;
# it pulls a matching torch/trl/transformers set. If the cluster's driver is
# too old for the default torch wheel, pin torch first, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
# and re-run.

set -euo pipefail

GRPO_ENV="${GRPO_ENV:-grpo_env}"
PY="${PYTHON_VERSION:-3.11}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$GRPO_ENV"; then
  echo "env $GRPO_ENV already exists; activating and upgrading"
else
  conda create -y -n "$GRPO_ENV" "python=$PY"
fi
conda activate "$GRPO_ENV"

python -m pip install --upgrade pip
python -m pip install --upgrade unsloth vllm
# Unsloth pins a trl range it has been tested with; make sure it is new
# enough for GRPO and do not override Unsloth's pin.
python - <<'EOF'
import importlib.metadata as m
from packaging.version import Version
v = m.version("trl")
assert Version(v) >= Version("0.15.0"), f"trl {v} too old for GRPOTrainer"
for p in ("torch", "transformers", "trl", "unsloth", "unsloth_zoo", "vllm", "peft", "bitsandbytes", "datasets"):
    try:
        print(f"{p:14s} {m.version(p)}")
    except Exception:
        print(f"{p:14s} MISSING")
EOF

# CPU-only smoke test of the reward code (needs nothing but the stdlib)
cd "$(dirname "$0")"
python reward.py

echo
echo "Done. Use:  GRPO_ENV=$GRPO_ENV sbatch run_grpo.slurm"
echo "First GPU check:  sbatch --export=ALL,GRPO_ENV=$GRPO_ENV,STAGE=dry run_grpo.slurm"
