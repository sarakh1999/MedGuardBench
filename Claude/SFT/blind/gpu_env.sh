# Source this on an OSC Pitzer GPU node before running SFT / eval scripts.
#   source Claude/SFT/gpu_env.sh
#
# vllm_env carries torch 2.10+cu128 (sm_70 is in its arch list, so the V100S
# works), transformers 4.57.6, unsloth 2026.9.x, trl 0.24, bitsandbytes 0.50.
# torch loads the system /lib64/libstdc++ first, which lacks GLIBCXX_3.4.30
# that the env's libzmq needs; putting the env's lib dir first fixes it.
source /apps/python/3.12/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"
conda activate vllm_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export HF_HOME="${HF_HOME:-/fs/scratch/PCS0289/${USER}/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd /users/PCS0289/sarakhosravi/Guardrail
