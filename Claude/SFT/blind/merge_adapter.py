"""
Merge a LoRA adapter produced by sft_train.py into 16-bit base weights.

GRPO and DPO must start from the MERGED SFT model (see grpo_config.py), and
sft_train.py only writes the merged copy when run with --merge. This does it
after the fact for any adapter or checkpoint directory.

Usage (GPU node):
    source Claude/SFT/gpu_env.sh
    python Claude/SFT/merge_adapter.py --model qwen3-4b \
        --adapter Claude/SFT/outputs_clean/v1_cleaned/qwen3-4b/final_adapter
    # -> Claude/SFT/outputs_clean/v1_cleaned/qwen3-4b/final_merged
"""

import argparse
import os
import sys

import torch
import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda cls: cls
for _i in range(1, 8):
    if not hasattr(torch, f"int{_i}"):
        setattr(torch, f"int{_i}", torch.int8)
    if not hasattr(torch, f"uint{_i}"):
        setattr(torch, f"uint{_i}", torch.uint8)

from unsloth import FastLanguageModel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_train import MODELS, apply_template  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default=None, help="default: <adapter parent>/final_merged")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    args = ap.parse_args()

    out = args.out or os.path.join(os.path.dirname(args.adapter.rstrip("/")), "final_merged")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter, max_seq_length=args.max_seq_length,
        load_in_4bit=True, device_map="auto")
    apply_template(tokenizer, MODELS[args.model])
    model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")
    print(f"merged -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
