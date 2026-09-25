"""
Quick diagnostic to figure out exactly how Qwen3Guard tokenizes the
<think></think> scaffold inside the chat template. Run this once to find
the right marker IDs, then we plug them into the SFT script.
"""
import torch
import os
import sys
import re
import json
import gc
import numpy as np
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_recall_curve, auc,
)

# ==============================================================================
# 1. CRITICAL HPC & BACKEND PATCHES
# ==============================================================================
import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    def register_constant(cls): return cls
    torch.utils._pytree.register_constant = register_constant

def patch_torch_dtypes():
    for i in range(1, 8):
        if not hasattr(torch, f"int{i}"): setattr(torch, f"int{i}", torch.int8)
        if not hasattr(torch, f"uint{i}"): setattr(torch, f"uint{i}", torch.uint8)
patch_torch_dtypes()

os.environ["BNB_CUDA_VERSION"] = "121"
os.environ["LD_LIBRARY_PATH"] = (
    "/apps/spack/0.21/ascend/linux-rhel9-zen2/cuda/gcc/11.4.1/12.4.1-rni5fqf/targets/x86_64-linux/lib:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

# ==============================================================================
# 2. IMPORTS
# ==============================================================================
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import EarlyStoppingCallback
from tqdm import tqdm

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "Qwen/Qwen3Guard-Gen-4B",
    max_seq_length = 2048,
    load_in_4bit   = True,
    device_map     = "auto",
)

# Look up special tokens by name
print("\n=== SPECIAL TOKEN IDS ===")
for tok in ["<think>", "</think>", "<|im_start|>", "<|im_end|>"]:
    tid = tokenizer.convert_tokens_to_ids(tok)
    print(f"  {tok!r:25s} -> id {tid}")

# Load one training example
train_ds = load_dataset("json", data_files="Claude/SFT/data_chatml/train.jsonl", split="train")
convo = train_ds[0]["messages"]

# Apply chat template
text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
ids = tokenizer(text, add_special_tokens=False)["input_ids"]

print(f"\n=== CHAT TEMPLATE TOKENIZATION ===")
print(f"Total tokens: {len(ids)}")

# Look up the </think> special token id
end_think_id = tokenizer.convert_tokens_to_ids("</think>")
print(f"\nLooking for </think> token id {end_think_id} in sequence...")

positions = [i for i, t in enumerate(ids) if t == end_think_id]
print(f"Found {len(positions)} occurrence(s) at positions: {positions}")

if positions:
    pos = positions[0]
    print(f"\nTokens around position {pos} (the </think>):")
    for offset in range(-3, 5):
        idx = pos + offset
        if 0 <= idx < len(ids):
            tid = ids[idx]
            decoded = tokenizer.decode([tid], skip_special_tokens=False)
            marker = " <-- </think>" if idx == pos else ""
            print(f"  [{idx:4d}] id={tid:6d}  {decoded!r}{marker}")

# Also try standalone tokenization for comparison
print(f"\n=== STANDALONE MARKER TOKENIZATION ===")
for marker in ["</think>", "</think>\n", "</think>\n\n", "\n</think>\n\n", "\n\n</think>\n\n"]:
    standalone = tokenizer(marker, add_special_tokens=False)["input_ids"]
    print(f"  {marker!r:30s} -> {standalone}")