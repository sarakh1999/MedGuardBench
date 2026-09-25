"""Find out what Qwen3Guard's chat template actually does with the
assistant message. We expect the JSON output to appear in the rendered
text; the previous diagnostic (debug_tokens.py) suggests it doesn't."""
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
    model_name="Qwen/Qwen3Guard-Gen-4B",
    max_seq_length=2048, load_in_4bit=True, device_map="auto",
)

train_ds = load_dataset("json", data_files="Claude/SFT/data_chatml/train.jsonl", split="train")
convo = train_ds[0]["messages"]

# Show what the raw assistant content actually contains
assistant_msg = next(m for m in convo if m["role"] == "assistant")
print("=== RAW ASSISTANT MESSAGE FROM JSONL ===")
print(f"Length: {len(assistant_msg['content'])} chars")
print(f"First 300 chars:\n{assistant_msg['content'][:300]}")
print(f"Last 300 chars:\n{assistant_msg['content'][-300:]}")

# Apply chat template and look at the END of the rendered text
text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
print(f"\n=== RENDERED TEMPLATE ===")
print(f"Total length: {len(text)} chars")
print(f"Last 600 chars of rendered template:")
print(repr(text[-600:]))

# Does the assistant JSON appear anywhere in the rendered text?
json_snippet = assistant_msg['content'][:100]
print(f"\n=== DOES ASSISTANT CONTENT APPEAR? ===")
print(f"Looking for first 100 chars of assistant content in rendered text...")
if json_snippet in text:
    pos = text.index(json_snippet)
    print(f"YES, found at character position {pos} of {len(text)}")
    print(f"Context (50 chars before, 100 after):")
    print(repr(text[max(0, pos-50):pos+100]))
else:
    print(f"NO -- the assistant content was DROPPED by the chat template.")
    print(f"This is the bug.")

# Show the chat_template source
print(f"\n=== CHAT TEMPLATE (first 1500 chars) ===")
ct = getattr(tokenizer, "chat_template", None)
if ct:
    print(ct[:1500])
else:
    print("No chat_template attribute")