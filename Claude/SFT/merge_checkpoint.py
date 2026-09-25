import torch
import os
import sys
import gc

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
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import EarlyStoppingCallback, DataCollatorForSeq2Seq


# Load the BEST checkpoint (Step 2900)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "medguard_outputs/checkpoint-2900",
    max_seq_length = 2048,
    load_in_4bit = True, # Use 4bit for loading to save memory
    device_map = "auto",
)

# Merge and save to 16bit for high-precision inference
model.save_pretrained_merged("medguard_final_merged", tokenizer, save_method = "merged_16bit")
print("Model merged and saved to 'medguard_final_merged'")