#!/usr/bin/env python3
"""
Merge an SFT LoRA checkpoint onto the fp16/bf16 Hugging Face base with plain
PEFT merge_and_unload. This is the merge grpo_config.SFT_MERGED_PATH expects
(the Unsloth merged_16bit path re-quantizes through bnb-4bit and produced
broken weights; see the note in grpo_config.py).

    python merge_ckpt.py --base Qwen/Qwen3-8B \
        --adapter Claude/SFT/new_outputs/Qwen3-8B-Instruct/checkpoint-950 \
        --out     Claude/SFT/new_outputs/Qwen3-8B-Instruct/checkpoint-950-merged

--base may be a hub id (resolved from the local HF cache) or a snapshot dir.
Tokenizer and chat template are copied from the adapter directory when it
has them, otherwise from the base.
"""
import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "config.json").exists() and any(out.glob("*.safetensors")):
        print(f"exists, skipping: {out}")
        return
    dtype = getattr(torch, a.dtype)
    print(f"loading base {a.base} ({a.dtype})")
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype, device_map="cpu",
                                                 low_cpu_mem_usage=True)
    print(f"applying adapter {a.adapter}")
    model = PeftModel.from_pretrained(model, a.adapter)
    model = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    tok_src = a.adapter if (Path(a.adapter) / "tokenizer_config.json").exists() else a.base
    try:
        tok = AutoTokenizer.from_pretrained(tok_src)
    except Exception as e:  # e.g. adapter tokenizer_config incompatible with this transformers
        print(f"adapter tokenizer failed ({e!r}); falling back to base tokenizer")
        tok = AutoTokenizer.from_pretrained(a.base)
    tok.save_pretrained(out)
    for extra in ("chat_template.jinja", "added_tokens.json", "special_tokens_map.json"):
        p = Path(a.adapter) / extra
        if p.exists() and not (out / extra).exists():
            shutil.copy(p, out / extra)
    print(f"merged -> {out}")


if __name__ == "__main__":
    main()
