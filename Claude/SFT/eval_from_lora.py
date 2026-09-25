"""
Evaluate from a LoRA checkpoint directly, bypassing the broken merged model.

The save_pretrained_merged step produced corrupted weights, but the
intermediate LoRA-adapter checkpoints saved every 50 steps are fine.
This script loads the base model + the LoRA adapter, generates predictions,
and writes them incrementally.

To run:
  1. Find your best checkpoint (typically the latest, but use whichever
     one early-stopping promoted to load_best_model_at_end -- usually
     marked in trainer_state.json):
       ls Claude/SFT/outputs/Qwen3-4B-Instruct/checkpoint-*
  2. Set CHECKPOINT_PATH below to that directory
  3. python Claude/SFT/eval_from_lora.py
"""

import os
import sys
import re
import json
import time
import torch
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_recall_curve, auc,
)

os.environ["LD_LIBRARY_PATH"] = (
    "/apps/spack/0.21/ascend/linux-rhel9-zen2/cuda/gcc/11.4.1/12.4.1-rni5fqf/targets/x86_64-linux/lib:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# ==============================================================================
# CONFIG
# ==============================================================================
BASE_MODEL    = "Qwen/Qwen3-4B-Instruct-2507"
CHECKPOINT_PATH = os.path.abspath(
    "Claude/SFT/outputs/Qwen3-4B-Instruct/checkpoint-528"  # <-- CHANGE TO YOUR BEST CHECKPOINT
)
VAL_JSONL     = os.path.abspath("Claude/SFT/data_chatml/val.jsonl")
OUT_JSONL     = "val_predictions.jsonl"
OUT_TXT       = "val_predictions.txt"
MAX_NEW_TOKENS = 1024

# ==============================================================================
# SANITY CHECKS
# ==============================================================================
if not os.path.isdir(CHECKPOINT_PATH):
    print(f"ERROR: checkpoint directory not found: {CHECKPOINT_PATH}")
    print(f"\nAvailable checkpoints:")
    parent = os.path.dirname(CHECKPOINT_PATH)
    if os.path.isdir(parent):
        for d in sorted(os.listdir(parent)):
            if d.startswith("checkpoint-"):
                print(f"  {os.path.join(parent, d)}")
    sys.exit(1)

# Look for adapter files to confirm this is a LoRA checkpoint, not a full model.
adapter_files = [f for f in os.listdir(CHECKPOINT_PATH)
                 if f.startswith("adapter") or "lora" in f.lower()]
if not adapter_files:
    print(f"WARNING: no adapter files found in {CHECKPOINT_PATH}")
    print(f"  Contents: {os.listdir(CHECKPOINT_PATH)}")
    print(f"  This may not be a LoRA checkpoint. Proceeding anyway.")

# ==============================================================================
# HELPERS
# ==============================================================================

def extract_is_safe(text):
    cleaned = text
    for noise in ("</tool_call>", "<tool_call>", "<think>", "</think>"):
        cleaned = cleaned.replace(noise, "")
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, dict) and "is_safe" in data:
                return bool(data["is_safe"])
        except json.JSONDecodeError:
            pass
    m = re.search(r'"is_safe"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    return None


def extract_reasoning(text):
    cleaned = text
    for noise in ("</tool_call>", "<tool_call>", "<think>", "</think>"):
        cleaned = cleaned.replace(noise, "")
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, dict) and "reasoning" in data:
                return str(data["reasoning"])
        except json.JSONDecodeError:
            pass
    return None


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_txt_block(path, record):
    lines = [
        "=" * 78,
        f"idx: {record['idx']}",
        f"gt_is_safe:    {record['gt_is_safe']}",
        f"pred_is_safe:  {record['pred_is_safe']}",
        f"parsed_ok:     {record['parsed_ok']}",
        f"correct:       {record['gt_is_safe'] == record['pred_is_safe']}",
        f"gen_seconds:   {record['gen_seconds']:.1f}",
        "",
        "-- predicted reasoning --",
        (record.get("pred_reasoning") or "(could not extract reasoning)").strip(),
        "",
        "-- raw model output (first 1200 chars) --",
        record["raw_response"][:1200],
        "",
    ]
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_completed_indices(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if "idx" in r:
                    done.add(int(r["idx"]))
            except json.JSONDecodeError:
                continue
    return done


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"Base model:     {BASE_MODEL}")
    print(f"LoRA adapter:   {CHECKPOINT_PATH}")
    print(f"Val data:       {VAL_JSONL}")

    print(f"\nLoading base model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16,
    ).to("cuda")

    print(f"Loading LoRA adapter from {CHECKPOINT_PATH}...")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()

    # Quick sanity check: generate one example before doing the full eval.
    print("\nSanity check generation on val example 0...")
    val_ds = load_dataset("json", data_files=VAL_JSONL, split="train")
    ex0 = val_ds[0]
    prompt_msgs = [m for m in ex0["messages"] if m["role"] != "assistant"]
    inputs = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs, max_new_tokens=200, do_sample=False, use_cache=True,
        )
    sample = tokenizer.decode(out[0][len(inputs[0]):], skip_special_tokens=False)
    print(f"  First 300 chars of generation:")
    print(f"  {sample[:300]!r}")
    if sample.startswith("<tool_call>") and "<tool_call>" in sample[:200]:
        print("  *** WARNING: sample starts with tool_call loop. ***")
        print("  Either this checkpoint is also broken, or LoRA didn't load. Exiting.")
        return
    print("  Looks like real output. Proceeding to full eval.")

    # Resume support
    done = load_completed_indices(OUT_JSONL)
    if done:
        print(f"\nFound {len(done)} already-completed samples; skipping them.")
    todo = [i for i in range(len(val_ds)) if i not in done]
    print(f"Will process {len(todo)} new samples")

    for idx in tqdm(todo, desc="Generating"):
        messages = val_ds[idx]["messages"]
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        gt_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        gt_safe = extract_is_safe(gt_msg)

        inputs = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to("cuda")

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, use_cache=True,
            )
        response = tokenizer.decode(
            out[0][len(inputs[0]):], skip_special_tokens=True,
        )
        gen_seconds = time.time() - t0

        pred_safe = extract_is_safe(response)
        parsed_ok = pred_safe is not None
        if not parsed_ok:
            pred_safe = True
        pred_reasoning = extract_reasoning(response)

        record = {
            "idx": int(idx),
            "gt_is_safe": gt_safe,
            "pred_is_safe": bool(pred_safe),
            "parsed_ok": bool(parsed_ok),
            "pred_reasoning": pred_reasoning,
            "raw_response": response,
            "gen_seconds": gen_seconds,
            "n_generated_tokens": len(out[0]) - len(inputs[0]),
        }
        append_jsonl(OUT_JSONL, record)
        append_txt_block(OUT_TXT, record)

    # -------- Summary --------
    print("\n" + "=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)
    all_records = []
    with open(OUT_JSONL) as f:
        for line in f:
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    y_true, y_pred = [], []
    n_unparseable = 0
    for r in all_records:
        if r["gt_is_safe"] is None:
            continue
        y_true.append(0 if r["gt_is_safe"] else 1)
        y_pred.append(0 if r["pred_is_safe"] else 1)
        if not r["parsed_ok"]:
            n_unparseable += 1

    if y_true:
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision, recall_curve, _ = precision_recall_curve(y_true, y_pred)
        auprc = auc(recall_curve, precision)
        print(f"  Total scored:               {len(y_true)}")
        print(f"  Accuracy:                   {acc:.4f}")
        print(f"  Recall (unsafe class):      {rec:.4f}")
        print(f"  F1 (unsafe class):          {f1:.4f}")
        print(f"  AUPRC (placeholder):        {auprc:.4f}")
        print(f"  Unparseable predictions:    {n_unparseable}/{len(all_records)}")


if __name__ == "__main__":
    main()