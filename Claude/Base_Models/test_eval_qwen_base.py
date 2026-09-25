"""
Zero-shot baseline: evaluate the base Qwen3-4B-Instruct on the test set
WITHOUT any fine-tuning. Mirrors eval_unsloth_test.py exactly except for
the model load (no LoRA adapter applied).

This is the "does SFT actually help?" reference number. Expect lower
accuracy and much higher parse-failure rates than the fine-tuned model,
because the base model wasn't trained to emit your JSON schema.

Outputs:
  test_base_predictions.jsonl  (machine-readable, per-sample)
  test_base_predictions.txt    (human-readable, per-sample)
"""

import torch
import os
import sys
import re
import json
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_recall_curve, auc,
)

# ==============================================================================
# 1. HPC PATCHES
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
from datasets import load_dataset
from tqdm import tqdm

# ==============================================================================
# 3. CONFIG
# ==============================================================================
# Load the BASE model directly (no LoRA adapter). This is the same checkpoint
# the fine-tuned model was started from, so the SFT vs zero-shot comparison
# is apples-to-apples.
BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
TEST_JSONL      = os.path.abspath("Claude/SFT/data_chatml/test.jsonl")
OUT_JSONL       = "test_base_predictions.jsonl"
OUT_TXT         = "test_base_predictions.txt"
MAX_NEW_TOKENS  = 1024

if not os.path.isfile(TEST_JSONL):
    sys.exit(
        f"Test jsonl not found: {TEST_JSONL}\n\n"
        f"Generate it first by running:\n"
        f"  python Claude/SFT/convert_csv_to_chatml.py Claude/SFT/data_splits "
        f"Claude/SFT/data_chatml --include-test"
    )

# ==============================================================================
# 4. OUTPUT PARSING + I/O (identical to fine-tuned eval)
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
# 5. MAIN
# ==============================================================================

def main():
    print(f"Base model: {BASE_MODEL_NAME}  (no fine-tuning, no LoRA)")
    print(f"Test data:  {TEST_JSONL}")

    print(f"\nLoading base model via Unsloth...")
    # No LoRA adapter — just the raw base model. We pass the HF model id directly.
    # Unsloth will download the weights from HF if not cached locally.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL_NAME,
        max_seq_length = 2048,
        load_in_4bit   = True,
        device_map     = "auto",
    )
    FastLanguageModel.for_inference(model)
    model.eval()

    test_ds = load_dataset("json", data_files=TEST_JSONL, split="train")
    print(f"Test: {len(test_ds)} examples")

    # -------- Sanity check (informational, doesn't abort on failure) --------
    print("\nSanity check generation on test example 0...")
    ex0 = test_ds[0]
    prompt_msgs = [m for m in ex0["messages"] if m["role"] != "assistant"]
    inputs = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs, max_new_tokens=200,
            use_cache=True, do_sample=False,
        )
    sanity_time = time.time() - t0
    sample = tokenizer.decode(out[0][len(inputs[0]):], skip_special_tokens=False)
    print(f"  Generation took: {sanity_time:.1f}s for 200 tokens")
    print(f"  First 300 chars:")
    print(f"  {sample[:300]!r}")
    # Note: we DON'T abort on tool_call loops here. For the base model, weird
    # outputs are expected and we want them recorded in the metrics as failures.
    print("  Proceeding to full eval regardless of output quality.\n")

    # -------- Resume support --------
    done = load_completed_indices(OUT_JSONL)
    if done:
        print(f"Found {len(done)} already-completed samples; skipping them.")
    todo = [i for i in range(len(test_ds)) if i not in done]
    print(f"Will process {len(todo)} new samples")

    # -------- Full generation loop --------
    for idx in tqdm(todo, desc="Generating"):
        messages = test_ds[idx]["messages"]
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
                use_cache=True, do_sample=False,
            )
        gen_seconds = time.time() - t0
        response = tokenizer.decode(
            out[0][len(inputs[0]):], skip_special_tokens=True,
        )

        pred_safe = extract_is_safe(response)
        parsed_ok = pred_safe is not None
        if not parsed_ok:
            pred_safe = True  # conservative default when JSON parse fails
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
    print("BASE MODEL (ZERO-SHOT) TEST EVAL SUMMARY")
    print("=" * 70)
    all_records = []
    with open(OUT_JSONL) as f:
        for line in f:
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    y_true, y_pred, y_scores = [], [], []
    n_unparseable = 0
    n_skipped = 0
    for r in all_records:
        if r["gt_is_safe"] is None:
            n_skipped += 1
            continue
        y_true.append(0 if r["gt_is_safe"] else 1)
        y_pred.append(0 if r["pred_is_safe"] else 1)
        y_scores.append(0 if r["pred_is_safe"] else 1)
        if not r["parsed_ok"]:
            n_unparseable += 1

    if y_true:
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        auprc = auc(recall_curve, precision)
        print(f"  Total scored:               {len(y_true)}")
        print(f"  Accuracy:                   {acc:.4f}")
        print(f"  Recall (unsafe class):      {rec:.4f}")
        print(f"  F1 (unsafe class):          {f1:.4f}")
        print(f"  AUPRC (placeholder):        {auprc:.4f}")
        print(f"  Unparseable predictions:    {n_unparseable}/{len(all_records)}")
        print(f"  Skipped (bad ground truth): {n_skipped}")

    print(f"\nOutputs:")
    print(f"  {OUT_JSONL}  (machine-readable, per-sample)")
    print(f"  {OUT_TXT}   (human-readable, per-sample)")


if __name__ == "__main__":
    main()