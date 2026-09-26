"""
SFT model test eval, aligned with eval_base_schema_test_v3.py for fair
comparison. Same generation params, same parsing, same metrics.

Outputs:
  test_sft_v3_predictions.jsonl
  test_sft_v3_predictions.txt
"""

import torch
import os
import sys
import re
import json
import time

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

from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm

# ==============================================================================
# 2. CONFIG (aligned with base v3)
# ==============================================================================
CHECKPOINT_PATH      = os.path.abspath("Claude/SFT/new_outputs/Qwen3-8B-Instruct/checkpoint-950")
TEST_JSONL           = os.path.abspath("Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl")
RISK_CATEGORIES_FILE = os.path.abspath("risk_categories.txt")
# OUT_JSONL            = "test_sft_v3_predictions.jsonl"
OUT_JSONL            = "Claude/SFT/new_outputs/Qwen3-8B-Instruct/test_predictions_w_schema.jsonl"
# OUT_TXT              = "test_sft_v3_predictions.txt"
OUT_TXT              = "Claude/SFT/new_outputs/Qwen3-8B-Instruct/test_predictions_w_schema.txt"

MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 2048

if not os.path.isdir(CHECKPOINT_PATH):
    sys.exit(f"Checkpoint not found: {CHECKPOINT_PATH}")
if not os.path.isfile(TEST_JSONL):
    sys.exit(f"Test jsonl not found: {TEST_JSONL}")
if not os.path.isfile(RISK_CATEGORIES_FILE):
    sys.exit(f"Risk categories file not found: {RISK_CATEGORIES_FILE}")


def load_risk_categories(path):
    cats = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                cats.append(line)
    return cats


# ==============================================================================
# 3. PARSING (identical to base v3 — guarantees apples-to-apples metrics)
# ==============================================================================

def _strip_noise(text):
    cleaned = text
    for noise in ("</tool_call>", "<tool_call>", "<think>", "</think>"):
        cleaned = cleaned.replace(noise, "")
    return cleaned


def extract_json_dict(text):
    cleaned = _strip_noise(text)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


def recover_partial_risk_analysis(text, categories):
    cleaned = _strip_noise(text)
    out = {c: False for c in categories}
    n_found = 0
    for c in categories:
        for variant in (c, c.replace("-", "\u2013")):
            esc = re.escape(variant)
            pat = rf'"{esc}"\s*:\s*(true|false)'
            m = re.search(pat, cleaned, re.IGNORECASE)
            if m:
                out[c] = m.group(1).lower() == "true"
                n_found += 1
                break
    return out, n_found


def extract_is_safe(text):
    parsed = extract_json_dict(text)
    if parsed and "is_safe" in parsed:
        return bool(parsed["is_safe"])
    m = re.search(r'"is_safe"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    return None


def extract_reasoning(text):
    parsed = extract_json_dict(text)
    if parsed and "reasoning" in parsed:
        return str(parsed["reasoning"])
    m = re.search(r'"reasoning"\s*:\s*"(.*?)(?<!\\)"', text, re.DOTALL)
    if m:
        return m.group(1)
    return None


def extract_risk_analysis(text, categories):
    parsed = extract_json_dict(text)
    if parsed and "risk_analysis" in parsed and isinstance(parsed["risk_analysis"], dict):
        ra_raw = parsed["risk_analysis"]
        out = {}
        for c in categories:
            if c in ra_raw:
                out[c] = bool(ra_raw[c])
            else:
                alt = c.replace("-", "\u2013")
                out[c] = bool(ra_raw.get(alt, False))
        return out, True, len(categories)
    out, n_found = recover_partial_risk_analysis(text, categories)
    return out, False, n_found


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_txt_block(path, record):
    lines = [
        "=" * 78,
        f"idx: {record['idx']}",
        f"gt_is_safe:      {record['gt_is_safe']}",
        f"pred_is_safe:    {record['pred_is_safe']}",
        f"parsed_ok:       {record['parsed_ok']}",
        f"ra_parsed_ok:    {record['ra_parsed_ok']}",
        f"ra_recovered:    {record['ra_n_recovered']}/17",
        f"correct:         {record['gt_is_safe'] == record['pred_is_safe']}",
        f"gen_seconds:     {record['gen_seconds']:.1f}",
        f"n_gen_tokens:    {record['n_generated_tokens']}",
        "",
        "-- predicted reasoning --",
        (record.get("pred_reasoning") or "(could not extract reasoning)").strip(),
        "",
        "-- raw model output (first 1500 chars) --",
        record["raw_response"][:1500],
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
# 4. MAIN
# ==============================================================================

def main():
    categories = load_risk_categories(RISK_CATEGORIES_FILE)
    print(f"Loaded {len(categories)} risk categories")

    print(f"Checkpoint:     {CHECKPOINT_PATH}")
    print(f"Test data:      {TEST_JSONL}")
    print(f"max_seq_length: {MAX_SEQ_LENGTH}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")

    print(f"\nLoading SFT LoRA adapter via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = CHECKPOINT_PATH,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit   = True,
        device_map     = "auto",
    )
    FastLanguageModel.for_inference(model)
    model.eval()

    test_ds = load_dataset("json", data_files=TEST_JSONL, split="train")
    print(f"Test: {len(test_ds)} examples")

    # NOTE: For the SFT model, we use the prompts already in test.jsonl
    # (which contain the original SFT system prompt). We do NOT add the
    # 17-category list — the SFT model learned the schema from training.

    # -------- Sanity check --------
    print("\nSanity check on test example 0...")
    ex0 = test_ds[0]
    prompt_msgs = [m for m in ex0["messages"] if m["role"] != "assistant"]
    inputs = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to("cuda")
    print(f"  Prompt length: {inputs.shape[1]} tokens")
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
    if sample.lstrip().startswith("<tool_call>") and "<tool_call>" in sample[:200] and "{" not in sample[:200]:
        print("  *** WARNING: tool_call loop. Aborting. ***")
        return
    print("  Looks like real output. Proceeding to full eval.\n")

    # -------- Resume support --------
    done = load_completed_indices(OUT_JSONL)
    if done:
        print(f"Found {len(done)} already-completed samples; skipping.")
    todo = [i for i in range(len(test_ds)) if i not in done]
    print(f"Will process {len(todo)} new samples\n")

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
            pred_safe = True
        pred_reasoning = extract_reasoning(response)
        pred_ra, ra_parsed_ok, ra_n_recovered = extract_risk_analysis(response, categories)

        record = {
            "idx": int(idx),
            "gt_is_safe": gt_safe,
            "pred_is_safe": bool(pred_safe),
            "parsed_ok": bool(parsed_ok),
            "ra_parsed_ok": bool(ra_parsed_ok),
            "ra_n_recovered": int(ra_n_recovered),
            "pred_reasoning": pred_reasoning,
            "pred_risk_analysis": pred_ra,
            "raw_response": response,
            "gen_seconds": gen_seconds,
            "n_generated_tokens": len(out[0]) - len(inputs[0]),
        }
        append_jsonl(OUT_JSONL, record)
        append_txt_block(OUT_TXT, record)

    # -------- Summary (identical format to base v3) --------
    from sklearn.metrics import (
        accuracy_score, recall_score, f1_score, precision_score,
        precision_recall_curve, auc,
    )

    print("\n" + "=" * 70)
    print(f"SFT MODEL (v3, FAIR) TEST EVAL SUMMARY")
    print(f"Checkpoint:     {CHECKPOINT_PATH}")
    print(f"max_seq_length: {MAX_SEQ_LENGTH}  max_new_tokens: {MAX_NEW_TOKENS}")
    print("=" * 70)

    all_records = []
    with open(OUT_JSONL) as f:
        for line in f:
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    y_true, y_pred = [], []
    n_is_safe_unparseable = 0
    n_ra_fully_parsed = 0
    n_ra_partially_recovered = 0
    n_ra_completely_failed = 0
    tokens_at_max = 0
    for r in all_records:
        if r["gt_is_safe"] is None:
            continue
        y_true.append(0 if r["gt_is_safe"] else 1)
        y_pred.append(0 if r["pred_is_safe"] else 1)
        if not r["parsed_ok"]:
            n_is_safe_unparseable += 1
        if r.get("ra_parsed_ok"):
            n_ra_fully_parsed += 1
        elif r.get("ra_n_recovered", 0) > 0:
            n_ra_partially_recovered += 1
        else:
            n_ra_completely_failed += 1
        if r.get("n_generated_tokens", 0) >= MAX_NEW_TOKENS:
            tokens_at_max += 1

    if y_true:
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision, recall_curve, _ = precision_recall_curve(y_true, y_pred)
        auprc = auc(recall_curve, precision)
        print(f"\nIS_SAFE METRICS:")
        print(f"  Total scored:               {len(y_true)}")
        print(f"  Accuracy:                   {acc:.4f}")
        print(f"  Recall (unsafe class):      {rec:.4f}")
        print(f"  F1 (unsafe class):          {f1:.4f}")
        print(f"  AUPRC (placeholder):        {auprc:.4f}")

        print(f"\nOUTPUT STRUCTURE HEALTH:")
        print(f"  is_safe unparseable:           {n_is_safe_unparseable}/{len(all_records)}")
        print(f"  risk_analysis fully parsed:    {n_ra_fully_parsed}/{len(all_records)}")
        print(f"  risk_analysis partial recov:   {n_ra_partially_recovered}/{len(all_records)}")
        print(f"  risk_analysis completely fail: {n_ra_completely_failed}/{len(all_records)}")
        print(f"  Hit MAX_NEW_TOKENS cap:        {tokens_at_max}/{len(all_records)}")

    # ------ Per-category metrics ------
    print(f"\nPER-CATEGORY METRICS:")
    print(f"{'Category':<42s}  {'Support':>8s}  {'Prec':>6s}  {'Recall':>7s}  {'F1':>6s}")
    print("-" * 80)

    y_true_per_cat = {c: [] for c in categories}
    y_pred_per_cat = {c: [] for c in categories}
    exact_matches = 0
    n_compared = 0

    for r in all_records:
        if r["gt_is_safe"] is None:
            continue
        gt_assistant = next(
            (m["content"] for m in test_ds[r["idx"]]["messages"] if m["role"] == "assistant"),
            "",
        )
        gt_ra, gt_ok, _ = extract_risk_analysis(gt_assistant, categories)
        if not gt_ok:
            continue
        pred_ra = r.get("pred_risk_analysis") or {c: False for c in categories}
        n_compared += 1
        all_match = True
        for c in categories:
            y_true_per_cat[c].append(int(gt_ra[c]))
            y_pred_per_cat[c].append(int(pred_ra.get(c, False)))
            if gt_ra[c] != pred_ra.get(c, False):
                all_match = False
        if all_match:
            exact_matches += 1

    per_cat_f1 = []
    pooled_yt, pooled_yp = [], []
    for c in categories:
        yt = y_true_per_cat[c]
        yp = y_pred_per_cat[c]
        support = sum(yt)
        pooled_yt.extend(yt)
        pooled_yp.extend(yp)
        if support == 0:
            print(f"{c:<42s}  {support:>8d}    --        --       --")
            continue
        p = precision_score(yt, yp, zero_division=0)
        rr = recall_score(yt, yp, zero_division=0)
        f = f1_score(yt, yp, zero_division=0)
        per_cat_f1.append(f)
        print(f"{c:<42s}  {support:>8d}  {p:>6.3f}  {rr:>7.3f}  {f:>6.3f}")

    print("-" * 80)
    if per_cat_f1:
        macro_f1 = sum(per_cat_f1) / len(per_cat_f1)
        print(f"  Macro F1: {macro_f1:.4f}")
    if pooled_yt:
        micro_p = precision_score(pooled_yt, pooled_yp, zero_division=0)
        micro_r = recall_score(pooled_yt, pooled_yp, zero_division=0)
        micro_f = f1_score(pooled_yt, pooled_yp, zero_division=0)
        print(f"  Micro Precision: {micro_p:.4f}")
        print(f"  Micro Recall:    {micro_r:.4f}")
        print(f"  Micro F1:        {micro_f:.4f}")
    if n_compared > 0:
        print(f"\n  Exact-match accuracy: {exact_matches}/{n_compared} = "
              f"{exact_matches/n_compared:.4f}")


if __name__ == "__main__":
    main()