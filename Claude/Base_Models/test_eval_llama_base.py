# """
# Zero-shot baseline: evaluate Llama-3.1-8B-Instruct (base, no fine-tuning)
# on the test set.

# Mirrors eval_base_test.py but swaps the base model from Qwen3-4B-Instruct
# to Llama-3.1-8B-Instruct. Comparing this number against your fine-tuned
# Qwen result tells you "did our SFT win, or could a bigger zero-shot model
# have matched it?"

# Outputs:
#   test_base_llama_predictions.jsonl
#   test_base_llama_predictions.txt

# Prerequisite: HuggingFace access to meta-llama/Llama-3.1-8B-Instruct.
#   1. Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct,
#      accept the license (usually approved within minutes).
#   2. Get a HF token from https://huggingface.co/settings/tokens (read access).
#   3. Run `huggingface-cli login` on the HPC and paste the token.
# """

# import torch
# import os
# import sys
# import re
# import json
# import time
# import numpy as np
# from sklearn.metrics import (
#     accuracy_score, recall_score, f1_score,
#     precision_recall_curve, auc,
# )

# # ==============================================================================
# # 1. HPC PATCHES
# # ==============================================================================
# import torch.utils._pytree
# if not hasattr(torch.utils._pytree, "register_constant"):
#     def register_constant(cls): return cls
#     torch.utils._pytree.register_constant = register_constant

# def patch_torch_dtypes():
#     for i in range(1, 8):
#         if not hasattr(torch, f"int{i}"): setattr(torch, f"int{i}", torch.int8)
#         if not hasattr(torch, f"uint{i}"): setattr(torch, f"uint{i}", torch.uint8)
# patch_torch_dtypes()

# os.environ["BNB_CUDA_VERSION"] = "121"
# os.environ["LD_LIBRARY_PATH"] = (
#     "/apps/spack/0.21/ascend/linux-rhel9-zen2/cuda/gcc/11.4.1/12.4.1-rni5fqf/targets/x86_64-linux/lib:"
#     + os.environ.get("LD_LIBRARY_PATH", "")
# )

# # ==============================================================================
# # 2. IMPORTS
# # ==============================================================================
# from unsloth import FastLanguageModel
# from datasets import load_dataset
# from tqdm import tqdm

# # ==============================================================================
# # 3. CONFIG
# # ==============================================================================
# # Llama-3.1-8B-Instruct, 4-bit quantized. Unsloth pre-quantizes some popular
# # models — using the "unsloth/" version saves a few minutes on first load.
# # If you prefer the official Meta repo:
# #   BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# # Both load identically once cached.
# BASE_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# TEST_JSONL      = os.path.abspath("Claude/SFT/data_chatml/test.jsonl")
# OUT_JSONL       = "test_base_llama_predictions.jsonl"
# OUT_TXT         = "test_base_llama_predictions.txt"
# MAX_NEW_TOKENS  = 1024

# if not os.path.isfile(TEST_JSONL):
#     sys.exit(
#         f"Test jsonl not found: {TEST_JSONL}\n\n"
#         f"Generate it first by running:\n"
#         f"  python Claude/SFT/convert_csv_to_chatml.py Claude/SFT/data_splits "
#         f"Claude/SFT/data_chatml --include-test"
#     )

# # ==============================================================================
# # 4. OUTPUT PARSING + I/O
# # ==============================================================================

# def extract_is_safe(text):
#     cleaned = text
#     for noise in ("</tool_call>", "<tool_call>", "<think>", "</think>"):
#         cleaned = cleaned.replace(noise, "")
#     m = re.search(r"\{.*\}", cleaned, re.DOTALL)
#     if m:
#         try:
#             data = json.loads(m.group())
#             if isinstance(data, dict) and "is_safe" in data:
#                 return bool(data["is_safe"])
#         except json.JSONDecodeError:
#             pass
#     m = re.search(r'"is_safe"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
#     if m:
#         return m.group(1).lower() == "true"
#     return None


# def extract_reasoning(text):
#     cleaned = text
#     for noise in ("</tool_call>", "<tool_call>", "<think>", "</think>"):
#         cleaned = cleaned.replace(noise, "")
#     m = re.search(r"\{.*\}", cleaned, re.DOTALL)
#     if m:
#         try:
#             data = json.loads(m.group())
#             if isinstance(data, dict) and "reasoning" in data:
#                 return str(data["reasoning"])
#         except json.JSONDecodeError:
#             pass
#     return None


# def append_jsonl(path, record):
#     with open(path, "a") as f:
#         f.write(json.dumps(record) + "\n")
#         f.flush()
#         os.fsync(f.fileno())


# def append_txt_block(path, record):
#     lines = [
#         "=" * 78,
#         f"idx: {record['idx']}",
#         f"gt_is_safe:    {record['gt_is_safe']}",
#         f"pred_is_safe:  {record['pred_is_safe']}",
#         f"parsed_ok:     {record['parsed_ok']}",
#         f"correct:       {record['gt_is_safe'] == record['pred_is_safe']}",
#         f"gen_seconds:   {record['gen_seconds']:.1f}",
#         "",
#         "-- predicted reasoning --",
#         (record.get("pred_reasoning") or "(could not extract reasoning)").strip(),
#         "",
#         "-- raw model output (first 1200 chars) --",
#         record["raw_response"][:1200],
#         "",
#     ]
#     with open(path, "a") as f:
#         f.write("\n".join(lines) + "\n")
#         f.flush()
#         os.fsync(f.fileno())


# def load_completed_indices(path):
#     if not os.path.exists(path):
#         return set()
#     done = set()
#     with open(path) as f:
#         for line in f:
#             try:
#                 r = json.loads(line)
#                 if "idx" in r:
#                     done.add(int(r["idx"]))
#             except json.JSONDecodeError:
#                 continue
#     return done


# # ==============================================================================
# # 5. MAIN
# # ==============================================================================

# def main():
#     print(f"Base model: {BASE_MODEL_NAME}  (no fine-tuning, no LoRA)")
#     print(f"Test data:  {TEST_JSONL}")

#     print(f"\nLoading base model via Unsloth...")
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name     = BASE_MODEL_NAME,
#         max_seq_length = 2048,
#         load_in_4bit   = True,
#         device_map     = "auto",
#     )
#     FastLanguageModel.for_inference(model)
#     model.eval()

#     test_ds = load_dataset("json", data_files=TEST_JSONL, split="train")
#     print(f"Test: {len(test_ds)} examples")

#     # -------- Sanity check (informational, doesn't abort) --------
#     print("\nSanity check generation on test example 0...")
#     ex0 = test_ds[0]
#     prompt_msgs = [m for m in ex0["messages"] if m["role"] != "assistant"]
#     inputs = tokenizer.apply_chat_template(
#         prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
#     ).to("cuda")
#     t0 = time.time()
#     with torch.no_grad():
#         out = model.generate(
#             input_ids=inputs, max_new_tokens=200,
#             use_cache=True, do_sample=False,
#         )
#     sanity_time = time.time() - t0
#     sample = tokenizer.decode(out[0][len(inputs[0]):], skip_special_tokens=False)
#     print(f"  Generation took: {sanity_time:.1f}s for 200 tokens")
#     print(f"  First 300 chars:")
#     print(f"  {sample[:300]!r}")
#     print("  Proceeding to full eval regardless of output quality.\n")

#     # -------- Resume support --------
#     done = load_completed_indices(OUT_JSONL)
#     if done:
#         print(f"Found {len(done)} already-completed samples; skipping them.")
#     todo = [i for i in range(len(test_ds)) if i not in done]
#     print(f"Will process {len(todo)} new samples")

#     # -------- Full generation loop --------
#     for idx in tqdm(todo, desc="Generating"):
#         messages = test_ds[idx]["messages"]
#         prompt_msgs = [m for m in messages if m["role"] != "assistant"]
#         gt_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
#         gt_safe = extract_is_safe(gt_msg)

#         inputs = tokenizer.apply_chat_template(
#             prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
#         ).to("cuda")

#         t0 = time.time()
#         with torch.no_grad():
#             out = model.generate(
#                 input_ids=inputs, max_new_tokens=MAX_NEW_TOKENS,
#                 use_cache=True, do_sample=False,
#             )
#         gen_seconds = time.time() - t0
#         response = tokenizer.decode(
#             out[0][len(inputs[0]):], skip_special_tokens=True,
#         )

#         pred_safe = extract_is_safe(response)
#         parsed_ok = pred_safe is not None
#         if not parsed_ok:
#             pred_safe = True
#         pred_reasoning = extract_reasoning(response)

#         record = {
#             "idx": int(idx),
#             "gt_is_safe": gt_safe,
#             "pred_is_safe": bool(pred_safe),
#             "parsed_ok": bool(parsed_ok),
#             "pred_reasoning": pred_reasoning,
#             "raw_response": response,
#             "gen_seconds": gen_seconds,
#             "n_generated_tokens": len(out[0]) - len(inputs[0]),
#         }
#         append_jsonl(OUT_JSONL, record)
#         append_txt_block(OUT_TXT, record)

#     # -------- Summary --------
#     print("\n" + "=" * 70)
#     print("LLAMA-3.1-8B BASE (ZERO-SHOT) TEST EVAL SUMMARY")
#     print("=" * 70)
#     all_records = []
#     with open(OUT_JSONL) as f:
#         for line in f:
#             try:
#                 all_records.append(json.loads(line))
#             except json.JSONDecodeError:
#                 continue

#     y_true, y_pred, y_scores = [], [], []
#     n_unparseable = 0
#     n_skipped = 0
#     for r in all_records:
#         if r["gt_is_safe"] is None:
#             n_skipped += 1
#             continue
#         y_true.append(0 if r["gt_is_safe"] else 1)
#         y_pred.append(0 if r["pred_is_safe"] else 1)
#         y_scores.append(0 if r["pred_is_safe"] else 1)
#         if not r["parsed_ok"]:
#             n_unparseable += 1

#     if y_true:
#         acc = accuracy_score(y_true, y_pred)
#         rec = recall_score(y_true, y_pred, zero_division=0)
#         f1 = f1_score(y_true, y_pred, zero_division=0)
#         precision, recall_curve, _ = precision_recall_curve(y_true, y_scores)
#         auprc = auc(recall_curve, precision)
#         print(f"  Total scored:               {len(y_true)}")
#         print(f"  Accuracy:                   {acc:.4f}")
#         print(f"  Recall (unsafe class):      {rec:.4f}")
#         print(f"  F1 (unsafe class):          {f1:.4f}")
#         print(f"  AUPRC (placeholder):        {auprc:.4f}")
#         print(f"  Unparseable predictions:    {n_unparseable}/{len(all_records)}")
#         print(f"  Skipped (bad ground truth): {n_skipped}")

#     print(f"\nOutputs:")
#     print(f"  {OUT_JSONL}  (machine-readable, per-sample)")
#     print(f"  {OUT_TXT}   (human-readable, per-sample)")


# if __name__ == "__main__":
#     main()

###########################################################################################################
# """
# Base model + schema-in-prompt test eval. Aligned with eval_sft_test_v3.py
# for fair comparison.

# The ONLY difference from eval_sft_test_v3.py:
#   - Loads the base model (no LoRA adapter)
#   - Adds the 17 canonical risk categories to the system prompt

# Everything else (params, parsing, metrics, summary format) is identical,
# so head-to-head numbers are directly comparable.

# To switch base model: change BASE_MODEL_NAME below.

# Outputs:
#   test_base_<model>_schema_v3_predictions.jsonl
#   test_base_<model>_schema_v3_predictions.txt
# """

# import torch
# import os
# import sys
# import re
# import json
# import time

# # ==============================================================================
# # 1. HPC PATCHES (same as SFT script)
# # ==============================================================================
# import torch.utils._pytree
# if not hasattr(torch.utils._pytree, "register_constant"):
#     def register_constant(cls): return cls
#     torch.utils._pytree.register_constant = register_constant

# def patch_torch_dtypes():
#     for i in range(1, 8):
#         if not hasattr(torch, f"int{i}"): setattr(torch, f"int{i}", torch.int8)
#         if not hasattr(torch, f"uint{i}"): setattr(torch, f"uint{i}", torch.uint8)
# patch_torch_dtypes()

# os.environ["BNB_CUDA_VERSION"] = "121"
# os.environ["LD_LIBRARY_PATH"] = (
#     "/apps/spack/0.21/ascend/linux-rhel9-zen2/cuda/gcc/11.4.1/12.4.1-rni5fqf/targets/x86_64-linux/lib:"
#     + os.environ.get("LD_LIBRARY_PATH", "")
# )

# from unsloth import FastLanguageModel
# from datasets import load_dataset
# from tqdm import tqdm

# # ==============================================================================
# # 2. CONFIG (same generation params as SFT script)
# # ==============================================================================
# BASE_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
# # Other options to try (uncomment one at a time):
# # BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
# # BASE_MODEL_NAME = "unsloth/llama-3-70b-Instruct-bnb-4bit"

# TEST_JSONL           = os.path.abspath("Claude/SFT/data_chatml/test.jsonl")
# RISK_CATEGORIES_FILE = os.path.abspath("risk_categories.txt")

# # Output filenames auto-derived from model name
# _model_short = BASE_MODEL_NAME.split("/")[-1].replace("-", "_")
# OUT_JSONL    = f"test_base_{_model_short}_schema_v3_predictions.jsonl"
# OUT_TXT      = f"test_base_{_model_short}_schema_v3_predictions.txt"

# MAX_SEQ_LENGTH = 4096
# MAX_NEW_TOKENS = 2048

# if not os.path.isfile(TEST_JSONL):
#     sys.exit(f"Test jsonl not found: {TEST_JSONL}")
# if not os.path.isfile(RISK_CATEGORIES_FILE):
#     sys.exit(f"Risk categories file not found: {RISK_CATEGORIES_FILE}")


# def load_risk_categories(path):
#     cats = []
#     with open(path) as f:
#         for line in f:
#             line = line.strip()
#             if line and not line.startswith("#"):
#                 cats.append(line)
#     return cats


# # ==============================================================================
# # 3. SYSTEM PROMPT (the ONE difference from the SFT eval)
# # ==============================================================================
# # Verbatim SFT system prompt (from convert_csv_to_chatml.py), plus the
# # canonical 17-category list. The list is the entire experimental treatment.

# SFT_SYSTEM_PROMPT = (
#     "You are an expert clinical safety guardrail AI. Analyze the patient profile, "
#     "physician assessment report, and clinical scenario provided. Think step by step: "
#     "first work through the clinical reasoning, then evaluate each predefined risk "
#     "category individually, and only then commit to a final safety verdict. Output "
#     "your response strictly as a JSON object with this exact key order: 'reasoning' "
#     "(your step-by-step clinical analysis as a string), 'risk_analysis' (an object "
#     "mapping each risk category to true or false), and 'is_safe' (the final boolean "
#     "verdict, true only if every risk category is false)."
# )


# def build_system_prompt(categories):
#     category_list_str = "\n".join(f"  - {c}" for c in categories)
#     return (
#         SFT_SYSTEM_PROMPT
#         + "\n\nThe predefined risk categories are:\n"
#         + category_list_str
#         + f"\n\nUse these exact {len(categories)} category names (verbatim) "
#         + "as keys in the 'risk_analysis' object."
#     )


# # ==============================================================================
# # 4. PARSING (identical to SFT eval)
# # ==============================================================================

# def _strip_noise(text):
#     cleaned = text
#     for noise in ("</tool_call>", "<tool_call>", "<think>", "</think>"):
#         cleaned = cleaned.replace(noise, "")
#     return cleaned


# def extract_json_dict(text):
#     cleaned = _strip_noise(text)
#     m = re.search(r"\{.*\}", cleaned, re.DOTALL)
#     if m:
#         try:
#             data = json.loads(m.group())
#             if isinstance(data, dict):
#                 return data
#         except json.JSONDecodeError:
#             pass
#     return None


# def recover_partial_risk_analysis(text, categories):
#     cleaned = _strip_noise(text)
#     out = {c: False for c in categories}
#     n_found = 0
#     for c in categories:
#         for variant in (c, c.replace("-", "\u2013")):
#             esc = re.escape(variant)
#             pat = rf'"{esc}"\s*:\s*(true|false)'
#             m = re.search(pat, cleaned, re.IGNORECASE)
#             if m:
#                 out[c] = m.group(1).lower() == "true"
#                 n_found += 1
#                 break
#     return out, n_found


# def extract_is_safe(text):
#     parsed = extract_json_dict(text)
#     if parsed and "is_safe" in parsed:
#         return bool(parsed["is_safe"])
#     m = re.search(r'"is_safe"\s*:\s*(true|false)', text, re.IGNORECASE)
#     if m:
#         return m.group(1).lower() == "true"
#     return None


# def extract_reasoning(text):
#     parsed = extract_json_dict(text)
#     if parsed and "reasoning" in parsed:
#         return str(parsed["reasoning"])
#     m = re.search(r'"reasoning"\s*:\s*"(.*?)(?<!\\)"', text, re.DOTALL)
#     if m:
#         return m.group(1)
#     return None


# def extract_risk_analysis(text, categories):
#     parsed = extract_json_dict(text)
#     if parsed and "risk_analysis" in parsed and isinstance(parsed["risk_analysis"], dict):
#         ra_raw = parsed["risk_analysis"]
#         out = {}
#         for c in categories:
#             if c in ra_raw:
#                 out[c] = bool(ra_raw[c])
#             else:
#                 alt = c.replace("-", "\u2013")
#                 out[c] = bool(ra_raw.get(alt, False))
#         return out, True, len(categories)
#     out, n_found = recover_partial_risk_analysis(text, categories)
#     return out, False, n_found


# def append_jsonl(path, record):
#     with open(path, "a") as f:
#         f.write(json.dumps(record) + "\n")
#         f.flush()
#         os.fsync(f.fileno())


# def append_txt_block(path, record):
#     lines = [
#         "=" * 78,
#         f"idx: {record['idx']}",
#         f"gt_is_safe:      {record['gt_is_safe']}",
#         f"pred_is_safe:    {record['pred_is_safe']}",
#         f"parsed_ok:       {record['parsed_ok']}",
#         f"ra_parsed_ok:    {record['ra_parsed_ok']}",
#         f"ra_recovered:    {record['ra_n_recovered']}/17",
#         f"correct:         {record['gt_is_safe'] == record['pred_is_safe']}",
#         f"gen_seconds:     {record['gen_seconds']:.1f}",
#         f"n_gen_tokens:    {record['n_generated_tokens']}",
#         "",
#         "-- predicted reasoning --",
#         (record.get("pred_reasoning") or "(could not extract reasoning)").strip(),
#         "",
#         "-- raw model output (first 1500 chars) --",
#         record["raw_response"][:1500],
#         "",
#     ]
#     with open(path, "a") as f:
#         f.write("\n".join(lines) + "\n")
#         f.flush()
#         os.fsync(f.fileno())


# def load_completed_indices(path):
#     if not os.path.exists(path):
#         return set()
#     done = set()
#     with open(path) as f:
#         for line in f:
#             try:
#                 r = json.loads(line)
#                 if "idx" in r:
#                     done.add(int(r["idx"]))
#             except json.JSONDecodeError:
#                 continue
#     return done


# # ==============================================================================
# # 5. MAIN
# # ==============================================================================

# def main():
#     categories = load_risk_categories(RISK_CATEGORIES_FILE)
#     print(f"Loaded {len(categories)} risk categories")

#     system_prompt = build_system_prompt(categories)
#     print(f"System prompt length: ~{len(system_prompt) // 4} tokens")
#     print(f"max_seq_length: {MAX_SEQ_LENGTH}")
#     print(f"max_new_tokens: {MAX_NEW_TOKENS}")
#     print()
#     print(f"Base model: {BASE_MODEL_NAME}")
#     print(f"Output:     {OUT_JSONL}")
#     print()
#     print(f"Loading base model via Unsloth...")
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name     = BASE_MODEL_NAME,
#         max_seq_length = MAX_SEQ_LENGTH,
#         load_in_4bit   = True,
#         device_map     = "auto",
#     )
#     FastLanguageModel.for_inference(model)
#     model.eval()

#     test_ds = load_dataset("json", data_files=TEST_JSONL, split="train")
#     print(f"Test: {len(test_ds)} examples")

#     # -------- Sanity check --------
#     print("\nSanity check on test example 0...")
#     ex0 = test_ds[0]
#     user_msg = next((m["content"] for m in ex0["messages"]
#                      if m["role"] == "user"), "")
#     # CRITICAL: replace the system prompt with our schema-in-prompt version,
#     # keep the user message from test.jsonl unchanged.
#     prompt_msgs = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user",   "content": user_msg},
#     ]
#     inputs = tokenizer.apply_chat_template(
#         prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
#     ).to("cuda")
#     print(f"  Prompt length: {inputs.shape[1]} tokens")
#     t0 = time.time()
#     with torch.no_grad():
#         out = model.generate(
#             input_ids=inputs, max_new_tokens=200,
#             use_cache=True, do_sample=False,
#         )
#     sanity_time = time.time() - t0
#     sample = tokenizer.decode(out[0][len(inputs[0]):], skip_special_tokens=True)
#     print(f"  Generation took: {sanity_time:.1f}s")
#     print(f"  Generated tokens: {len(out[0]) - len(inputs[0])}")
#     sanity_ra, sanity_ok, sanity_recovered = extract_risk_analysis(sample, categories)
#     print(f"  risk_analysis full parse OK: {sanity_ok}")
#     print(f"  risk_analysis recovered categories: {sanity_recovered}/17")
#     print()

#     # -------- Resume support --------
#     done = load_completed_indices(OUT_JSONL)
#     if done:
#         print(f"Found {len(done)} already-completed samples; skipping.")
#     todo = [i for i in range(len(test_ds)) if i not in done]
#     print(f"Will process {len(todo)} new samples\n")

#     # -------- Full generation loop --------
#     for idx in tqdm(todo, desc="Generating"):
#         messages = test_ds[idx]["messages"]
#         user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
#         gt_msg   = next((m["content"] for m in messages if m["role"] == "assistant"), "")
#         gt_safe  = extract_is_safe(gt_msg)

#         prompt_msgs = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user",   "content": user_msg},
#         ]
#         inputs = tokenizer.apply_chat_template(
#             prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
#         ).to("cuda")

#         t0 = time.time()
#         with torch.no_grad():
#             out = model.generate(
#                 input_ids=inputs, max_new_tokens=MAX_NEW_TOKENS,
#                 use_cache=True, do_sample=False,
#             )
#         gen_seconds = time.time() - t0
#         response = tokenizer.decode(
#             out[0][len(inputs[0]):], skip_special_tokens=True,
#         )

#         pred_safe = extract_is_safe(response)
#         parsed_ok = pred_safe is not None
#         if not parsed_ok:
#             pred_safe = True
#         pred_reasoning = extract_reasoning(response)
#         pred_ra, ra_parsed_ok, ra_n_recovered = extract_risk_analysis(response, categories)

#         record = {
#             "idx": int(idx),
#             "gt_is_safe": gt_safe,
#             "pred_is_safe": bool(pred_safe),
#             "parsed_ok": bool(parsed_ok),
#             "ra_parsed_ok": bool(ra_parsed_ok),
#             "ra_n_recovered": int(ra_n_recovered),
#             "pred_reasoning": pred_reasoning,
#             "pred_risk_analysis": pred_ra,
#             "raw_response": response,
#             "gen_seconds": gen_seconds,
#             "n_generated_tokens": len(out[0]) - len(inputs[0]),
#         }
#         append_jsonl(OUT_JSONL, record)
#         append_txt_block(OUT_TXT, record)

#     # -------- Summary (identical to SFT eval) --------
#     from sklearn.metrics import (
#         accuracy_score, recall_score, f1_score, precision_score,
#         precision_recall_curve, auc,
#     )

#     print("\n" + "=" * 70)
#     print(f"BASE + SCHEMA (v3, FAIR) TEST EVAL SUMMARY")
#     print(f"Model:           {BASE_MODEL_NAME}")
#     print(f"max_seq_length:  {MAX_SEQ_LENGTH}  max_new_tokens: {MAX_NEW_TOKENS}")
#     print("=" * 70)

#     all_records = []
#     with open(OUT_JSONL) as f:
#         for line in f:
#             try:
#                 all_records.append(json.loads(line))
#             except json.JSONDecodeError:
#                 continue

#     y_true, y_pred = [], []
#     n_is_safe_unparseable = 0
#     n_ra_fully_parsed = 0
#     n_ra_partially_recovered = 0
#     n_ra_completely_failed = 0
#     tokens_at_max = 0
#     for r in all_records:
#         if r["gt_is_safe"] is None:
#             continue
#         y_true.append(0 if r["gt_is_safe"] else 1)
#         y_pred.append(0 if r["pred_is_safe"] else 1)
#         if not r["parsed_ok"]:
#             n_is_safe_unparseable += 1
#         if r.get("ra_parsed_ok"):
#             n_ra_fully_parsed += 1
#         elif r.get("ra_n_recovered", 0) > 0:
#             n_ra_partially_recovered += 1
#         else:
#             n_ra_completely_failed += 1
#         if r.get("n_generated_tokens", 0) >= MAX_NEW_TOKENS:
#             tokens_at_max += 1

#     if y_true:
#         acc = accuracy_score(y_true, y_pred)
#         rec = recall_score(y_true, y_pred, zero_division=0)
#         f1 = f1_score(y_true, y_pred, zero_division=0)
#         precision, recall_curve, _ = precision_recall_curve(y_true, y_pred)
#         auprc = auc(recall_curve, precision)
#         print(f"\nIS_SAFE METRICS:")
#         print(f"  Total scored:               {len(y_true)}")
#         print(f"  Accuracy:                   {acc:.4f}")
#         print(f"  Recall (unsafe class):      {rec:.4f}")
#         print(f"  F1 (unsafe class):          {f1:.4f}")
#         print(f"  AUPRC (placeholder):        {auprc:.4f}")

#         print(f"\nOUTPUT STRUCTURE HEALTH:")
#         print(f"  is_safe unparseable:           {n_is_safe_unparseable}/{len(all_records)}")
#         print(f"  risk_analysis fully parsed:    {n_ra_fully_parsed}/{len(all_records)}")
#         print(f"  risk_analysis partial recov:   {n_ra_partially_recovered}/{len(all_records)}")
#         print(f"  risk_analysis completely fail: {n_ra_completely_failed}/{len(all_records)}")
#         print(f"  Hit MAX_NEW_TOKENS cap:        {tokens_at_max}/{len(all_records)}")

#     # ------ Per-category metrics ------
#     print(f"\nPER-CATEGORY METRICS:")
#     print(f"{'Category':<42s}  {'Support':>8s}  {'Prec':>6s}  {'Recall':>7s}  {'F1':>6s}")
#     print("-" * 80)

#     y_true_per_cat = {c: [] for c in categories}
#     y_pred_per_cat = {c: [] for c in categories}
#     exact_matches = 0
#     n_compared = 0

#     for r in all_records:
#         if r["gt_is_safe"] is None:
#             continue
#         gt_assistant = next(
#             (m["content"] for m in test_ds[r["idx"]]["messages"] if m["role"] == "assistant"),
#             "",
#         )
#         gt_ra, gt_ok, _ = extract_risk_analysis(gt_assistant, categories)
#         if not gt_ok:
#             continue
#         pred_ra = r.get("pred_risk_analysis") or {c: False for c in categories}
#         n_compared += 1
#         all_match = True
#         for c in categories:
#             y_true_per_cat[c].append(int(gt_ra[c]))
#             y_pred_per_cat[c].append(int(pred_ra.get(c, False)))
#             if gt_ra[c] != pred_ra.get(c, False):
#                 all_match = False
#         if all_match:
#             exact_matches += 1

#     per_cat_f1 = []
#     pooled_yt, pooled_yp = [], []
#     for c in categories:
#         yt = y_true_per_cat[c]
#         yp = y_pred_per_cat[c]
#         support = sum(yt)
#         pooled_yt.extend(yt)
#         pooled_yp.extend(yp)
#         if support == 0:
#             print(f"{c:<42s}  {support:>8d}    --        --       --")
#             continue
#         p = precision_score(yt, yp, zero_division=0)
#         rr = recall_score(yt, yp, zero_division=0)
#         f = f1_score(yt, yp, zero_division=0)
#         per_cat_f1.append(f)
#         print(f"{c:<42s}  {support:>8d}  {p:>6.3f}  {rr:>7.3f}  {f:>6.3f}")

#     print("-" * 80)
#     if per_cat_f1:
#         macro_f1 = sum(per_cat_f1) / len(per_cat_f1)
#         print(f"  Macro F1: {macro_f1:.4f}")
#     if pooled_yt:
#         micro_p = precision_score(pooled_yt, pooled_yp, zero_division=0)
#         micro_r = recall_score(pooled_yt, pooled_yp, zero_division=0)
#         micro_f = f1_score(pooled_yt, pooled_yp, zero_division=0)
#         print(f"  Micro Precision: {micro_p:.4f}")
#         print(f"  Micro Recall:    {micro_r:.4f}")
#         print(f"  Micro F1:        {micro_f:.4f}")
#     if n_compared > 0:
#         print(f"\n  Exact-match accuracy: {exact_matches}/{n_compared} = "
#               f"{exact_matches/n_compared:.4f}")


# if __name__ == "__main__":
#     main()

###########################################################################################################
# """
# Zero-shot Llama-3-70B baseline on the test set.

# Uses Unsloth + 4-bit quantization to fit on a single A100-80GB.
# Mirrors eval_base_llama_test.py but with a much larger base model.

# Outputs:
#   test_base_llama70b_predictions.jsonl
#   test_base_llama70b_predictions.txt

# WARNING: This is slow (~60-120 sec/sample) and big. Expect 8-14 hours
# for 411 samples. Run under nohup and request a long-running SLURM job:
#   salloc -A PCS0289 --gres=gpu:a100:1 --mem=128G --time=16:00:00
# """

# import torch
# import os
# import sys
# import re
# import json
# import time
# import numpy as np
# from sklearn.metrics import (
#     accuracy_score, recall_score, f1_score,
#     precision_recall_curve, auc,
# )

# # ==============================================================================
# # 1. HPC PATCHES
# # ==============================================================================
# import torch.utils._pytree
# if not hasattr(torch.utils._pytree, "register_constant"):
#     def register_constant(cls): return cls
#     torch.utils._pytree.register_constant = register_constant

# def patch_torch_dtypes():
#     for i in range(1, 8):
#         if not hasattr(torch, f"int{i}"): setattr(torch, f"int{i}", torch.int8)
#         if not hasattr(torch, f"uint{i}"): setattr(torch, f"uint{i}", torch.uint8)
# patch_torch_dtypes()

# os.environ["BNB_CUDA_VERSION"] = "121"
# os.environ["LD_LIBRARY_PATH"] = (
#     "/apps/spack/0.21/ascend/linux-rhel9-zen2/cuda/gcc/11.4.1/12.4.1-rni5fqf/targets/x86_64-linux/lib:"
#     + os.environ.get("LD_LIBRARY_PATH", "")
# )

# # ==============================================================================
# # 2. IMPORTS
# # ==============================================================================
# from unsloth import FastLanguageModel
# from datasets import load_dataset
# from tqdm import tqdm

# # ==============================================================================
# # 3. CONFIG
# # ==============================================================================
# # Unsloth pre-quantized Llama-3-70B-Instruct in 4-bit (~40GB on disk).
# # If you want Llama-3.1-70B instead (newer):
# #   BASE_MODEL_NAME = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
# BASE_MODEL_NAME = "unsloth/llama-3-70b-Instruct-bnb-4bit"

# TEST_JSONL      = os.path.abspath("Claude/SFT/data_chatml/test.jsonl")
# OUT_JSONL       = "test_base_llama70b_predictions.jsonl"
# OUT_TXT         = "test_base_llama70b_predictions.txt"
# MAX_NEW_TOKENS  = 1024

# if not os.path.isfile(TEST_JSONL):
#     sys.exit(f"Test jsonl not found: {TEST_JSONL}")

# # ==============================================================================
# # 4. PARSING + I/O (identical to other eval scripts)
# # ==============================================================================

# def extract_is_safe(text):
#     cleaned = text
#     for noise in ("</tool_call>", "<tool_call>", "<think>", "</think>"):
#         cleaned = cleaned.replace(noise, "")
#     m = re.search(r"\{.*\}", cleaned, re.DOTALL)
#     if m:
#         try:
#             data = json.loads(m.group())
#             if isinstance(data, dict) and "is_safe" in data:
#                 return bool(data["is_safe"])
#         except json.JSONDecodeError:
#             pass
#     m = re.search(r'"is_safe"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
#     if m:
#         return m.group(1).lower() == "true"
#     return None


# def extract_reasoning(text):
#     cleaned = text
#     for noise in ("</tool_call>", "<tool_call>", "<think>", "</think>"):
#         cleaned = cleaned.replace(noise, "")
#     m = re.search(r"\{.*\}", cleaned, re.DOTALL)
#     if m:
#         try:
#             data = json.loads(m.group())
#             if isinstance(data, dict) and "reasoning" in data:
#                 return str(data["reasoning"])
#         except json.JSONDecodeError:
#             pass
#     return None


# def append_jsonl(path, record):
#     with open(path, "a") as f:
#         f.write(json.dumps(record) + "\n")
#         f.flush()
#         os.fsync(f.fileno())


# def append_txt_block(path, record):
#     lines = [
#         "=" * 78,
#         f"idx: {record['idx']}",
#         f"gt_is_safe:    {record['gt_is_safe']}",
#         f"pred_is_safe:  {record['pred_is_safe']}",
#         f"parsed_ok:     {record['parsed_ok']}",
#         f"correct:       {record['gt_is_safe'] == record['pred_is_safe']}",
#         f"gen_seconds:   {record['gen_seconds']:.1f}",
#         "",
#         "-- predicted reasoning --",
#         (record.get("pred_reasoning") or "(could not extract reasoning)").strip(),
#         "",
#         "-- raw model output (first 1200 chars) --",
#         record["raw_response"][:1200],
#         "",
#     ]
#     with open(path, "a") as f:
#         f.write("\n".join(lines) + "\n")
#         f.flush()
#         os.fsync(f.fileno())


# def load_completed_indices(path):
#     if not os.path.exists(path):
#         return set()
#     done = set()
#     with open(path) as f:
#         for line in f:
#             try:
#                 r = json.loads(line)
#                 if "idx" in r:
#                     done.add(int(r["idx"]))
#             except json.JSONDecodeError:
#                 continue
#     return done


# # ==============================================================================
# # 5. MAIN
# # ==============================================================================

# def main():
#     print(f"Base model: {BASE_MODEL_NAME}  (no fine-tuning, no LoRA)")
#     print(f"Test data:  {TEST_JSONL}")

#     print(f"\nLoading Llama-3-70B in 4-bit via Unsloth...")
#     print(f"(This will take ~10-15 minutes for download + load on first run)")
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name     = BASE_MODEL_NAME,
#         max_seq_length = 2048,
#         load_in_4bit   = True,
#         device_map     = "auto",
#     )
#     FastLanguageModel.for_inference(model)
#     model.eval()

#     test_ds = load_dataset("json", data_files=TEST_JSONL, split="train")
#     print(f"Test: {len(test_ds)} examples")

#     # -------- Sanity check --------
#     print("\nSanity check generation on test example 0...")
#     ex0 = test_ds[0]
#     prompt_msgs = [m for m in ex0["messages"] if m["role"] != "assistant"]
#     inputs = tokenizer.apply_chat_template(
#         prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
#     ).to("cuda")
#     t0 = time.time()
#     with torch.no_grad():
#         out = model.generate(
#             input_ids=inputs, max_new_tokens=200,
#             use_cache=True, do_sample=False,
#         )
#     sanity_time = time.time() - t0
#     sample = tokenizer.decode(out[0][len(inputs[0]):], skip_special_tokens=False)
#     print(f"  Generation took: {sanity_time:.1f}s for 200 tokens")
#     print(f"  ({sanity_time/200:.2f}s per token; expect ~{sanity_time*5*411/60:.0f} min for full eval)")
#     print(f"  First 300 chars:")
#     print(f"  {sample[:300]!r}")
#     print("  Proceeding to full eval regardless of output quality.\n")

#     # -------- Resume support --------
#     done = load_completed_indices(OUT_JSONL)
#     if done:
#         print(f"Found {len(done)} already-completed samples; skipping them.")
#     todo = [i for i in range(len(test_ds)) if i not in done]
#     print(f"Will process {len(todo)} new samples")

#     # -------- Full generation loop --------
#     for idx in tqdm(todo, desc="Generating"):
#         messages = test_ds[idx]["messages"]
#         prompt_msgs = [m for m in messages if m["role"] != "assistant"]
#         gt_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
#         gt_safe = extract_is_safe(gt_msg)

#         inputs = tokenizer.apply_chat_template(
#             prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
#         ).to("cuda")

#         t0 = time.time()
#         with torch.no_grad():
#             out = model.generate(
#                 input_ids=inputs, max_new_tokens=MAX_NEW_TOKENS,
#                 use_cache=True, do_sample=False,
#             )
#         gen_seconds = time.time() - t0
#         response = tokenizer.decode(
#             out[0][len(inputs[0]):], skip_special_tokens=True,
#         )

#         pred_safe = extract_is_safe(response)
#         parsed_ok = pred_safe is not None
#         if not parsed_ok:
#             pred_safe = True
#         pred_reasoning = extract_reasoning(response)

#         record = {
#             "idx": int(idx),
#             "gt_is_safe": gt_safe,
#             "pred_is_safe": bool(pred_safe),
#             "parsed_ok": bool(parsed_ok),
#             "pred_reasoning": pred_reasoning,
#             "raw_response": response,
#             "gen_seconds": gen_seconds,
#             "n_generated_tokens": len(out[0]) - len(inputs[0]),
#         }
#         append_jsonl(OUT_JSONL, record)
#         append_txt_block(OUT_TXT, record)

#     # -------- Summary --------
#     print("\n" + "=" * 70)
#     print("LLAMA-3-70B BASE (ZERO-SHOT) TEST EVAL SUMMARY")
#     print("=" * 70)
#     all_records = []
#     with open(OUT_JSONL) as f:
#         for line in f:
#             try:
#                 all_records.append(json.loads(line))
#             except json.JSONDecodeError:
#                 continue

#     y_true, y_pred, y_scores = [], [], []
#     n_unparseable = 0
#     n_skipped = 0
#     for r in all_records:
#         if r["gt_is_safe"] is None:
#             n_skipped += 1
#             continue
#         y_true.append(0 if r["gt_is_safe"] else 1)
#         y_pred.append(0 if r["pred_is_safe"] else 1)
#         y_scores.append(0 if r["pred_is_safe"] else 1)
#         if not r["parsed_ok"]:
#             n_unparseable += 1

#     if y_true:
#         acc = accuracy_score(y_true, y_pred)
#         rec = recall_score(y_true, y_pred, zero_division=0)
#         f1 = f1_score(y_true, y_pred, zero_division=0)
#         precision, recall_curve, _ = precision_recall_curve(y_true, y_scores)
#         auprc = auc(recall_curve, precision)
#         print(f"  Total scored:               {len(y_true)}")
#         print(f"  Accuracy:                   {acc:.4f}")
#         print(f"  Recall (unsafe class):      {rec:.4f}")
#         print(f"  F1 (unsafe class):          {f1:.4f}")
#         print(f"  AUPRC (placeholder):        {auprc:.4f}")
#         print(f"  Unparseable predictions:    {n_unparseable}/{len(all_records)}")
#         print(f"  Skipped (bad ground truth): {n_skipped}")

#     print(f"\nOutputs:")
#     print(f"  {OUT_JSONL}")
#     print(f"  {OUT_TXT}")


# if __name__ == "__main__":
#     main()

############################################################################################################
"""
Base model + schema-in-prompt test eval. Aligned with eval_sft_test_v3.py
for fair comparison.

The ONLY difference from eval_sft_test_v3.py:
  - Loads the base model (no LoRA adapter)
  - Adds the 17 canonical risk categories to the system prompt

Everything else (params, parsing, metrics, summary format) is identical,
so head-to-head numbers are directly comparable.

To switch base model: change BASE_MODEL_NAME below.

Outputs:
  test_base_<model>_schema_v3_predictions.jsonl
  test_base_<model>_schema_v3_predictions.txt
"""

import torch
import os
import sys
import re
import json
import time

# ==============================================================================
# 1. HPC PATCHES (same as SFT script)
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
# 2. CONFIG (same generation params as SFT script)
# ==============================================================================
BASE_MODEL_NAME = "unsloth/llama-3-70b-Instruct-bnb-4bit"
# Other options to try (uncomment one at a time):
# BASE_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
# BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

TEST_JSONL           = os.path.abspath("Claude/SFT/data_chatml/test.jsonl")
RISK_CATEGORIES_FILE = os.path.abspath("risk_categories.txt")

# Output filenames auto-derived from model name
_model_short = BASE_MODEL_NAME.split("/")[-1].replace("-", "_")
OUT_JSONL    = f"test_base_{_model_short}_schema_v3_predictions.jsonl"
OUT_TXT      = f"test_base_{_model_short}_schema_v3_predictions.txt"

MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 2048

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
# 3. SYSTEM PROMPT (the ONE difference from the SFT eval)
# ==============================================================================
# Verbatim SFT system prompt (from convert_csv_to_chatml.py), plus the
# canonical 17-category list. The list is the entire experimental treatment.

SFT_SYSTEM_PROMPT = (
    "You are an expert clinical safety guardrail AI. Analyze the patient profile, "
    "physician assessment report, and clinical scenario provided. Think step by step: "
    "first work through the clinical reasoning, then evaluate each predefined risk "
    "category individually, and only then commit to a final safety verdict. Output "
    "your response strictly as a JSON object with this exact key order: 'reasoning' "
    "(your step-by-step clinical analysis as a string), 'risk_analysis' (an object "
    "mapping each risk category to true or false), and 'is_safe' (the final boolean "
    "verdict, true only if every risk category is false)."
)


def build_system_prompt(categories):
    category_list_str = "\n".join(f"  - {c}" for c in categories)
    return (
        SFT_SYSTEM_PROMPT
        + "\n\nThe predefined risk categories are:\n"
        + category_list_str
        + f"\n\nUse these exact {len(categories)} category names (verbatim) "
        + "as keys in the 'risk_analysis' object."
    )


# ==============================================================================
# 4. PARSING (identical to SFT eval)
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
# 5. MAIN
# ==============================================================================

def main():
    categories = load_risk_categories(RISK_CATEGORIES_FILE)
    print(f"Loaded {len(categories)} risk categories")

    system_prompt = build_system_prompt(categories)
    print(f"System prompt length: ~{len(system_prompt) // 4} tokens")
    print(f"max_seq_length: {MAX_SEQ_LENGTH}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")
    print()
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"Output:     {OUT_JSONL}")
    print()
    print(f"Loading base model via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit   = True,
        device_map     = "auto",
    )
    FastLanguageModel.for_inference(model)
    model.eval()

    test_ds = load_dataset("json", data_files=TEST_JSONL, split="train")
    print(f"Test: {len(test_ds)} examples")

    # -------- Sanity check --------
    print("\nSanity check on test example 0...")
    ex0 = test_ds[0]
    user_msg = next((m["content"] for m in ex0["messages"]
                     if m["role"] == "user"), "")
    # CRITICAL: replace the system prompt with our schema-in-prompt version,
    # keep the user message from test.jsonl unchanged.
    prompt_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_msg},
    ]
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
    sample = tokenizer.decode(out[0][len(inputs[0]):], skip_special_tokens=True)
    print(f"  Generation took: {sanity_time:.1f}s")
    print(f"  Generated tokens: {len(out[0]) - len(inputs[0])}")
    sanity_ra, sanity_ok, sanity_recovered = extract_risk_analysis(sample, categories)
    print(f"  risk_analysis full parse OK: {sanity_ok}")
    print(f"  risk_analysis recovered categories: {sanity_recovered}/17")
    print()

    # -------- Resume support --------
    done = load_completed_indices(OUT_JSONL)
    if done:
        print(f"Found {len(done)} already-completed samples; skipping.")
    todo = [i for i in range(len(test_ds)) if i not in done]
    print(f"Will process {len(todo)} new samples\n")

    # -------- Full generation loop --------
    for idx in tqdm(todo, desc="Generating"):
        messages = test_ds[idx]["messages"]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        gt_msg   = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        gt_safe  = extract_is_safe(gt_msg)

        prompt_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ]
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

    # -------- Summary (identical to SFT eval) --------
    from sklearn.metrics import (
        accuracy_score, recall_score, f1_score, precision_score,
        precision_recall_curve, auc,
    )

    print("\n" + "=" * 70)
    print(f"BASE + SCHEMA (v3, FAIR) TEST EVAL SUMMARY")
    print(f"Model:           {BASE_MODEL_NAME}")
    print(f"max_seq_length:  {MAX_SEQ_LENGTH}  max_new_tokens: {MAX_NEW_TOKENS}")
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