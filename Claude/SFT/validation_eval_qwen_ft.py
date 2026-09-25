# """
# Standalone post-training validation eval using vLLM for fast batched inference.

# Loads the merged 16-bit model from SFT/outputs/Qwen3-4B-Instruct/final/,
# generates predictions for all val examples in one batched call, then writes
# results to:
#   - val_predictions.jsonl  : one JSON object per sample (machine-readable)
#   - val_predictions.txt    : one block per sample (human-readable)

# Expected wall time: ~10-15 min for 173 samples on A100-80GB.
# Requires: pip install vllm
# """

# import os
# import sys
# import re
# import json
# import time
# from sklearn.metrics import (
#     accuracy_score, recall_score, f1_score,
#     precision_recall_curve, auc,
# )

# # HPC: keep CUDA library path.
# os.environ["LD_LIBRARY_PATH"] = (
#     "/apps/spack/0.21/ascend/linux-rhel9-zen2/cuda/gcc/11.4.1/12.4.1-rni5fqf/targets/x86_64-linux/lib:"
#     + os.environ.get("LD_LIBRARY_PATH", "")
# )

# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer
# from datasets import load_dataset

# # ==============================================================================
# # CONFIG
# # ==============================================================================
# # Use os.path.abspath so transformers/huggingface_hub doesn't mistake the
# # relative path for a HF Hub repo id and reject it via the validator.
# MODEL_PATH = os.path.abspath("Claude/SFT/outputs/Qwen3-4B-Instruct/final")
# VAL_JSONL  = os.path.abspath("Claude/SFT/data_chatml/val.jsonl")
# OUT_JSONL  = "val_predictions.jsonl"
# OUT_TXT    = "val_predictions.txt"

# # Sanity check before doing anything expensive.
# if not os.path.isdir(MODEL_PATH):
#     sys.exit(f"ERROR: model directory not found:\n  {MODEL_PATH}\n"
#              f"Check that training completed and saved to that path.")
# if not os.path.isfile(VAL_JSONL):
#     sys.exit(f"ERROR: val jsonl not found:\n  {VAL_JSONL}")

# # ==============================================================================
# # HELPERS
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


# def write_jsonl(path, records):
#     with open(path, "w") as f:
#         for r in records:
#             f.write(json.dumps(r) + "\n")


# def write_txt_blocks(path, records):
#     with open(path, "w") as f:
#         for r in records:
#             lines = [
#                 "=" * 78,
#                 f"idx: {r['idx']}",
#                 f"gt_is_safe:    {r['gt_is_safe']}",
#                 f"pred_is_safe:  {r['pred_is_safe']}",
#                 f"parsed_ok:     {r['parsed_ok']}",
#                 f"correct:       {r['gt_is_safe'] == r['pred_is_safe']}",
#                 "",
#                 "-- predicted reasoning --",
#                 (r.get("pred_reasoning") or "(could not extract reasoning)").strip(),
#                 "",
#                 "-- raw model output (first 1200 chars) --",
#                 r["raw_response"][:1200],
#                 "",
#             ]
#             f.write("\n".join(lines) + "\n")


# # ==============================================================================
# # MAIN
# # ==============================================================================

# def main():
#     print(f"Model path: {MODEL_PATH}")
#     print(f"Val data:   {VAL_JSONL}")

#     print(f"\nLoading val data...")
#     val_ds = load_dataset("json", data_files=VAL_JSONL, split="train")
#     print(f"Val: {len(val_ds)} examples")

#     print("Building prompts...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#     prompts = []
#     gt_safe_list = []
#     for idx in range(len(val_ds)):
#         messages = val_ds[idx]["messages"]
#         prompt_msgs = [m for m in messages if m["role"] != "assistant"]
#         gt_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
#         gt_safe_list.append(extract_is_safe(gt_msg))

#         prompt_text = tokenizer.apply_chat_template(
#             prompt_msgs, tokenize=False, add_generation_prompt=True,
#         )
#         prompts.append(prompt_text)

#     print(f"\nLoading model into vLLM...")
#     llm = LLM(
#         model=MODEL_PATH,
#         dtype="bfloat16",
#         gpu_memory_utilization=0.85,
#         max_model_len=4096,
#         trust_remote_code=False,
#     )

#     sampling = SamplingParams(
#         temperature=0.0,
#         max_tokens=1024,
#         stop=["<|im_end|>"],
#         skip_special_tokens=True,
#     )

#     print(f"\nGenerating {len(prompts)} predictions (batched)...")
#     t0 = time.time()
#     outputs = llm.generate(prompts, sampling)
#     gen_time = time.time() - t0
#     print(f"Done. Generation wall time: {gen_time:.1f}s "
#           f"({gen_time / len(prompts):.2f}s per sample on average)")

#     print("Scoring + writing outputs...")
#     records = []
#     for idx, output in enumerate(outputs):
#         response = output.outputs[0].text
#         pred_safe = extract_is_safe(response)
#         parsed_ok = pred_safe is not None
#         if not parsed_ok:
#             pred_safe = True
#         pred_reasoning = extract_reasoning(response)

#         records.append({
#             "idx": int(idx),
#             "gt_is_safe": gt_safe_list[idx],
#             "pred_is_safe": bool(pred_safe),
#             "parsed_ok": bool(parsed_ok),
#             "pred_reasoning": pred_reasoning,
#             "raw_response": response,
#             "n_generated_tokens": len(output.outputs[0].token_ids),
#         })

#     write_jsonl(OUT_JSONL, records)
#     write_txt_blocks(OUT_TXT, records)
#     print(f"\nWrote per-sample results:")
#     print(f"  Machine-readable: {OUT_JSONL}")
#     print(f"  Human-readable:   {OUT_TXT}")

#     print("\n" + "=" * 70)
#     print("EVAL SUMMARY")
#     print("=" * 70)
#     y_true, y_pred, y_scores = [], [], []
#     n_unparseable = 0
#     n_skipped = 0
#     for r in records:
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
#         print(f"  Unparseable predictions:    {n_unparseable}/{len(records)}")
#         print(f"  Skipped (bad ground truth): {n_skipped}")


# if __name__ == "__main__":
#     main()





import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/users/PCS0289/sarakhosravi/Guardrail/Claude/SFT/outputs/Qwen3-4B-Instruct/final"

tok = AutoTokenizer.from_pretrained(path)
print(f"Tokenizer EOS: {tok.eos_token!r}  (id={tok.eos_token_id})")
print(f"Tokenizer PAD: {tok.pad_token!r}  (id={tok.pad_token_id})")
print(f"Token 151645 decodes to: {tok.decode([151645])!r}")
print(f"Token for <|im_end|> is: {tok.convert_tokens_to_ids('<|im_end|>')}")
print(f"Token for <tool_call> is: {tok.convert_tokens_to_ids('<tool_call>')}")

model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda")
print(f"\nModel config EOS:  {model.config.eos_token_id}")
print(f"Model gen config EOS: {model.generation_config.eos_token_id}")
model.eval()

with open("Claude/SFT/data_chatml/val.jsonl") as f:
    ex = json.loads(f.readline())
prompt = [m for m in ex["messages"] if m["role"] != "assistant"]

inputs = tok.apply_chat_template(
    prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")
print(f"\nPrompt token count: {inputs.shape[1]}")
print(f"Last 5 prompt tokens: {inputs[0][-5:].tolist()}")
print(f"Last 5 decoded: {tok.decode(inputs[0][-5:], skip_special_tokens=False)!r}")

with torch.no_grad():
    out = model.generate(
        input_ids=inputs, max_new_tokens=200, do_sample=False,
    )
generated = out[0][len(inputs[0]):]
print(f"\nGenerated {len(generated)} tokens")
print(f"First 30 token IDs: {generated[:30].tolist()}")
print(f"Decoded:\n{tok.decode(generated, skip_special_tokens=False)!r}")