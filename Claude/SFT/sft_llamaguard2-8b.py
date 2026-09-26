"""
SFT for Meta-Llama-Guard-2-8B on the clinical safety dataset.

WHY THIS DIFFERS FROM THE LLAMAGUARD-7B VERSION
===============================================
LlamaGuard-7b is Llama-2 based (SentencePiece), so it needed Llama-2
chat format with [INST] / [/INST] markers.

Llama Guard 2 is Llama-3 based (8B, tiktoken BPE). Its chat format is
the Llama-3 header format:

    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {assistant}<|eot_id|>

<|start_header_id|>, <|end_header_id|> and <|eot_id|> are genuine single
tokens in the Llama-3 vocabulary, so they tokenize consistently in
context. That is the same property that makes ChatML safe on Qwen and
that <|im_start|> lacked on Llama-2, where every sample ended up fully
masked.

Llama Guard 2 ships a template hardcoded for safety classification which
discards the assistant turn, so it still has to be replaced. We keep the
weights and swap the template.

ACCESS
======
meta-llama/Meta-Llama-Guard-2-8B is gated under the Llama 3 license.
Access to other meta-llama repos does not carry over; each is approved
separately. Verify before launching a long run:

    python -c "
    from huggingface_hub import hf_hub_download
    print(hf_hub_download('meta-llama/Meta-Llama-Guard-2-8B','config.json'))
    "

If that raises GatedRepoError, request access on the model page and
export HF_TOKEN=$(cat ~/.cache/huggingface/token) once approved.
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
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import EarlyStoppingCallback
from tqdm import tqdm

# ==============================================================================
# 3. LLAMA-3 CHAT TEMPLATE
# ==============================================================================
# Canonical Llama-3 format. Unlike Llama-2, system is its own header block
# rather than being folded into the first user turn.
#
# Markers used for response masking:
#   instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
#   response_part    = "<|start_header_id|>assistant<|end_header_id|>\n\n"
#
# Both are sequences of real special tokens, so the subsequence search in
# train_on_responses_only matches reliably.

LLAMA3_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{%- endif %}"
)

INSTRUCTION_PART = "<|start_header_id|>user<|end_header_id|>\n\n"
RESPONSE_PART = "<|start_header_id|>assistant<|end_header_id|>\n\n"

# ==============================================================================
# 4. DATA FORMATTING
# ==============================================================================

def format_for_sft(examples, tokenizer):
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False,
        )
        for convo in examples["messages"]
    ]
    return {"text": texts}

# ==============================================================================
# 5. MARKER AND LENGTH DIAGNOSTICS (run before training)
# ==============================================================================

def check_marker_tokenization(tokenizer, sample_text):
    """Verify the masking markers are findable as token subsequences.

    This is the check that catches a template/tokenizer mismatch before a
    full model load and a silently no-op training run. It tokenizes each
    marker on its own, then looks for that exact id sequence inside a real
    rendered example.

    Also warns on a duplicated BOS, which happens when the template emits
    bos_token and the tokenizer prepends another.
    """
    print("\n" + "=" * 70)
    print("MARKER TOKENIZATION CHECK")
    print("=" * 70)

    full_ids = tokenizer(sample_text, add_special_tokens=False)["input_ids"]

    ok = True
    for label, marker in (("instruction", INSTRUCTION_PART),
                          ("response", RESPONSE_PART)):
        marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
        found_at = -1
        for i in range(len(full_ids) - len(marker_ids) + 1):
            if full_ids[i:i + len(marker_ids)] == marker_ids:
                found_at = i
                break
        status = f"found at token {found_at}" if found_at >= 0 else "NOT FOUND"
        shown = marker.replace("\n", "\\n")
        print(f"  {label:12s} {shown!r:52s}")
        print(f"               ids {marker_ids} : {status}")
        if found_at < 0:
            ok = False

    # Double-BOS check
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        with_special = tokenizer(sample_text, add_special_tokens=True)["input_ids"]
        if len(with_special) >= 2 and with_special[0] == bos_id and with_special[1] == bos_id:
            print(f"\n  NOTE: BOS token appears twice at the start.")
            print(f"  The template emits bos_token and the tokenizer adds another.")
            print(f"  Harmless in practice but wastes a token; remove '{{{{- bos_token }}}}'")
            print(f"  from LLAMA3_CHAT_TEMPLATE if you want it clean.")

    if not ok:
        print("\n  A marker did not tokenize consistently in context.")
        print("  train_on_responses_only will mask everything and training")
        print("  will be a no-op. Fix the template before proceeding.")
    print("=" * 70 + "\n")
    return ok


def report_length_stats(tokenizer, dataset, max_seq_length, name="train"):
    """How many examples get truncated at max_seq_length.

    Truncation that removes the assistant turn produces the same
    all-masked symptom as a marker mismatch, so it is worth separating
    the two causes.
    """
    print("=" * 70)
    print(f"TOKEN LENGTH REPORT ({name})")
    print("=" * 70)
    lengths = []
    n = min(len(dataset), 500)
    for i in range(n):
        ids = tokenizer(dataset[i]["text"], add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))
    lengths = np.array(lengths)
    print(f"  sampled {n} examples")
    print(f"  median {np.median(lengths):.0f}  p90 {np.percentile(lengths, 90):.0f}  "
          f"p99 {np.percentile(lengths, 99):.0f}  max {lengths.max()}")
    over = int((lengths > max_seq_length).sum())
    print(f"  over max_seq_length={max_seq_length}: {over}/{n} "
          f"({100 * over / n:.1f}%)")
    if over > 0.05 * n:
        print(f"  *** more than 5% will be truncated; raise max_seq_length ***")
    print("=" * 70 + "\n")
    return lengths

# ==============================================================================
# 6. PRE-TRAINING MASKING DIAGNOSTIC
# ==============================================================================

def report_masking_health(trainer, tokenizer, n_samples=3):
    print("\n" + "=" * 70)
    print("MASKING DIAGNOSTIC")
    print("=" * 70)
    all_ok = True
    for split_name in ("train_dataset", "eval_dataset"):
        ds = getattr(trainer, split_name, None)
        if ds is None:
            continue
        print(f"\n--- {split_name} ---")
        if len(ds) == 0:
            print("  *** BROKEN: dataset is empty (all samples were dropped) ***")
            all_ok = False
            continue
        ratios = []
        for i in range(min(n_samples, len(ds))):
            ex = ds[i]
            n_total = len(ex["labels"])
            n_unmasked = sum(1 for l in ex["labels"] if l != -100)
            ratio = n_unmasked / n_total if n_total else 0
            ratios.append(ratio)
            print(f"  example {i}: {n_unmasked}/{n_total} unmasked ({ratio:.1%})")
            if i == 0:
                unmasked_ids = [l for l in ex["labels"] if l != -100]
                if unmasked_ids:
                    snippet = tokenizer.decode(unmasked_ids[:60], skip_special_tokens=False)
                    print(f"    first 60 unmasked tokens decoded:")
                    print(f"    {snippet!r}")
        avg = sum(ratios) / len(ratios) if ratios else 0
        print(f"  avg unmasked ratio: {avg:.1%}")
        if avg < 0.05:
            print(f"  *** BROKEN: almost no tokens unmasked ***")
            all_ok = False
        elif avg > 0.95:
            print(f"  *** BROKEN: prompt is not being masked ***")
            all_ok = False
    print("=" * 70 + "\n")
    return all_ok

# ==============================================================================
# 7. POST-TRAINING SANITY CHECK
# ==============================================================================

def quick_generation_check(model, tokenizer, dataset, n=3):
    print("\n" + "=" * 70)
    print(f"QUICK GENERATION CHECK -- {n} val examples, raw output")
    print("=" * 70)
    model.eval()
    FastLanguageModel.for_inference(model)

    for i in range(min(n, len(dataset))):
        messages = dataset[i]["messages"]
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        gt = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        inputs = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs, max_new_tokens=400,
                use_cache=True, do_sample=False,
            )
        response = tokenizer.decode(
            outputs[0][len(inputs[0]):], skip_special_tokens=False,
        )
        print(f"\n--- Example {i} ---")
        print(f"Output length: {len(response)} chars")
        print(f"Repr (first 500):\n  {response[:500]!r}")
        print(f"Ground truth (first 300):\n  {gt[:300]}")
    print("=" * 70 + "\n")

# ==============================================================================
# 8. FULL EVALUATION
# ==============================================================================

def extract_is_safe(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, dict) and "is_safe" in data:
                return bool(data["is_safe"])
        except json.JSONDecodeError:
            pass
    m = re.search(r'"is_safe"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    # Fallback for leftover Llama Guard 2 outputs ("safe" / "unsafe\nS1")
    m = re.search(r"^\s*(safe|unsafe)\b", text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).lower() == "safe"
    return None


def run_evaluation(model, tokenizer, dataset, split_name="Test"):
    print(f"\n--- Running Full Evaluation on {split_name} Set ---")
    model.eval()
    FastLanguageModel.for_inference(model)

    y_true, y_pred, y_scores = [], [], []
    n_unparseable_pred = 0
    n_skipped_gt = 0

    for i in tqdm(range(len(dataset))):
        messages = dataset[i]["messages"]
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        gt_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        gt_safe = extract_is_safe(gt_msg)
        if gt_safe is None:
            n_skipped_gt += 1
            continue
        y_true.append(0 if gt_safe else 1)

        inputs = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs, max_new_tokens=1536,
                use_cache=True, do_sample=False,
            )
        response = tokenizer.decode(
            outputs[0][len(inputs[0]):], skip_special_tokens=True,
        )

        pred_safe = extract_is_safe(response)
        if pred_safe is None:
            n_unparseable_pred += 1
            pred_safe = True
        y_pred.append(0 if pred_safe else 1)
        y_scores.append(0 if pred_safe else 1)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall_curve, precision)

    print(f"{split_name} Metrics:")
    print(f"  Accuracy:                 {acc:.4f}")
    print(f"  Recall (unsafe class):    {recall:.4f}")
    print(f"  F1 (unsafe class):        {f1:.4f}")
    print(f"  AUPRC (placeholder):      {auprc:.4f}")
    print(f"  Unparseable predictions:  {n_unparseable_pred}/{len(dataset)}")
    print(f"  Skipped (bad ground truth): {n_skipped_gt}")
    return {
        "acc": acc, "recall": recall, "f1": f1, "auprc": auprc,
        "n_unparseable_pred": n_unparseable_pred,
        "n_skipped_gt": n_skipped_gt,
    }

# ==============================================================================
# 9. MAIN
# ==============================================================================

def main():
    BASE_MODEL = "meta-llama/Meta-Llama-Guard-2-8B"
    OUTPUT_DIR = "SFT/new_outputs/Llama-Guard-2-8B"
    MAX_SEQ_LENGTH = 2048

    print(f"Loading base model: {BASE_MODEL}")
    print()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit   = True,
        device_map     = "auto",
    )

    # ==========================================================================
    # Override Llama Guard 2's classifier template with Llama-3 chat format
    # ==========================================================================
    # Llama Guard 2 ships a template hardcoded for safety classification
    # that renders a "Task: Check if there is unsafe content..." wrapper and
    # drops the assistant turn entirely. We replace it with the plain
    # Llama-3 chat format so arbitrary input/output pairs can be trained on.
    print("--- Before override ---")
    _existing = tokenizer.chat_template or "(none - model ships no chat template)"
    print(f"Tokenizer template starts with: {_existing[:200]!r}")
    print()

    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    print("Overrode tokenizer.chat_template with Llama-3 chat format.")
    print()

    sample_msgs = [
        {"role": "system", "content": "test system"},
        {"role": "user", "content": "test user"},
        {"role": "assistant", "content": "test assistant"},
    ]
    rendered = tokenizer.apply_chat_template(
        sample_msgs, tokenize=False, add_generation_prompt=False)
    print("--- After override: rendered sample ---")
    print(repr(rendered))
    print()
    assert "<|start_header_id|>system<|end_header_id|>\n\ntest system" in rendered, \
        "Chat template override failed: system header block missing"
    assert "<|start_header_id|>user<|end_header_id|>\n\ntest user" in rendered, \
        "Chat template override failed: user header block missing"
    assert "<|start_header_id|>assistant<|end_header_id|>\n\ntest assistant" in rendered, \
        "Chat template override failed: assistant header block missing"
    print("Sanity check passed: Llama-3 chat structure confirmed.\n")

    # Verify the masking markers survive tokenization in context. This is
    # the check that catches a template/tokenizer mismatch before training.
    if not check_marker_tokenization(tokenizer, rendered):
        print("ABORTING: marker tokenization check failed.")
        return

    # ==========================================================================
    # Standard SFT from here
    # ==========================================================================
    model = FastLanguageModel.get_peft_model(
        model,
        r              = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha     = 32,
        lora_dropout   = 0,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 3407,
    )

    train_ds = load_dataset(
        "json", data_files="Claude/SFT/new_data_chatml_llama_and_llamaguard/train.jsonl", split="train",
    )
    val_ds = load_dataset(
        "json", data_files="Claude/SFT/new_data_chatml_llama_and_llamaguard/val.jsonl", split="train",
    )
    print(f"Train: {len(train_ds)} examples  |  Val: {len(val_ds)} examples")

    train_dataset = train_ds.map(
        format_for_sft, batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=train_ds.column_names, num_proc=2,
    )
    eval_dataset = val_ds.map(
        format_for_sft, batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=val_ds.column_names, num_proc=2,
    )

    # Truncation produces the same all-masked symptom as a marker mismatch,
    # so measure it separately.
    report_length_stats(tokenizer, train_dataset, MAX_SEQ_LENGTH, name="train")

    is_bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        callbacks     = [EarlyStoppingCallback(early_stopping_patience=3)],
        args = SFTConfig(
            dataset_text_field           = "text",
            dataset_num_proc             = 2,
            remove_unused_columns        = True,
            max_seq_length               = MAX_SEQ_LENGTH,
            per_device_train_batch_size  = 2,
            per_device_eval_batch_size   = 2,
            gradient_accumulation_steps  = 4,
            num_train_epochs             = 3,
            learning_rate                = 1e-4,
            bf16                         = is_bf16_supported,
            fp16                         = not is_bf16_supported,
            max_grad_norm                = 1.0,
            warmup_steps                 = 20,
            lr_scheduler_type            = "cosine",
            optim                        = "adamw_8bit",
            weight_decay                 = 0.01,
            output_dir                   = OUTPUT_DIR,
            save_strategy                = "steps",
            save_steps                   = 50,
            save_total_limit             = 50,
            eval_strategy                = "steps",
            eval_steps                   = 50,
            logging_steps                = 10,
            load_best_model_at_end       = True,
            metric_for_best_model        = "eval_loss",
            greater_is_better            = False,
            report_to                    = "none",
            seed                         = 3407,
        ),
    )

    # Llama-3 header markers, not Llama-2 [INST] and not ChatML.
    trainer = train_on_responses_only(
        trainer,
        instruction_part = INSTRUCTION_PART,
        response_part    = RESPONSE_PART,
    )

    for attr in ("train_dataset", "eval_dataset"):
        ds = getattr(trainer, attr, None)
        if ds is not None and "text" in ds.column_names:
            setattr(trainer, attr, ds.remove_columns(["text"]))

    if not report_masking_health(trainer, tokenizer):
        print("ABORTING: masking diagnostic failed.")
        print("Marker tokenization passed, so the likely cause is truncation.")
        print("Check the TOKEN LENGTH REPORT above and raise max_seq_length")
        print("if a large fraction of examples exceed it.")
        return

    trainer.train()

    model.save_pretrained_merged(
        f"{OUTPUT_DIR}/final", tokenizer, save_method="merged_16bit",
    )

    quick_generation_check(model, tokenizer, val_ds, n=3)
    run_evaluation(model, tokenizer, val_ds, split_name="Validation")


if __name__ == "__main__":
    main()