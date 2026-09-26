"""
SFT for LlamaGuard-7b on the clinical safety dataset.

WHY THIS DIFFERS FROM THE QWEN VERSION
======================================
The Qwen scripts override the tokenizer's chat template with ChatML
(<|im_start|> / <|im_end|>). That works there because Qwen tokenizers
carry those markers as genuine single tokens.

LlamaGuard-7b is Llama-2 based with a legacy SentencePiece tokenizer.
<|im_start|> is NOT in its vocabulary, so it fragments into roughly
'<', '|', 'im', '_', 'start', '|', '>'. SentencePiece is also whitespace
sensitive: the same characters tokenize differently in isolation than
they do mid-sequence. train_on_responses_only finds the response
boundary by searching for the tokenized marker as a subsequence, so the
search fails and every sample ends up fully masked:

    Removed 3772 out of 3772 samples where all labels were -100

The fix is to use Llama-2's native chat format, whose markers tokenize
consistently:

    [INST] <<SYS>>
    {system}
    <</SYS>>

    {user} [/INST] {assistant}</s>

LlamaGuard's own shipped template is a hardcoded safety-classifier
wrapper (it renders "[INST] Task: Check if there is unsafe content...")
which discards the assistant turn, so it still has to be replaced. We
keep the weights and swap the template, same idea as the Qwen3Guard
script, just landing on Llama-2 format rather than ChatML.
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
# 3. LLAMA-2 CHAT TEMPLATE
# ==============================================================================
# Native Llama-2 format. The system message is folded into the first user
# turn between <<SYS>> markers, which is what Llama-2 was trained on.
#
# Markers used for response masking:
#   instruction_part = "[INST]"
#   response_part    = "[/INST]"
#
# These tokenize stably under SentencePiece, unlike ChatML markers.

LLAMA2_CHAT_TEMPLATE = (
    "{%- set ns = namespace(system='') %}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{%- set ns.system = message['content'] %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- set loop_messages = messages | rejectattr('role', 'equalto', 'system') | list %}"
    "{%- for message in loop_messages %}"
    "{%- if message['role'] == 'user' %}"
    "{%- if loop.first and ns.system %}"
    "{{- '[INST] <<SYS>>\n' + ns.system + '\n<</SYS>>\n\n' + message['content'] + ' [/INST]' }}"
    "{%- else %}"
    "{{- '[INST] ' + message['content'] + ' [/INST]' }}"
    "{%- endif %}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- ' ' + message['content'] + eos_token }}"
    "{%- endif %}"
    "{%- endfor %}"
)

INSTRUCTION_PART = "[INST]"
RESPONSE_PART = "[/INST]"

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
    """Verify the response marker is findable as a token subsequence.

    This is the check that would have caught the ChatML failure before
    burning a model load. It tokenizes the marker on its own and then
    looks for that exact id sequence inside a real rendered example.
    """
    print("\n" + "=" * 70)
    print("MARKER TOKENIZATION CHECK")
    print("=" * 70)

    full_ids = tokenizer(sample_text, add_special_tokens=False)["input_ids"]

    ok = True
    for label, marker in (("instruction", INSTRUCTION_PART),
                          ("response", RESPONSE_PART)):
        marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
        # Naive subsequence search
        found_at = -1
        for i in range(len(full_ids) - len(marker_ids) + 1):
            if full_ids[i:i + len(marker_ids)] == marker_ids:
                found_at = i
                break
        status = f"found at token {found_at}" if found_at >= 0 else "NOT FOUND"
        print(f"  {label:12s} {marker!r:12s} -> {marker_ids} : {status}")
        if found_at < 0:
            ok = False

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
    # Fallback for leftover LlamaGuard-style outputs ("safe" / "unsafe\nO3")
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
    BASE_MODEL = "llamas-community/LlamaGuard-7b"
    # MAX_SEQ_LENGTH = 2048
    MAX_SEQ_LENGTH = 4096

    print(f"Loading base model: {BASE_MODEL}")
    print()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit   = True,
        device_map     = "auto",
    )

    # ==========================================================================
    # Override LlamaGuard's classifier template with Llama-2 chat format
    # ==========================================================================
    # LlamaGuard ships a template hardcoded for safety classification that
    # renders "[INST] Task: Check if there is unsafe content..." and drops
    # the assistant turn entirely. We replace it with the plain Llama-2
    # chat format so arbitrary input/output pairs can be trained on.
    print("--- Before override ---")
    _existing = tokenizer.chat_template or "(none - model ships no chat template)"
    print(f"Tokenizer template starts with: {_existing[:200]!r}")
    print()

    tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE
    print("Overrode tokenizer.chat_template with Llama-2 chat format.")
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
    assert "[INST]" in rendered, \
        "Chat template override failed: no [INST] marker in rendered output"
    assert "[/INST] test assistant" in rendered, \
        "Chat template override failed: assistant content not after [/INST]"
    assert "<<SYS>>\ntest system" in rendered, \
        "Chat template override failed: system message not in <<SYS>> block"
    print("Sanity check passed: Llama-2 chat structure confirmed.\n")

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
            output_dir                   = "SFT/new_outputs/LlamaGuard-7b",
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

    # Llama-2 markers, not ChatML.
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
        "SFT/new_outputs/LlamaGuard-7b/final", tokenizer, save_method="merged_16bit",
    )

    quick_generation_check(model, tokenizer, val_ds, n=3)
    run_evaluation(model, tokenizer, val_ds, split_name="Validation")


if __name__ == "__main__":
    main()