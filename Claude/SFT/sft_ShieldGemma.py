"""
SFT for ShieldGemma-2B on the clinical safety dataset.

WHY CHATML IS WRONG HERE
========================
The Qwen scripts override the tokenizer template with ChatML
(<|im_start|> / <|im_end|>). That is correct for Qwen and wrong for
ShieldGemma, for three independent reasons.

1. Different turn markers. ShieldGemma is built on Gemma 2, which uses
   <start_of_turn> and <end_of_turn>. <|im_start|> is not in the Gemma
   vocabulary at all, so it fragments into ordinary sub-tokens and
   train_on_responses_only cannot locate the response boundary. That is
   the same failure that produced

       Removed 3772 out of 3772 samples where all labels were -100

   on LlamaGuard-7b.

2. Different role name. Gemma calls the assistant turn "model", not
   "assistant".

3. No system role. Gemma has no system turn. The official template
   raises an exception if you pass one. Every record in this dataset
   carries a system message, so it must be folded into the first user
   turn. The template below does that, which keeps a single JSONL usable
   across Qwen, Llama, and Gemma rather than needing one copy per family.

Resulting format:

    <bos><start_of_turn>user
    {system}

    {user}<end_of_turn>
    <start_of_turn>model
    {assistant}<end_of_turn>

Masking markers, both real single tokens in the Gemma vocabulary:

    instruction_part = "<start_of_turn>user\n"
    response_part    = "<start_of_turn>model\n"

ShieldGemma also ships a specialized safety-classifier template that
takes a `guideline` argument and emits a "Yes"/"No" policy-violation
verdict. It discards the assistant turn, so it has to be replaced
regardless.

ACCESS
======
google/shieldgemma-2b is gated under the Gemma license. Verify before a
long run:

    python -c "
    from huggingface_hub import hf_hub_download
    print(hf_hub_download('google/shieldgemma-2b','config.json'))
    "

PRECISION WARNING
=================
Gemma 2 uses attention and final logit soft-capping, and is known to
overflow in float16. On a V100 (no bfloat16) this can surface as NaN
loss within the first few hundred steps. The script warns if it detects
that situation. If loss goes NaN, the options are a bf16-capable GPU
(A100/H100) or a lower learning rate, in that order of effectiveness.
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

# ==============================================================================
# 3. GEMMA CHAT TEMPLATE
# ==============================================================================
# Folds the system message into the first user turn, since Gemma has no
# system role, and renames assistant -> model.

GEMMA_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- set ns = namespace(system='') %}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{%- set ns.system = message['content'] %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- set loop_messages = messages | rejectattr('role', 'equalto', 'system') | list %}"
    "{%- for message in loop_messages %}"
    "{%- set role = 'model' if message['role'] == 'assistant' else message['role'] %}"
    "{%- if loop.first and ns.system and role == 'user' %}"
    "{{- '<start_of_turn>user\n' + ns.system + '\n\n' "
    "+ message['content'] | trim + '<end_of_turn>\n' }}"
    "{%- else %}"
    "{{- '<start_of_turn>' + role + '\n' + message['content'] | trim "
    "+ '<end_of_turn>\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<start_of_turn>model\n' }}"
    "{%- endif %}"
)

INSTRUCTION_PART = "<start_of_turn>user\n"
RESPONSE_PART = "<start_of_turn>model\n"

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
# 5. MARKER AND LENGTH DIAGNOSTICS
# ==============================================================================

def check_marker_tokenization(tokenizer, sample_text):
    """Verify the masking markers survive tokenization in context.

    Catches a template/tokenizer mismatch before a full training run
    silently becomes a no-op.
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
        print(f"  {label:12s} {marker.replace(chr(10), chr(92)+'n')!r:28s} "
              f"ids {marker_ids} : {status}")
        if found_at < 0:
            ok = False

    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        with_special = tokenizer(sample_text, add_special_tokens=True)["input_ids"]
        if len(with_special) >= 2 and with_special[:2] == [bos_id, bos_id]:
            print("\n  NOTE: BOS appears twice. The template emits bos_token and")
            print("  the tokenizer adds another. Harmless but wastes a token.")

    if not ok:
        print("\n  A marker did not tokenize consistently in context.")
        print("  train_on_responses_only would mask everything and training")
        print("  would be a no-op. Fix the template before proceeding.")
    print("=" * 70 + "\n")
    return ok


def report_length_stats(tokenizer, dataset, max_seq_length, name="train"):
    """Truncation removing the assistant turn looks identical to a marker
    mismatch, so measure it separately."""
    print("=" * 70)
    print(f"TOKEN LENGTH REPORT ({name})")
    print("=" * 70)
    n = min(len(dataset), 500)
    lengths = np.array([
        len(tokenizer(dataset[i]["text"], add_special_tokens=False)["input_ids"])
        for i in range(n)
    ])
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
    # Fallback for leftover ShieldGemma-style output. ShieldGemma answers
    # "Yes" when the content VIOLATES the policy, so Yes maps to unsafe.
    m = re.search(r"^\s*(yes|no)\b", text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).lower() == "no"
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
    BASE_MODEL = "google/shieldgemma-2b"
    OUTPUT_DIR = "Claude/SFT/new_outputs/ShieldGemma-2B"
    DATA_DIR = "Claude/SFT/new_data_chatml_ShieldGemma"
    MAX_SEQ_LENGTH = 2048

    print(f"Loading base model: {BASE_MODEL}")
    print()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit   = True,
        device_map     = "auto",
    )

    is_bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    if not is_bf16_supported:
        print("!" * 70)
        print("WARNING: bfloat16 is unavailable, so this will run in float16.")
        print("Gemma 2 uses logit soft-capping and is known to overflow in fp16,")
        print("which can show up as NaN loss in the first few hundred steps.")
        print("Watch the loss closely. If it goes NaN, move to a bf16-capable")
        print("GPU (A100/H100) or lower the learning rate.")
        print("!" * 70)
        print()

    # ==========================================================================
    # Replace ShieldGemma's classifier template with plain Gemma chat format
    # ==========================================================================
    print("--- Before override ---")
    _existing = tokenizer.chat_template or "(none - model ships no chat template)"
    print(f"Tokenizer template starts with: {_existing[:200]!r}")
    print()

    tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
    print("Overrode tokenizer.chat_template with Gemma chat format.")
    print("System messages are folded into the first user turn, since Gemma")
    print("has no system role.")
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
    assert "<start_of_turn>user\ntest system\n\ntest user" in rendered, \
        "Template failed: system not folded into the first user turn"
    assert "<start_of_turn>model\ntest assistant" in rendered, \
        "Template failed: assistant turn not rendered under the model role"
    assert "<|im_start|>" not in rendered, \
        "Template failed: ChatML markers leaked into Gemma output"
    print("Sanity check passed: Gemma chat structure confirmed.\n")

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
        "json", data_files=f"{DATA_DIR}/train.jsonl", split="train",
    )
    val_ds = load_dataset(
        "json", data_files=f"{DATA_DIR}/val.jsonl", split="train",
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

    report_length_stats(tokenizer, train_dataset, MAX_SEQ_LENGTH, name="train")

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

    # Gemma turn markers, not ChatML.
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
        print("Check the TOKEN LENGTH REPORT above and raise max_seq_length.")
        return

    trainer.train()

    model.save_pretrained_merged(
        f"{OUTPUT_DIR}/final", tokenizer, save_method="merged_16bit",
    )

    quick_generation_check(model, tokenizer, val_ds, n=3)
    run_evaluation(model, tokenizer, val_ds, split_name="Validation")


if __name__ == "__main__":
    main()