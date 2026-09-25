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
# 3. DATA FORMATTING
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
# 4. PRE-TRAINING DIAGNOSTICS
# ==============================================================================
# A healthy SFT setup has labels=-100 on the prompt and labels=input_ids
# on the assistant response. We expect ~30-60% unmasked on this dataset.

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
# 5. POST-TRAINING SANITY CHECK
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
# 6. FULL EVALUATION
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
# 7. MAIN
# ==============================================================================

def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = "Qwen/Qwen3-14B",  # general chat model, not the safety classifier
        max_seq_length = 2048,
        load_in_4bit   = True,
        device_map     = "auto",
    )

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
        "json", data_files="Claude/SFT/new_data_chatml_qwen_and_qwenguard/train.jsonl", split="train",
    )
    val_ds = load_dataset(
        "json", data_files="Claude/SFT/new_data_chatml_qwen_and_qwenguard/val.jsonl", split="train",
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
            max_seq_length               = 2048,
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
            output_dir                   = "Claude/SFT/new_outputs/Qwen3-14B-Instruct",
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

    # Standard prompt masking. Qwen3-Instruct uses a normal chat template
    # so this works correctly without the workarounds we needed for Qwen3Guard.
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )

    # Strip leftover "text" column.
    for attr in ("train_dataset", "eval_dataset"):
        ds = getattr(trainer, attr, None)
        if ds is not None and "text" in ds.column_names:
            setattr(trainer, attr, ds.remove_columns(["text"]))

    if not report_masking_health(trainer, tokenizer):
        print("ABORTING: masking diagnostic failed.")
        return

    trainer.train()

    model.save_pretrained_merged(
        "SFT/new_outputs/Qwen3-14B-Instruct/final", tokenizer, save_method="merged_16bit",
    )

    quick_generation_check(model, tokenizer, val_ds, n=3)
    run_evaluation(model, tokenizer, val_ds, split_name="Validation")


if __name__ == "__main__":
    main()