"""
Parametrised Unsloth LoRA SFT for every MedGuardBench model family.

One script replaces the eleven per-model sft_*.py files. Hyperparameters are
identical to those scripts (r=16, alpha=32, lr 1e-4, 3 epochs, bs 2 x 4 accum,
cosine, adamw_8bit, early stopping on eval_loss, train_on_responses_only).
What differs per model is held in MODELS below: the HF id, the chat template
that has to be forced onto the tokenizer, the masking markers, the ChatML data
directory and the max sequence length.

Why a template override at all: every guard model (Qwen3Guard, LlamaGuard,
ShieldGemma) ships a classifier template that discards the assistant turn, so
train_on_responses_only finds nothing to train on. We keep the weights and
swap in the family's native generative template.

Usage (from the repo root, on a GPU node):
    source Claude/SFT/gpu_env.sh
    python Claude/SFT/sft_train.py --model qwen3-4b
    python Claude/SFT/sft_train.py --model qwen3-4b --seed 1 --tag seed1
    python Claude/SFT/sft_train.py --list
"""

import argparse
import json
import os
import sys
import time

import torch

# ------------------------------------------------------------------ patches
import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    def register_constant(cls):
        return cls
    torch.utils._pytree.register_constant = register_constant

for _i in range(1, 8):
    if not hasattr(torch, f"int{_i}"):
        setattr(torch, f"int{_i}", torch.int8)
    if not hasattr(torch, f"uint{_i}"):
        setattr(torch, f"uint{_i}", torch.uint8)

from unsloth import FastLanguageModel                      # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402
from trl import SFTTrainer, SFTConfig                        # noqa: E402
from datasets import load_dataset                            # noqa: E402
from transformers import EarlyStoppingCallback               # noqa: E402

# ------------------------------------------------------------------ templates
CHATML_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\n' }}"
    "{%- endif %}"
)

LLAMA3_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{%- endif %}"
)

# Llama-2: system folded into the first user turn. The user turn ends with
# [/INST], so the generation prompt is implicit.
LLAMA2_TEMPLATE = (
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

# Gemma: no system role (folded into first user turn), assistant -> model.
GEMMA_TEMPLATE = (
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

FAMILIES = {
    "chatml": dict(template=CHATML_TEMPLATE,
                   instruction_part="<|im_start|>user\n",
                   response_part="<|im_start|>assistant\n"),
    "llama3": dict(template=LLAMA3_TEMPLATE,
                   instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                   response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "llama2": dict(template=LLAMA2_TEMPLATE,
                   instruction_part="[INST]",
                   response_part="[/INST]"),
    "gemma":  dict(template=GEMMA_TEMPLATE,
                   instruction_part="<start_of_turn>user\n",
                   response_part="<start_of_turn>model\n"),
}

DATA_QWEN = "Claude/SFT/new_data_chatml_qwen_and_qwenguard"
DATA_LLAMA = "Claude/SFT/new_data_chatml_llama_and_llamaguard"
DATA_GEMMA = "Claude/SFT/new_data_chatml_ShieldGemma"

# `override_template=False` keeps the model's own template (Qwen3-Instruct and
# Qwen3-8B/14B have a normal ChatML template with a <think> block that
# add_generation_prompt handles; the per-model scripts did not override it).
MODELS = {
    "qwen3-4b":       dict(hf="Qwen/Qwen3-4B-Instruct-2507", family="chatml", override_template=False,
                           data=DATA_QWEN, out="Qwen3-4B-Instruct", max_seq=2048),
    "qwen3-8b":       dict(hf="Qwen/Qwen3-8B", family="chatml", override_template=False,
                           data=DATA_QWEN, out="Qwen3-8B-Instruct", max_seq=2048),
    "qwen3-14b":      dict(hf="Qwen/Qwen3-14B", family="chatml", override_template=False,
                           data=DATA_QWEN, out="Qwen3-14B-Instruct", max_seq=2048),
    "qwen3guard-4b":  dict(hf="Qwen/Qwen3Guard-Gen-4B", family="chatml", override_template=True,
                           data=DATA_QWEN, out="Qwen3Guard-Gen-4B", max_seq=2048),
    "qwen3guard-8b":  dict(hf="Qwen/Qwen3Guard-Gen-8B", family="chatml", override_template=True,
                           data=DATA_QWEN, out="Qwen3Guard-Gen-8B", max_seq=2048),
    # Llama-3.1-8B is a base model with no chat template; the old script forced
    # ChatML onto it, whose markers are not Llama-3 tokens. The native Llama-3
    # header format is used here instead, same as Llama Guard 2.
    "llama3.1-8b":    dict(hf="meta-llama/Llama-3.1-8B", family="llama3", override_template=True,
                           data=DATA_LLAMA, out="Llama-3.1-8B", max_seq=2048),
    "llamaguard-7b":  dict(hf="llamas-community/LlamaGuard-7b", family="llama2", override_template=True,
                           data=DATA_LLAMA, out="LlamaGuard-7b", max_seq=4096),
    "llamaguard2-8b": dict(hf="meta-llama/Meta-Llama-Guard-2-8B", family="llama3", override_template=True,
                           data=DATA_LLAMA, out="Llama-Guard-2-8B", max_seq=2048),
    "shieldgemma-2b": dict(hf="google/shieldgemma-2b", family="gemma", override_template=True,
                           data=DATA_GEMMA, out="ShieldGemma-2B", max_seq=2048),
    "shieldgemma-9b": dict(hf="google/shieldgemma-9b", family="gemma", override_template=True,
                           data=DATA_GEMMA, out="ShieldGemma-9B", max_seq=2048),
}

OUTPUT_ROOT = os.environ.get("SFT_OUTPUT_ROOT", "Claude/SFT/outputs_blind_v2")


def apply_template(tokenizer, spec):
    fam = FAMILIES[spec["family"]]
    if spec["override_template"] or not tokenizer.chat_template:
        tokenizer.chat_template = fam["template"]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return fam["instruction_part"], fam["response_part"]


def format_for_sft(examples, tokenizer):
    return {"text": [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                     for c in examples["messages"]]}


def report_masking_health(trainer, tokenizer, n_samples=3):
    print("\n" + "=" * 70 + "\nMASKING DIAGNOSTIC\n" + "=" * 70)
    ok = True
    for name in ("train_dataset", "eval_dataset"):
        ds = getattr(trainer, name, None)
        if ds is None:
            continue
        ratios = []
        for i in range(min(n_samples, len(ds))):
            labels = ds[i]["labels"]
            un = [l for l in labels if l != -100]
            ratios.append(len(un) / max(len(labels), 1))
            if i == 0 and un:
                print(f"  {name} first unmasked tokens: "
                      f"{tokenizer.decode(un[:40], skip_special_tokens=False)!r}")
        avg = sum(ratios) / len(ratios) if ratios else 0
        print(f"  {name}: avg unmasked ratio {avg:.1%}")
        if avg < 0.05 or avg > 0.95:
            print("  *** BROKEN masking ***")
            ok = False
    print("=" * 70)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), help="model key")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--data-dir", default=None, help="override ChatML dir")
    ap.add_argument("--output-dir", default=None, help="override output dir")
    ap.add_argument("--tag", default=None, help="suffix for output dir (e.g. seed1)")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--save-total-limit", type=int, default=50)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--merge", action="store_true",
                    help="also save a merged 16-bit model under <out>/final_merged")
    ap.add_argument("--max-train", type=int, default=None, help="debug: cap train rows")
    ap.add_argument("--max-val", type=int, default=None, help="debug: cap val rows")
    ap.add_argument("--max-seq", type=int, default=None,
                    help="override the model's max_seq_length (e.g. 4096 for context-unrolled targets)")
    ap.add_argument("--init-adapter", default=None,
                    help="continue training from an existing LoRA adapter dir (e.g. an SFT checkpoint) "
                         "instead of attaching a fresh adapter to the base model")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint-* in the output dir if present")
    args = ap.parse_args()

    if args.list:
        for k, v in MODELS.items():
            print(f"{k:<16} {v['hf']:<40} family={v['family']:<7} data={v['data']}")
        return 0
    if not args.model:
        ap.error("--model is required")

    spec = dict(MODELS[args.model])
    if args.max_seq:
        spec["max_seq"] = args.max_seq
    data_dir = args.data_dir or spec["data"]
    out_dir = args.output_dir or os.path.join(OUTPUT_ROOT, spec["out"] + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(out_dir, exist_ok=True)

    print(f"model={spec['hf']}  family={spec['family']}  data={data_dir}\n"
          f"out={out_dir}  seed={args.seed}  epochs={args.epochs}")
    bf16 = torch.cuda.is_bf16_supported()
    if not bf16:
        print("bfloat16 unavailable on this GPU: training in float16")

    if args.init_adapter:
        # Continue training an existing LoRA adapter (the base model is read
        # from the adapter's config). The adapter's own r/alpha/targets apply.
        print(f"continuing from adapter: {args.init_adapter}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.init_adapter, max_seq_length=spec["max_seq"],
            load_in_4bit=True, device_map="auto",
        )
        instruction_part, response_part = apply_template(tokenizer, spec)
        FastLanguageModel.for_training(model)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=spec["hf"], max_seq_length=spec["max_seq"],
            load_in_4bit=True, device_map="auto",
        )
        instruction_part, response_part = apply_template(tokenizer, spec)

        model = FastLanguageModel.get_peft_model(
            model, r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32, lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth", random_state=args.seed,
        )

    train_ds = load_dataset("json", data_files=f"{data_dir}/train.jsonl", split="train")
    val_ds = load_dataset("json", data_files=f"{data_dir}/val.jsonl", split="train")
    if args.max_train:
        train_ds = train_ds.select(range(min(args.max_train, len(train_ds))))
    if args.max_val:
        val_ds = val_ds.select(range(min(args.max_val, len(val_ds))))
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    fmt = dict(batched=True, fn_kwargs={"tokenizer": tokenizer}, num_proc=2)
    train_text = train_ds.map(format_for_sft, remove_columns=train_ds.column_names, **fmt)
    val_text = val_ds.map(format_for_sft, remove_columns=val_ds.column_names, **fmt)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_text, eval_dataset=val_text,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        args=SFTConfig(
            dataset_text_field="text", dataset_num_proc=2, remove_unused_columns=True,
            max_seq_length=spec["max_seq"],
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs, learning_rate=args.lr,
            bf16=bf16, fp16=not bf16, max_grad_norm=1.0, warmup_steps=20,
            lr_scheduler_type="cosine", optim="adamw_8bit", weight_decay=0.01,
            output_dir=out_dir, save_strategy="steps", save_steps=args.eval_steps,
            save_total_limit=args.save_total_limit, eval_strategy="steps",
            eval_steps=args.eval_steps, logging_steps=10,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, report_to="none", seed=args.seed,
        ),
    )
    trainer = train_on_responses_only(trainer, instruction_part=instruction_part,
                                      response_part=response_part)
    for attr in ("train_dataset", "eval_dataset"):
        ds = getattr(trainer, attr, None)
        if ds is not None and "text" in ds.column_names:
            setattr(trainer, attr, ds.remove_columns(["text"]))

    if not report_masking_health(trainer, tokenizer):
        print("ABORTING: masking diagnostic failed.")
        return 1

    # Resume from the latest trainer checkpoint in out_dir if one exists
    # (lets a run interrupted by a node time limit continue elsewhere).
    resume = None
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(out_dir)
        print(f"resume_from_checkpoint={resume}" if resume else "no checkpoint to resume, starting fresh")
        if resume and getattr(trainer.accelerator, "scaler", None) is None:
            # Unsloth handles fp16 without an accelerate GradScaler; Trainer
            # would still try to load scaler.pt into it and crash. The loss
            # scale simply re-warms.
            sp = os.path.join(resume, "scaler.pt")
            if os.path.exists(sp):
                os.replace(sp, sp + ".skipped")
                print("scaler.pt set aside (no GradScaler in this trainer)")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    train_seconds = time.time() - t0

    # Best checkpoint (load_best_model_at_end) is what the model now holds.
    final_dir = os.path.join(out_dir, "final_adapter")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved best adapter -> {final_dir}")
    if args.merge:
        merged = os.path.join(out_dir, "final_merged")
        model.save_pretrained_merged(merged, tokenizer, save_method="merged_16bit")
        print(f"saved merged 16-bit -> {merged}")

    summary = {
        "model_key": args.model, "hf_id": spec["hf"], "family": spec["family"],
        "init_adapter": args.init_adapter,
        "data_dir": data_dir, "output_dir": out_dir, "seed": args.seed,
        "epochs": args.epochs, "lr": args.lr,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "train_loss": result.training_loss, "train_seconds": train_seconds,
        "gpu": torch.cuda.get_device_name(0), "precision": "bf16" if bf16 else "fp16",
        "log_history": trainer.state.log_history,
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"best eval_loss {summary['best_eval_loss']} at {summary['best_checkpoint']}  "
          f"({train_seconds/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
