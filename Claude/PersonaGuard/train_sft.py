"""
SFT for personalized guardrails (Qwen3 family, Unsloth + LoRA).

Training text per example (see Claude/SFT/qwen_think.py):
  <|im_start|>system  guardrail instructions + [USER PROFILE] <|im_end|>
  <|im_start|>user    [REQUEST] ... <|im_end|>
  <|im_start|>assistant
  <think> teacher reasoning </think>
  {"action": ..., "triggering_attributes": [...], ..., "is_safe": ...}<|im_end|>
Loss only on the assistant part.

Usage (from repo root):
  python Claude/PersonaGuard/train_sft.py --model Qwen/Qwen3-4B-Instruct-2507 --run qwen3-4b
  python Claude/PersonaGuard/train_sft.py --model Qwen/Qwen3-8B --run qwen3-8b --max_seq_length 4096
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "SFT"))

from unsloth import FastLanguageModel  # noqa: E402  (must precede transformers)
from unsloth.chat_templates import train_on_responses_only  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import EarlyStoppingCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from qwen_think import training_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--run", default="qwen3-4b")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data", "chatml"))
    ap.add_argument("--out_root", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--max_seq_length", type=int, default=3072)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=-1, help="e.g. 100 for a smoke test")
    args = ap.parse_args()
    out_dir = os.path.join(args.out_root, args.run)

    model, tok = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=args.max_seq_length,
                                                   load_in_4bit=True, device_map="auto")
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)

    def fmt(batch):
        return {"text": [training_text(tok, m) for m in batch["messages"]]}

    ds = {s: load_dataset("json", data_files=os.path.join(args.data_dir, f"{s}.jsonl"), split="train")
          for s in ("train", "val")}
    ds = {s: d.map(fmt, batched=True, remove_columns=d.column_names) for s, d in ds.items()}
    n_long = sum(len(tok(t).input_ids) > args.max_seq_length for t in ds["train"]["text"][:500])
    print(f"train {len(ds['train'])} | val {len(ds['val'])} | {n_long}/500 sampled examples exceed max_seq_length")
    print("example assistant part:\n", ds["train"][0]["text"].split("<|im_start|>assistant\n")[-1][:400])

    bf16 = torch.cuda.is_bf16_supported()
    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds["train"], eval_dataset=ds["val"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=args.max_seq_length, output_dir=out_dir,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, max_steps=args.max_steps,
            learning_rate=args.lr, warmup_steps=20, lr_scheduler_type="cosine", optim="adamw_8bit",
            weight_decay=0.01, max_grad_norm=1.0, bf16=bf16, fp16=not bf16,
            eval_strategy="steps", eval_steps=50, save_strategy="steps", save_steps=50, save_total_limit=10,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            logging_steps=10, report_to="none", seed=3407))
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")
    ex = trainer.train_dataset[0]
    frac = sum(l != -100 for l in ex["labels"]) / len(ex["labels"])
    print(f"unmasked fraction on example 0: {frac:.1%} (expect ~25-60%)")
    if frac < 0.05:
        raise SystemExit("Masking looks wrong (almost nothing is trained). Check the chat template.")
    trainer.train()
    model.save_pretrained(os.path.join(out_dir, "final"))
    tok.save_pretrained(os.path.join(out_dir, "final"))
    print(f"saved LoRA adapter -> {out_dir}/final")


if __name__ == "__main__":
    main()
