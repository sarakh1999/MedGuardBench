"""
DPO stage for MedGuardBench, starting from the clean-data SFT policy.

Sits between SFT and GRPO as a cheaper preference stage: no on-policy
sampling during training, one forward/backward per pair for policy and
reference. Pairs come from build_dpo_pairs.py (verifiable, label-anchored).

Reference policy: as in train_grpo.py, the policy is loaded from the MERGED
SFT weights with a fresh LoRA on top, so TRL's adapter-disabled reference
IS the SFT policy. Loading the SFT adapter directly would make the reference
the untuned base model and the KL anchor meaningless.

Usage (GPU node):
    source Claude/SFT/gpu_env.sh
    python Claude/DPO/train_dpo.py                       # grpo_config paths
    python Claude/DPO/train_dpo.py --pairs X --policy-init Y --out Z
Then:
    python Claude/SFT/eval_sft.py --model qwen3-4b --adapter <out>/final_adapter
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.utils._pytree
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda cls: cls
for _i in range(1, 8):
    if not hasattr(torch, f"int{_i}"):
        setattr(torch, f"int{_i}", torch.int8)
    if not hasattr(torch, f"uint{_i}"):
        setattr(torch, f"uint{_i}", torch.uint8)

from unsloth import FastLanguageModel, PatchDPOTrainer  # noqa: E402
PatchDPOTrainer()
from trl import DPOConfig, DPOTrainer                   # noqa: E402
from datasets import load_dataset                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "GRPO"))
sys.path.insert(0, os.path.join(HERE, "..", "SFT"))
import grpo_config as C                 # noqa: E402
from data_utils import apply_template_override  # noqa: E402

DPO_DIR = C.PROJECT_ROOT / "Claude" / "DPO"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=str(DPO_DIR / "data" / C.DATA_CONDITION / "dpo_pairs.jsonl"))
    ap.add_argument("--policy-init", default=str(C.POLICY_INIT_PATH),
                    help="merged SFT model dir (see docstring)")
    ap.add_argument("--out", default=str(DPO_DIR / "outputs" / C.DATA_CONDITION / "qwen3-4b"))
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rpo-alpha", type=float, default=0.0,
                    help="weight of the SFT/NLL auxiliary loss on the chosen answer "
                         "(0 = pure DPO; ~0.5-1.0 anchors the policy to SFT behavior)")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=C.MAX_SEQ_LENGTH)
    ap.add_argument("--max-prompt-length", type=int, default=C.MAX_PROMPT_LENGTH)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--load-in-4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    ap.add_argument("--save-all", action="store_true",
                    help="keep every checkpoint instead of the last 3 "
                         "(enables post-hoc mid-run evals)")
    args = ap.parse_args()

    if not os.path.exists(args.pairs):
        sys.exit(f"pairs not found: {args.pairs}  (run build_dpo_pairs.py)")
    if not os.path.isdir(args.policy_init):
        sys.exit(f"policy init not found: {args.policy_init}\n"
                 "merge the SFT adapter first: python Claude/SFT/merge_adapter.py ...")
    os.makedirs(args.out, exist_ok=True)

    bf16 = torch.cuda.is_bf16_supported()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.policy_init, max_seq_length=args.max_length,
        load_in_4bit=args.load_in_4bit, device_map="auto",
    )
    apply_template_override(tokenizer, args.policy_init)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model, r=C.LORA_RANK, lora_alpha=C.LORA_ALPHA, lora_dropout=0, bias="none",
        target_modules=C.LORA_TARGET_MODULES,
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )

    ds = load_dataset("json", data_files=args.pairs, split="train")
    ds = ds.remove_columns([c for c in ds.column_names
                            if c not in ("prompt", "chosen", "rejected")])
    split = ds.train_test_split(test_size=args.val_fraction, seed=args.seed) \
        if args.val_fraction > 0 else {"train": ds, "test": None}
    print(f"pairs: train {len(split['train'])}"
          + (f"  val {len(split['test'])}" if split["test"] is not None else ""))

    cfg = DPOConfig(
        output_dir=args.out, beta=args.beta,
        rpo_alpha=(args.rpo_alpha if args.rpo_alpha > 0 else None),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length, max_prompt_length=args.max_prompt_length,
        bf16=bf16, fp16=not bf16, max_grad_norm=1.0, warmup_ratio=0.1,
        lr_scheduler_type="cosine", optim="adamw_8bit", weight_decay=0.0,
        logging_steps=5, save_strategy="steps", save_steps=50,
        save_total_limit=(None if args.save_all else 3),
        eval_strategy="steps" if split["test"] is not None else "no", eval_steps=50,
        load_best_model_at_end=split["test"] is not None,
        metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", seed=args.seed, remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model, ref_model=None, args=cfg,
        train_dataset=split["train"], eval_dataset=split["test"],
        processing_class=tokenizer,
    )

    t0 = time.time()
    result = trainer.train()
    secs = time.time() - t0

    final = os.path.join(args.out, "final_adapter")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    summary = {
        "policy_init": args.policy_init, "pairs": args.pairs,
        "n_train_pairs": len(split["train"]),
        "beta": args.beta, "rpo_alpha": args.rpo_alpha,
        "lr": args.lr, "epochs": args.epochs, "seed": args.seed,
        "train_loss": result.training_loss, "train_seconds": secs,
        "best_eval_loss": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "log_history": trainer.state.log_history,
    }
    with open(os.path.join(args.out, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {final}  ({secs/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
