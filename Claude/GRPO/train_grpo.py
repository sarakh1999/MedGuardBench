#!/usr/bin/env python3
"""
GRPO training for MedGuardBench.

Starts from the SFT checkpoint (never the base model) and refines category
attribution using a verifiable reward. See reward.py for the reward design
and mine_hard_examples.py for how the training set is selected.

Usage:
    python train_grpo.py
    python train_grpo.py --seed 1 --run-name grpo-seed1
    python train_grpo.py --dry-run            # 20 steps, sanity check only
    python train_grpo.py --no-vllm            # if vLLM is unavailable
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

# Unsloth must be imported and patched before anything touches transformers.
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import torch
from trl import GRPOConfig, GRPOTrainer

import grpo_config as C
from data_utils import load_scenarios, to_trl_dataset, apply_template_override
from reward import make_trl_reward_func, reward_ceiling_for


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args):
    """Load the SFT checkpoint and attach a trainable LoRA adapter."""
    print(f"Loading SFT checkpoint: {C.SFT_ADAPTER_PATH}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(C.SFT_ADAPTER_PATH),
        max_seq_length=C.MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=args.use_vllm,
        max_lora_rank=C.LORA_RANK,
        gpu_memory_utilization=C.VLLM_GPU_MEMORY_UTILIZATION,
    )
    tokenizer = apply_template_override(tokenizer, str(C.SFT_ADAPTER_PATH))

    # Continue training the LoRA adapter; the 4-bit base stays frozen and
    # serves as the reference policy. Same pattern as the DPO stage.
    model = FastLanguageModel.get_peft_model(
        model,
        r=C.LORA_RANK,
        lora_alpha=C.LORA_ALPHA,
        target_modules=C.LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--train-file", default=str(C.HARD_EXAMPLES_JSONL))
    ap.add_argument("--dry-run", action="store_true",
                    help="20 steps on 32 scenarios, for pipeline validation")
    ap.add_argument("--no-vllm", dest="use_vllm", action="store_false",
                    default=C.USE_VLLM)
    ap.add_argument("--num-generations", type=int, default=C.NUM_GENERATIONS)
    ap.add_argument("--learning-rate", type=float, default=C.LEARNING_RATE)
    ap.add_argument("--beta", type=float, default=C.BETA)
    args = ap.parse_args()

    set_seed(args.seed)

    run_name = args.run_name or f"grpo-seed{args.seed}"
    out_dir = Path(C.GRPO_OUTPUT_DIR) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"GRPO  |  run={run_name}  seed={args.seed}")
    print("=" * 68)

    # ---- Data ----
    train_path = Path(args.train_file)
    if not train_path.exists():
        raise SystemExit(
            f"{train_path} not found. Run mine_hard_examples.py first, or pass "
            f"--train-file pointing at a ChatML JSONL."
        )

    scenarios = load_scenarios(train_path)
    if args.dry_run:
        scenarios = scenarios[:32]
    print(f"Training scenarios: {len(scenarios)}")

    n_decisive = sum(1 for s in scenarios if s.get("decisive_category"))
    print(f"With decisive_category: {n_decisive} "
          f"({100*n_decisive/max(len(scenarios),1):.1f}%)")

    train_ds = to_trl_dataset(scenarios)

    # ---- Sanity check the reward on reference completions ----
    # A teacher-written reference should score at or near the ceiling. If it
    # does not, the reward or the parser is misconfigured and training will
    # optimize the wrong thing.
    reward_func = make_trl_reward_func()
    refs = [s for s in scenarios if s.get("reference_completion")][:16]
    if refs:
        scores = reward_func(
            [s["reference_completion"] for s in refs],
            gold_verdict=[s["gold_verdict"] for s in refs],
            gold_categories=[s["gold_categories"] for s in refs],
            decisive_category=[s["decisive_category"] for s in refs],
        )
        ceilings = [reward_ceiling_for(s["decisive_category"]) for s in refs]
        ratios = [sc / cl for sc, cl in zip(scores, ceilings) if cl > 0]
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        print(f"\nReward check on {len(refs)} reference completions:")
        print(f"  mean score {sum(scores)/len(scores):.3f}")
        print(f"  mean fraction of per-scenario ceiling: {mean_ratio:.1%}")
        if mean_ratio < 0.6:
            print("\n  WARNING: reference completions score well below ceiling.")
            print("  The parser probably does not match your output format.")
            print("  Inspect one completion and adjust reward.py before training,")
            print("  otherwise GRPO will optimize against a broken signal.")
            worst = min(zip(ratios, refs), key=lambda t: t[0])[1]
            print(f"\n  Lowest-scoring example (uid={worst['uid']}):")
            print(f"  {worst['reference_completion'][:400]}\n")

    # ---- Model ----
    model, tokenizer = build_model(args)

    # ---- Config ----
    batch = C.PER_DEVICE_BATCH_SIZE
    if batch % args.num_generations != 0:
        batch = args.num_generations
        print(f"Adjusted batch size to {batch} (must be divisible by G)")

    grpo_args = GRPOConfig(
        output_dir=str(out_dir),
        run_name=run_name,
        seed=args.seed,

        # Group sampling
        num_generations=args.num_generations,
        temperature=C.TEMPERATURE,          # must be > 0 for within-group spread
        top_p=0.95,
        max_prompt_length=C.MAX_PROMPT_LENGTH,
        max_completion_length=C.MAX_COMPLETION_LENGTH,

        # Optimization. lr is ~100x below SFT; a large lr collapses the policy.
        learning_rate=args.learning_rate,
        beta=args.beta,                      # KL to the SFT reference
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=C.GRAD_ACCUM_STEPS,
        num_train_epochs=1 if args.dry_run else C.NUM_EPOCHS,
        max_steps=20 if args.dry_run else -1,
        warmup_ratio=C.WARMUP_RATIO,
        max_grad_norm=C.MAX_GRAD_NORM,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        # Logging and checkpoints
        logging_steps=C.LOGGING_STEPS,
        save_steps=C.SAVE_STEPS,
        save_total_limit=3,
        report_to="none",

        use_vllm=args.use_vllm,
        log_completions=True,
        num_completions_to_print=2,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=grpo_args,
        train_dataset=train_ds,
    )

    print("\nStarting GRPO training")
    print(f"  G (group size):     {args.num_generations}")
    print(f"  learning rate:      {args.learning_rate}")
    print(f"  beta (KL):          {args.beta}")
    print(f"  temperature:        {C.TEMPERATURE}")
    print(f"  epochs:             {grpo_args.num_train_epochs}")
    print(f"  output:             {out_dir}\n")

    trainer.train()

    final_dir = out_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved adapter to {final_dir}")

    with open(out_dir / "run_config.json", "w") as f:
        json.dump({
            "run_name": run_name,
            "seed": args.seed,
            "sft_checkpoint": str(C.SFT_ADAPTER_PATH),
            "num_generations": args.num_generations,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "temperature": C.TEMPERATURE,
            "n_train_scenarios": len(scenarios),
            "reward_weights": {
                "verdict": C.W_VERDICT,
                "decisive_category": C.W_DECISIVE_CATEGORY,
                "other_categories": C.W_OTHER_CATEGORIES,
                "schema": C.W_SCHEMA,
                "target_bonus": C.W_TARGET_BONUS,
                "length_penalty": C.W_LENGTH_PENALTY,
                "wrong_verdict_category_scale": C.WRONG_VERDICT_CATEGORY_SCALE,
            },
            "preregistered": C.PREREGISTERED,
        }, f, indent=2)

    print("\nNext: python eval_grpo.py --adapter " + str(final_dir))


if __name__ == "__main__":
    main()
