#!/usr/bin/env python3
"""
GRPO training for MedGuardBench.

Starts from the MERGED SFT model (never the base model, and never the SFT
adapter directory; see build_model) and refines verdict calibration and
category attribution using a verifiable reward. See reward.py for the reward
design and mine_hard_examples.py for how the training set is selected.

Usage:
    python train_grpo.py
    python train_grpo.py --seed 1 --run-name grpo-seed1
    python train_grpo.py --dry-run            # 20 steps, sanity check only
    python train_grpo.py --no-vllm            # if vLLM is unavailable (slow)
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

# Unsloth must be imported before anything touches transformers/trl.
from unsloth import FastLanguageModel
try:  # Older Unsloth needed an explicit RL patch; newer versions auto-patch.
    from unsloth import PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)
except Exception:
    pass

import torch
import trl
from packaging.version import Version

if Version(trl.__version__) < Version("0.15.0"):
    raise SystemExit(
        f"trl {trl.__version__} is too old: GRPOTrainer needs trl >= 0.15 "
        f"(the unsloth_env has 0.12.2). See setup_grpo_env.sh."
    )
from trl import GRPOConfig, GRPOTrainer

import grpo_config as C
from data_utils import load_scenarios, to_trl_dataset, apply_template_override
from reward import make_trl_reward_func, reward_ceiling_for, parse_completion


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args):
    """Load the merged SFT model and attach a FRESH trainable LoRA adapter.

    Why merged, not the adapter checkpoint:
      TRL's GRPOTrainer gets reference-policy log-probs for the KL term by
      calling model.disable_adapter(). If we loaded the SFT adapter and kept
      training it, "disabled" would be the base Qwen model and beta*KL would
      pull the policy AWAY from SFT. With the merged weights as the base,
      disable_adapter() is exactly the SFT policy, which is what we want to
      stay close to.
    """
    init_path = Path(C.POLICY_INIT_PATH)
    if not init_path.exists():
        raise SystemExit(f"POLICY_INIT_PATH not found: {init_path}")
    if (init_path / "adapter_config.json").exists():
        raise SystemExit(
            f"{init_path} is a LoRA adapter directory, not merged weights. "
            f"GRPO must start from the merged SFT model so the KL reference "
            f"is the SFT policy. Merge with model.save_pretrained_merged(...) "
            f"or point POLICY_INIT_PATH at {C.SFT_MERGED_PATH}."
        )

    print(f"Loading merged SFT model: {init_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(init_path),
        max_seq_length=C.MAX_SEQ_LENGTH,
        load_in_4bit=C.LOAD_IN_4BIT,
        fast_inference=args.use_vllm,
        max_lora_rank=C.LORA_RANK,
        gpu_memory_utilization=args.vllm_gpu_util,
    )
    tokenizer = apply_template_override(tokenizer, str(init_path))

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


def check_prompt_lengths(tokenizer, scenarios, limit):
    """TRL left-truncates prompts longer than max_prompt_length, which would
    silently cut the system prompt. Fail loudly instead."""
    lengths = []
    for s in scenarios:
        ids = tokenizer.apply_chat_template(
            s["messages_prompt"], tokenize=True, add_generation_prompt=True)
        lengths.append(len(ids))
    n_over = sum(1 for n in lengths if n > limit)
    print(f"Prompt tokens: max {max(lengths)}, p95 "
          f"{sorted(lengths)[int(0.95 * (len(lengths) - 1))]}, limit {limit}")
    if n_over:
        raise SystemExit(
            f"{n_over} prompts exceed MAX_PROMPT_LENGTH={limit}; TRL would "
            f"truncate them from the left and drop the system prompt. Raise "
            f"MAX_PROMPT_LENGTH (and MAX_SEQ_LENGTH) in grpo_config.py."
        )


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
    ap.add_argument("--temperature", type=float, default=C.TEMPERATURE,
                    help="sampling temperature for the G training completions")
    ap.add_argument("--epochs", type=float, default=C.NUM_EPOCHS)
    ap.add_argument("--per-device-batch", type=int, default=None,
                    help="micro-batch size; grad accumulation is scaled to keep "
                         "the effective batch from grpo_config (memory knob)")
    ap.add_argument("--vllm-gpu-util", type=float, default=C.VLLM_GPU_MEMORY_UTILIZATION)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop after this many optimizer steps (overrides epochs)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint-* in the run directory")
    ap.add_argument("--force", action="store_true",
                    help="train even if the reference-completion reward check fails")
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
    # A teacher-written reference should score at or near the ceiling under
    # the SAME strict parser used in training. If it does not, the reward or
    # the parser is misconfigured and training will optimize the wrong thing.
    reward_func = make_trl_reward_func(strict_json=C.REQUIRE_JSON_VERDICT_IN_TRAINING)
    refs = [s for s in scenarios if s.get("reference_completion")]
    if refs:
        scores = reward_func(
            [s["reference_completion"] for s in refs],
            gold_verdict=[s["gold_verdict"] for s in refs],
            gold_categories=[s["gold_categories"] for s in refs],
            decisive_category=[s["decisive_category"] or "" for s in refs],
        )
        ceilings = [reward_ceiling_for(s["decisive_category"]) for s in refs]
        ratios = [sc / cl for sc, cl in zip(scores, ceilings) if cl > 0]
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        n_fail = sum(1 for s in refs
                     if not parse_completion(s["reference_completion"],
                                             strict_json=True)["parse_ok"])
        print(f"\nReward check on {len(refs)} reference completions (strict JSON):")
        print(f"  mean score {sum(scores)/len(scores):.3f}")
        print(f"  mean fraction of per-scenario ceiling: {mean_ratio:.1%}")
        print(f"  parse failures: {n_fail}")
        if mean_ratio < 0.9 or n_fail:
            print("\n  WARNING: reference completions do not score at ceiling.")
            print("  The parser probably does not match your output format.")
            print("  Inspect one completion and adjust reward.py before training,")
            print("  otherwise GRPO will optimize against a broken signal.")
            worst = min(zip(ratios, refs), key=lambda t: t[0])[1]
            print(f"\n  Lowest-scoring example (uid={worst['uid']}):")
            print(f"  {worst['reference_completion'][:400]}\n")
            if not args.force:
                raise SystemExit("Aborting. Pass --force to train anyway.")

    # ---- Model ----
    model, tokenizer = build_model(args)
    check_prompt_lengths(tokenizer, scenarios, C.MAX_PROMPT_LENGTH)

    # ---- Config ----
    # Effective batch (completions per optimizer step) is fixed by config; a
    # smaller micro-batch only trades speed for memory.
    effective = C.PER_DEVICE_BATCH_SIZE * C.GRAD_ACCUM_STEPS
    batch = args.per_device_batch or C.PER_DEVICE_BATCH_SIZE
    grad_accum = max(1, effective // batch)
    if (batch * grad_accum) % args.num_generations != 0:
        raise SystemExit(f"generation batch {batch}x{grad_accum} must be divisible "
                         f"by G={args.num_generations}")
    n_prompts_per_step = batch * grad_accum // args.num_generations
    if n_prompts_per_step < 4:
        print(f"WARNING: only {n_prompts_per_step} unique prompts per optimizer "
              f"step; the policy gradient will be very noisy.")

    grpo_args = GRPOConfig(
        output_dir=str(out_dir),
        run_name=run_name,
        seed=args.seed,

        # Group sampling
        num_generations=args.num_generations,
        temperature=args.temperature,       # must be > 0 for within-group spread
        top_p=0.95,
        max_prompt_length=C.MAX_PROMPT_LENGTH,
        max_completion_length=C.MAX_COMPLETION_LENGTH,

        # Optimization. lr is ~100x below SFT; a large lr collapses the policy.
        learning_rate=args.learning_rate,
        beta=args.beta,                      # KL to the SFT reference
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=1 if args.dry_run else args.epochs,
        max_steps=(args.max_steps if args.max_steps else (20 if args.dry_run else -1)),
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
    print(f"  init policy / KL ref: {C.POLICY_INIT_PATH}")
    print(f"  G (group size):     {args.num_generations}")
    print(f"  prompts per step:   {n_prompts_per_step} "
          f"({batch} x {grad_accum} completions)")
    print(f"  learning rate:      {args.learning_rate}")
    print(f"  beta (KL):          {args.beta}")
    print(f"  temperature:        {args.temperature}")
    print(f"  max completion:     {C.MAX_COMPLETION_LENGTH} tokens")
    print(f"  epochs:             {grpo_args.num_train_epochs}")
    print(f"  output:             {out_dir}\n")

    resume = None
    if args.resume:
        ckpts = sorted(out_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        if ckpts:
            resume = str(ckpts[-1])
            print(f"Resuming from {resume}")
        else:
            print("--resume given but no checkpoint found; starting fresh")
    trainer.train(resume_from_checkpoint=resume)

    final_dir = out_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved adapter to {final_dir}")

    with open(out_dir / "run_config.json", "w") as f:
        json.dump({
            "run_name": run_name,
            "seed": args.seed,
            "policy_init_and_kl_reference": str(C.POLICY_INIT_PATH),
            "sft_adapter_checkpoint": str(C.SFT_ADAPTER_PATH),
            "train_file": str(train_path),
            "num_generations": args.num_generations,
            "prompts_per_step": n_prompts_per_step,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "temperature": args.temperature,
            "max_prompt_length": C.MAX_PROMPT_LENGTH,
            "max_completion_length": C.MAX_COMPLETION_LENGTH,
            "epochs": grpo_args.num_train_epochs,
            "max_steps": grpo_args.max_steps,
            "load_in_4bit": C.LOAD_IN_4BIT,
            "data_condition": C.DATA_CONDITION,
            "lora": {"r": C.LORA_RANK, "alpha": C.LORA_ALPHA},
            "n_train_scenarios": len(scenarios),
            "target_categories": C.TARGET_CATEGORIES,
            "reward_weights": {
                "verdict": C.W_VERDICT,
                "decisive_category": C.W_DECISIVE_CATEGORY,
                "other_categories": C.W_OTHER_CATEGORIES,
                "schema": C.W_SCHEMA,
                "target_bonus": C.W_TARGET_BONUS,
                "length_penalty": C.W_LENGTH_PENALTY,
                "length_budget_tokens": C.LENGTH_BUDGET_TOKENS,
                "consistency_penalty": C.W_CONSISTENCY_PENALTY,
                "wrong_verdict_category_scale": C.WRONG_VERDICT_CATEGORY_SCALE,
                "require_json_verdict": C.REQUIRE_JSON_VERDICT_IN_TRAINING,
            },
            "sft_baseline": C.SFT_BASELINE,
            "preregistered": C.PREREGISTERED,
        }, f, indent=2)

    print("\nNext: python eval_grpo.py --adapter " + str(final_dir))


if __name__ == "__main__":
    main()
