"""
GRPO on top of the SFT adapter, with the verifiable reward in reward.py.

Why these choices (same lessons as Claude/GRPO):
  * start from SFT, lr ~1e-6, KL beta 0.04, temperature 1.0, G=8
  * the reward's grounding term (triggering-attribute F1) gives a non-zero
    advantage even when the whole group agrees on the action
  * --hard_only keeps prompts where the SFT model is not already perfect
    (scored by sampling; see --mine), so the batch is not all zero-advantage
  * TWINS ARE KEPT TOGETHER in the training set, so the policy is pushed on
    both the harmful and the benign version of each request (the benign twin
    penalizes blanket refusal)

The prompt is pre-filled with "<think>\\n" (qwen_think.generation_prompt), so the
policy must use its scratchpad; empty thinking is penalized by the reward.

Usage (from repo root):
  python Claude/PersonaGuard/train_grpo.py --sft_adapter Claude/PersonaGuard/outputs/qwen3-4b/final --run grpo-4b
  python Claude/PersonaGuard/train_grpo.py ... --dry_run
"""

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "SFT"))

from unsloth import FastLanguageModel, PatchFastRL  # noqa: E402
PatchFastRL("GRPO", FastLanguageModel)
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from qwen_think import generation_prompt  # noqa: E402
from reward import compute_reward, make_trl_reward_func, reward_ceiling  # noqa: E402


def load_prompts(path, tok, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            ex = json.loads(l)
            prompt_msgs = [m for m in ex["messages"] if m["role"] != "assistant"]
            prof = {}
            for line in prompt_msgs[0]["content"].split("[USER PROFILE]\n", 1)[-1].splitlines():
                if line.startswith("- ") and ": " in line:
                    k, v = line[2:].split(": ", 1)
                    prof[k] = v
            asst = next(m for m in ex["messages"] if m["role"] == "assistant")
            rows.append({"prompt": generation_prompt(tok, prompt_msgs), "gold": ex["gold"],
                         "profile": json.dumps(prof), "id": ex["id"], "pair_id": ex["pair_id"],
                         "reference": asst.get("reasoning_content", "") + "\n</think>\n\n" + asst["content"]})
    return rows[:limit] if limit else rows


def mine_hard(model, tok, rows, n=4, max_new_tokens=900):
    """Keep pairs where at least one twin is not solved perfectly by the SFT policy."""
    FastLanguageModel.for_inference(model)
    keep_pairs = set()
    for r in rows:
        ids = tok(r["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        outs = model.generate(input_ids=ids.repeat(n, 1), do_sample=True, temperature=1.0, top_p=0.95,
                              max_new_tokens=max_new_tokens)
        gold, prof = json.loads(r["gold"]), json.loads(r["profile"])
        scores = [compute_reward(tok.decode(o[ids.shape[1]:], skip_special_tokens=False), gold, list(prof), prof)
                  for o in outs]
        if min(scores) < reward_ceiling(gold) - 1e-6 or max(scores) - min(scores) > 1e-6:
            keep_pairs.add(r["pair_id"])
    FastLanguageModel.for_training(model)
    return [r for r in rows if r["pair_id"] in keep_pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_adapter", required=True)
    ap.add_argument("--run", default="grpo")
    ap.add_argument("--train_file", default=os.path.join(HERE, "data", "chatml", "train.jsonl"))
    ap.add_argument("--out_root", default=os.path.join(HERE, "outputs"))
    ap.add_argument("--hard_only", action="store_true", help="mine prompts the SFT policy has not mastered")
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--max_completion_length", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_vllm", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.sft_adapter, max_seq_length=args.max_seq_length, load_in_4bit=True,
        fast_inference=not args.no_vllm, max_lora_rank=16, gpu_memory_utilization=0.35)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed)

    rows = load_prompts(args.train_file, tok, 32 if args.dry_run else args.max_prompts)
    if args.hard_only and not args.dry_run:
        before = len(rows)
        rows = mine_hard(model, tok, rows)
        print(f"hard-example mining kept {len(rows)}/{before} prompts (twins kept together)")

    reward_func = make_trl_reward_func()
    refs = rows[:16]
    scores = reward_func([r["reference"] for r in refs], gold=[r["gold"] for r in refs],
                         profile=[r["profile"] for r in refs])
    ratio = sum(s / reward_ceiling(json.loads(r["gold"])) for s, r in zip(scores, refs)) / max(len(refs), 1)
    print(f"reference completions reach {ratio:.1%} of the reward ceiling (expect ~100%)")
    if ratio < 0.8:
        raise SystemExit("Reward/parser mismatch with the data format; fix before training.")

    ds = Dataset.from_list([{k: r[k] for k in ("prompt", "gold", "profile")} for r in rows])
    bf16 = torch.cuda.is_bf16_supported()
    cfg = GRPOConfig(
        output_dir=os.path.join(args.out_root, args.run), seed=args.seed,
        num_generations=args.num_generations, temperature=1.0, top_p=0.95,
        max_prompt_length=args.max_seq_length - args.max_completion_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr, beta=args.beta, per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=4, num_train_epochs=args.epochs, max_steps=20 if args.dry_run else -1,
        warmup_ratio=0.1, max_grad_norm=0.2, lr_scheduler_type="cosine", optim="adamw_8bit",
        bf16=bf16, fp16=not bf16, logging_steps=5, save_steps=50, save_total_limit=3, report_to="none",
        use_vllm=not args.no_vllm, log_completions=True)
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[reward_func], args=cfg, train_dataset=ds)
    trainer.train()
    model.save_pretrained(os.path.join(args.out_root, args.run, "final"))
    tok.save_pretrained(os.path.join(args.out_root, args.run, "final"))


if __name__ == "__main__":
    main()
