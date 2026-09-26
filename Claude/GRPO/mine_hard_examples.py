#!/usr/bin/env python3
"""
Build the GRPO training set by mining hard examples from the SFT model.

Why this step exists:
  GRPO computes advantage as a completion's reward relative to the group mean
  for that prompt. If all G samples score identically, the advantage is zero
  and the prompt contributes no gradient. At 94% SFT accuracy, most training
  prompts produce unanimous groups, so training on the full split would spend
  8x the generation cost to learn from maybe 10-15% of the data.

  This script samples G completions per scenario from the SFT checkpoint,
  keeps only scenarios where the group disagrees or is consistently wrong,
  and oversamples the categories that are currently at floor.

Usage:
    python mine_hard_examples.py
    python mine_hard_examples.py --limit 200        # quick smoke test
    python mine_hard_examples.py --reuse-cache      # re-filter without regenerating
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from tqdm import tqdm

import grpo_config as C
from reward import compute_reward, canonical_category
from data_utils import load_scenarios, apply_template_override


def generate_samples(model, tokenizer, scenarios, n_samples, temperature,
                     max_new_tokens, cache_path):
    """Sample n completions per scenario. Streams to cache for resumability."""
    from vllm import SamplingParams

    done = set()
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["uid"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resuming: {len(done)} scenarios already sampled")

    todo = [s for s in scenarios if s["uid"] not in done]
    if not todo:
        print("All scenarios already sampled")
        return

    sampling = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_new_tokens,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = 32
    with open(cache_path, "a") as out:
        for i in tqdm(range(0, len(todo), batch_size), desc="Sampling"):
            batch = todo[i:i + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    s["messages_prompt"], tokenize=False, add_generation_prompt=True)
                for s in batch
            ]
            outputs = model.fast_generate(prompts, sampling_params=sampling)
            for scenario, output in zip(batch, outputs):
                completions = [o.text for o in output.outputs]
                out.write(json.dumps({
                    "uid": scenario["uid"],
                    "completions": completions,
                }) + "\n")
            out.flush()


def score_and_filter(scenarios, cache_path):
    """Score cached samples and decide which scenarios to keep."""
    by_uid = {s["uid"]: s for s in scenarios}
    kept, stats = [], []

    with open(cache_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            scenario = by_uid.get(rec["uid"])
            if scenario is None:
                continue

            rewards = [
                compute_reward(
                    c,
                    gold_verdict=scenario["gold_verdict"],
                    gold_categories=scenario["gold_categories"],
                    decisive_category=scenario.get("decisive_category"),
                )
                for c in rec["completions"]
            ]
            if not rewards:
                continue

            mean_r = statistics.fmean(rewards)
            var_r = statistics.pvariance(rewards) if len(rewards) > 1 else 0.0

            has_signal = var_r > C.MINING_MIN_VARIANCE
            is_hard = mean_r < C.MINING_MAX_MEAN_REWARD

            stats.append({
                "uid": rec["uid"],
                "mean": mean_r,
                "var": var_r,
                "keep": has_signal or is_hard,
                "reason": "variance" if has_signal else ("low_mean" if is_hard else "skip"),
            })

            if has_signal or is_hard:
                item = dict(scenario)
                item["mining_mean_reward"] = round(mean_r, 4)
                item["mining_reward_variance"] = round(var_r, 6)
                item["mining_reason"] = "variance" if has_signal else "low_mean"
                kept.append(item)

    return kept, stats


def oversample_targets(kept, factor):
    """Duplicate scenarios whose decisive category is a target category."""
    if factor <= 1.0:
        return kept
    extra = []
    targets = {canonical_category(c) for c in C.TARGET_CATEGORIES}
    n_extra = int(round(factor)) - 1
    for item in kept:
        canon = canonical_category(item.get("decisive_category") or "")
        if canon in targets:
            for k in range(n_extra):
                dup = dict(item)
                dup["uid"] = f"{item['uid']}#dup{k+1}"
                dup["oversampled"] = True
                extra.append(dup)
    return kept + extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N scenarios")
    ap.add_argument("--reuse-cache", action="store_true",
                    help="skip generation, re-filter the existing cache")
    ap.add_argument("--out", default=str(C.HARD_EXAMPLES_JSONL))
    ap.add_argument("--cache", default=str(C.MINING_CACHE_JSONL))
    args = ap.parse_args()

    cache_path = Path(args.cache)
    out_path = Path(args.out)

    print("Loading training scenarios...")
    scenarios = load_scenarios(C.TRAIN_JSONL)
    if args.limit:
        scenarios = scenarios[:args.limit]
    print(f"{len(scenarios)} scenarios")

    n_decisive = sum(1 for s in scenarios if s.get("decisive_category"))
    print(f"{n_decisive} have a decisive_category label "
          f"({100*n_decisive/max(len(scenarios),1):.1f}%)")
    if n_decisive < 0.5 * len(scenarios):
        print("\n  NOTE: fewer than half your scenarios carry a decisive_category.")
        print("  The reward falls back to full-set F1 for those, which is a")
        print("  weaker signal. If your generation pipeline recorded the target")
        print("  category per scenario, add it to the JSONL as 'decisive_category'")
        print("  or 'target_category' to get the stronger reward.\n")

    if not args.reuse_cache:
        print(f"\nLoading SFT checkpoint from {C.SFT_ADAPTER_PATH}")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(C.SFT_ADAPTER_PATH),
            max_seq_length=C.MAX_SEQ_LENGTH,
            load_in_4bit=True,
            fast_inference=True,
            gpu_memory_utilization=0.6,
        )
        tokenizer = apply_template_override(tokenizer, str(C.SFT_ADAPTER_PATH))
        FastLanguageModel.for_inference(model)

        generate_samples(model, tokenizer, scenarios,
                         C.MINING_N_SAMPLES, C.MINING_TEMPERATURE,
                         C.MINING_MAX_NEW_TOKENS, cache_path)
    else:
        print(f"Reusing cache at {cache_path}")

    print("\nScoring and filtering...")
    kept, stats = score_and_filter(scenarios, cache_path)

    reasons = Counter(s["reason"] for s in stats)
    print(f"\n  scored:          {len(stats)}")
    print(f"  kept (variance): {reasons['variance']}")
    print(f"  kept (low mean): {reasons['low_mean']}")
    print(f"  dropped:         {reasons['skip']}")
    if stats:
        means = [s["mean"] for s in stats]
        print(f"  mean reward:     {statistics.fmean(means):.3f}")
        zero_var = sum(1 for s in stats if s["var"] <= C.MINING_MIN_VARIANCE)
        print(f"  zero-variance:   {zero_var} "
              f"({100*zero_var/len(stats):.1f}% would give no gradient)")

    kept = oversample_targets(kept, C.TARGET_CATEGORY_OVERSAMPLE)
    print(f"\n  after target oversampling: {len(kept)}")

    cat_counts = Counter(
        canonical_category(k.get("decisive_category") or "") or "(none)"
        for k in kept
    )
    print("\n  decisive category distribution in GRPO set:")
    for cat, n in cat_counts.most_common():
        flag = "  <-- target" if cat in C.TARGET_CATEGORIES else ""
        print(f"    {str(cat):44s} {n:5d}{flag}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for item in kept:
            f.write(json.dumps(item) + "\n")
    print(f"\nWrote {len(kept)} scenarios to {out_path}")

    if len(kept) < 100:
        print("\n  WARNING: fewer than 100 hard examples. GRPO has little to")
        print("  work with. Consider raising MINING_MAX_MEAN_REWARD or")
        print("  MINING_TEMPERATURE to surface more disagreement.")


if __name__ == "__main__":
    main()
