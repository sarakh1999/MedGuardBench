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

Recipe v3 adds --min-safe-frac: hard-example mining keeps a safe scenario only
when the SFT policy already wavers on it, so the mined set ends up ~78 %
gold-unsafe (test is 53 %) and GRPO generalises "flag something" to safe
patients. This tops up gold-safe scenarios from the cache until they are at
least that fraction. FIX (2026-10-15): if the cache does not contain enough
safe scenarios to reach the target, the script now WARNS loudly instead of
silently writing an under-balanced set -- an under-balanced set is exactly
what caused the v2 over-blocking drift.

Usage:
    python mine_hard_examples.py
    python mine_hard_examples.py --limit 200        # quick smoke test
    python mine_hard_examples.py --reuse-cache      # re-filter without regenerating
"""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from tqdm import tqdm

import grpo_config as C
from reward import compute_reward, canonical_category, reward_ceiling_for
from data_utils import load_scenarios, apply_template_override, write_scenarios, model_prompt


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
                    model_prompt(s), tokenize=False, add_generation_prompt=True)
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
    kept, stats, skipped_safe = [], [], []

    with open(cache_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            scenario = by_uid.get(rec["uid"])
            if scenario is None:
                continue

            scored = [
                compute_reward(
                    c,
                    gold_verdict=scenario["gold_verdict"],
                    gold_categories=scenario["gold_categories"],
                    decisive_category=scenario.get("decisive_category"),
                    return_parts=True,
                )
                for c in rec["completions"]
            ]
            if not scored:
                continue

            rewards = [r for r, _ in scored]
            mean_r = statistics.fmean(rewards)
            n_parse_fail = sum(1 for _, p in scored if "parse_fail" in p)

            # The objective is per-category attribution, so "disagreement"
            # is measured on the CATEGORY terms only (decisive + others +
            # target bonus) over the parseable samples. Verdict, schema,
            # length and parse-failure terms are excluded: they would flag
            # groups whose samples all agree on the categories (no category
            # gradient) merely because formatting varied.
            cat_scores = [
                parts.get("decisive", 0.0) + parts.get("others", 0.0)
                + parts.get("target_bonus", 0.0) + parts.get("category_fp", 0.0)
                for _, parts in scored if "parse_fail" not in parts
            ]
            cat_ceiling = reward_ceiling_for(scenario.get("decisive_category")) - C.W_VERDICT - C.W_SCHEMA
            var_cat = statistics.pvariance(cat_scores) if len(cat_scores) > 1 else 0.0
            mean_cat = statistics.fmean(cat_scores) if cat_scores else 0.0

            # Disagreement on categories -> useful GRPO signal.
            has_signal = var_cat > C.MINING_MIN_VARIANCE
            # Consistently wrong on categories: zero variance here, but fresh
            # samples at training time may still differ, so keep as a
            # secondary pool.
            is_hard = (len(cat_scores) >= 2
                       and mean_cat < C.MINING_MAX_MEAN_CATEGORY_FRAC * cat_ceiling)

            stats.append({
                "uid": rec["uid"],
                "mean": mean_r,
                "var": var_cat,
                "n": len(scored),
                "parse_fail": n_parse_fail,
                "keep": has_signal or is_hard,
                "reason": "variance" if has_signal else ("low_mean" if is_hard else "skip"),
            })

            item = dict(scenario)
            item["mining_mean_reward"] = round(mean_r, 4)
            item["mining_reward_variance"] = round(var_cat, 6)
            item["mining_mean_category_score"] = round(mean_cat, 4)
            item["mining_parse_failures"] = n_parse_fail
            if has_signal or is_hard:
                item["mining_reason"] = "variance" if has_signal else "low_mean"
                kept.append(item)
            elif scenario["gold_verdict"] is True:
                # Zero-variance-correct SAFE scenarios: no gradient on their
                # own, kept aside so --min-safe-frac can rebalance the set.
                item["mining_reason"] = "safe_balance"
                skipped_safe.append(item)

    return kept, stats, skipped_safe


def top_up_safe(kept, skipped_safe, min_frac, seed):
    """Append gold-safe scenarios until they are >= min_frac of the set.

    Returns (new_kept, n_added, need, available). n_added < need means the
    mining cache ran out of safe scenarios and the target fraction was NOT
    reached -- the caller must warn, because an under-balanced set is what
    caused the v2 over-blocking drift.

    Candidates are taken from the mining cache, highest category-score
    variance first (closest to the decision boundary), then random; every
    added safe scenario still scores the policy's samples at training time,
    so a sample that over-blocks gets a negative advantage against its group.
    """
    if min_frac <= 0 or not skipped_safe:
        return kept, 0, 0, len(skipped_safe)
    n_safe = sum(1 for s in kept if s["gold_verdict"] is True)
    n_total = len(kept)
    # smallest k with (n_safe + k) / (n_total + k) >= min_frac
    need = max(0, math.ceil((min_frac * n_total - n_safe) / (1.0 - min_frac)))
    if need == 0:
        return kept, 0, 0, len(skipped_safe)
    import random
    pool = sorted(skipped_safe,
                  key=lambda s: (-s["mining_reward_variance"], s["mining_mean_reward"]))
    boundary = [s for s in pool if s["mining_reward_variance"] > 0]
    rest = [s for s in pool if s["mining_reward_variance"] <= 0]
    random.Random(seed).shuffle(rest)
    added = (boundary + rest)[:need]
    return kept + added, len(added), need, len(skipped_safe)


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
    ap.add_argument("--sample", type=int, default=None,
                    help="process a seeded random sample of N scenarios")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=C.MINING_TEMPERATURE,
                    help="sampling temperature (default MINING_TEMPERATURE)")
    ap.add_argument("--reuse-cache", action="store_true",
                    help="skip generation, re-filter the existing cache")
    ap.add_argument("--out", default=str(C.HARD_EXAMPLES_JSONL))
    ap.add_argument("--cache", default=str(C.MINING_CACHE_JSONL))
    ap.add_argument("--min-safe-frac", type=float, default=C.MINING_MIN_SAFE_FRAC,
                    help="top up gold-safe scenarios from the cache until they are at least "
                         "this fraction of the GRPO set (0 = off; recipe v3 default 0.40)")
    ap.add_argument("--allow-unbalanced", action="store_true",
                    help="write the set even if the cache cannot reach --min-safe-frac "
                         "(default: still write, but exit non-zero so a chain can stop)")
    args = ap.parse_args()

    cache_path = Path(args.cache)
    out_path = Path(args.out)

    print("Loading training scenarios...")
    scenarios = load_scenarios(C.TRAIN_JSONL)
    if args.sample:
        import random
        rng = random.Random(args.sample_seed)
        scenarios = rng.sample(scenarios, min(args.sample, len(scenarios)))
        print(f"seeded random sample (seed {args.sample_seed})")
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
        # Sample from the same weights GRPO will start from: the merged SFT
        # model, 4-bit, so mining reflects the actual initial policy.
        init_path = Path(C.POLICY_INIT_PATH)
        if not init_path.exists():
            raise SystemExit(f"POLICY_INIT_PATH not found: {init_path}")
        print(f"\nLoading merged SFT model from {init_path}")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(init_path),
            max_seq_length=C.MAX_SEQ_LENGTH,
            load_in_4bit=C.LOAD_IN_4BIT,
            fast_inference=True,
            gpu_memory_utilization=0.5,
        )
        tokenizer = apply_template_override(tokenizer, str(init_path))
        FastLanguageModel.for_inference(model)

        generate_samples(model, tokenizer, scenarios,
                         C.MINING_N_SAMPLES, args.temperature,
                         C.MINING_MAX_NEW_TOKENS, cache_path)
    else:
        print(f"Reusing cache at {cache_path}")

    print("\nScoring and filtering...")
    kept, stats, skipped_safe = score_and_filter(scenarios, cache_path)

    reasons = Counter(s["reason"] for s in stats)
    print(f"\n  scored:          {len(stats)}")
    print(f"  kept (variance): {reasons['variance']}")
    print(f"  kept (low mean): {reasons['low_mean']}")
    print(f"  dropped:         {reasons['skip']}")
    print(f"  safe held aside: {len(skipped_safe)} (available for --min-safe-frac top-up)")
    if stats:
        means = [s["mean"] for s in stats]
        print(f"  mean reward:     {statistics.fmean(means):.3f}")
        zero_var = sum(1 for s in stats if s["var"] <= C.MINING_MIN_VARIANCE)
        print(f"  zero-variance:   {zero_var} "
              f"({100*zero_var/len(stats):.1f}% would give no gradient)")
        n_pf = sum(s["parse_fail"] for s in stats)
        n_tot = sum(s["n"] for s in stats)
        if n_pf:
            print(f"  parse failures:  {n_pf}/{n_tot} sampled completions "
                  f"({100*n_pf/max(n_tot,1):.1f}%). If this is high, check "
                  f"MINING_MAX_NEW_TOKENS / truncation before training.")

    kept = oversample_targets(kept, C.TARGET_CATEGORY_OVERSAMPLE)
    print(f"\n  after target oversampling: {len(kept)}")

    n_safe0 = sum(1 for k in kept if k["gold_verdict"] is True)
    kept, n_added, need, available = top_up_safe(
        kept, skipped_safe, args.min_safe_frac, args.sample_seed)
    n_safe1 = sum(1 for k in kept if k["gold_verdict"] is True)
    achieved = n_safe1 / max(len(kept), 1)
    print(f"  gold-safe share: {n_safe0}/{len(kept) - n_added} = {n_safe0 / max(len(kept) - n_added, 1):.2f}"
          f" -> {n_safe1}/{len(kept)} = {achieved:.2f}"
          f" (+{n_added} safe scenarios from cache, min-safe-frac {args.min_safe_frac})")

    # FIX: warn loudly if the cache could not supply enough safe scenarios to
    # reach the target. An under-balanced set is the exact condition that made
    # three v2 runs drift into over-blocking.
    unbalanced = args.min_safe_frac > 0 and achieved < args.min_safe_frac - 1e-6
    if unbalanced:
        print("\n" + "!" * 72)
        print(f"  WARNING: could not reach min-safe-frac={args.min_safe_frac:.2f}. "
              f"Needed {need} safe scenarios, only {available} were held aside "
              f"(added {n_added}); achieved {achieved:.2f}.")
        print("  The GRPO set is still unsafe-heavy -- this is the setup that caused the")
        print("  v2 over-blocking drift. Options: lower --min-safe-frac, mine MORE training")
        print("  scenarios so more gold-safe ones land in the cache, or add safe scenarios")
        print("  directly. Proceeding to write the file, but exiting non-zero so a chain stops.")
        print("!" * 72)

    cat_counts = Counter(
        canonical_category(k.get("decisive_category") or "") or "(none)"
        for k in kept
    )
    print("\n  decisive category distribution in GRPO set:")
    for cat, n in cat_counts.most_common():
        flag = "  <-- target" if cat in C.TARGET_CATEGORIES else ""
        print(f"    {str(cat):44s} {n:5d}{flag}")

    # Written in the same ChatML-plus-labels format load_scenarios reads, so
    # train_grpo.py can consume it directly.
    write_scenarios(kept, out_path)
    reloaded = load_scenarios(out_path)
    if len(reloaded) != len(kept):
        raise SystemExit(
            f"Round-trip check failed: wrote {len(kept)} scenarios but "
            f"load_scenarios read back {len(reloaded)}. Fix data_utils before training."
        )
    print(f"\nWrote {len(kept)} scenarios to {out_path} (round-trip verified)")

    if len(kept) < 100:
        print("\n  WARNING: fewer than 100 hard examples. GRPO has little to")
        print("  work with. Consider raising MINING_MAX_MEAN_CATEGORY_FRAC or")
        print("  MINING_TEMPERATURE to surface more disagreement.")

    if unbalanced and not args.allow_unbalanced:
        import sys
        sys.exit(3)


if __name__ == "__main__":
    main()
