"""
Build preference pairs for DPO from the GRPO mining cache.

Pairs are label-anchored and verifiable, no judge model involved:

  chosen    the highest-reward sampled completion whose verdict is correct and
            whose reward reaches CHOSEN_MIN_FRACTION of the ceiling for that
            scenario; if no sample qualifies, the reference (teacher) trace
  rejected  the lowest-reward sampled completion with the WRONG verdict; if
            every sample has the right verdict, the lowest-reward sample,
            kept only when the chosen-rejected gap is >= MIN_REWARD_GAP

Scenarios where the policy is already always right by a wide margin produce
no pair: DPO on them only sharpens what SFT already does. Rewards come from
Claude/GRPO/reward.py, so "better" means the same thing here as in GRPO.

Requires the mining cache written by Claude/GRPO/mine_hard_examples.py
(uid -> N sampled completions) for the same data condition.

Usage:
    python Claude/DPO/build_dpo_pairs.py                   # uses grpo_config paths
    python Claude/DPO/build_dpo_pairs.py --cache X --train Y --out Z
"""

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "GRPO"))
import grpo_config as C                              # noqa: E402
from data_utils import load_scenarios               # noqa: E402
from reward import compute_reward, parse_completion, reward_ceiling_for  # noqa: E402

CHOSEN_MIN_FRACTION = 0.8   # of the scenario's reward ceiling
MIN_REWARD_GAP = 0.5
DPO_DIR = C.PROJECT_ROOT / "Claude" / "DPO" / "data" / C.DATA_CONDITION


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(C.MINING_CACHE_JSONL))
    ap.add_argument("--train", default=str(C.TRAIN_JSONL))
    ap.add_argument("--out", default=str(DPO_DIR / "dpo_pairs.jsonl"))
    ap.add_argument("--chosen-min-fraction", type=float, default=CHOSEN_MIN_FRACTION)
    ap.add_argument("--min-gap", type=float, default=MIN_REWARD_GAP)
    ap.add_argument("--allow-reference-chosen", action="store_true", default=True)
    ap.add_argument("--no-reference-chosen", dest="allow_reference_chosen", action="store_false")
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        sys.exit(f"mining cache not found: {args.cache}\n"
                 "run Claude/GRPO/mine_hard_examples.py first")

    scenarios = {s["uid"]: s for s in load_scenarios(args.train)}
    samples = {}
    with open(args.cache) as f:
        for line in f:
            r = json.loads(line)
            samples.setdefault(r["uid"], []).extend(r["completions"])
    print(f"{len(scenarios)} scenarios, {len(samples)} with samples")

    stats = Counter()
    pairs = []
    for uid, comps in samples.items():
        s = scenarios.get(uid)
        if s is None:
            stats["uid_not_in_train"] += 1
            continue
        gold_v, gold_c, dec = s["gold_verdict"], s["gold_categories"], s.get("decisive_category")
        ceiling = reward_ceiling_for(dec)
        scored = []
        for c in comps:
            r = compute_reward(c, gold_v, gold_c, dec)
            p = parse_completion(c, strict_json=True)
            scored.append((r, c, p.get("verdict") if isinstance(p, dict) else None))
        scored.sort(key=lambda t: t[0])

        correct = [t for t in scored if t[2] is not None and t[2] == gold_v]
        wrong = [t for t in scored if t[2] is None or t[2] != gold_v]

        chosen_src = None
        if correct and correct[-1][0] >= args.chosen_min_fraction * ceiling:
            chosen, chosen_src = correct[-1][1], "sample"
        elif args.allow_reference_chosen and s.get("reference_completion"):
            chosen, chosen_src = s["reference_completion"], "reference"
        else:
            stats["no_chosen"] += 1
            continue
        chosen_r = compute_reward(chosen, gold_v, gold_c, dec)

        if wrong:
            rejected, rejected_src = wrong[0][1], "wrong_verdict_sample"
        else:
            rejected, rejected_src = scored[0][1], "low_reward_sample"
        rejected_r = compute_reward(rejected, gold_v, gold_c, dec)

        if chosen_r - rejected_r < args.min_gap:
            stats["gap_too_small"] += 1
            continue
        if chosen == rejected:
            stats["identical"] += 1
            continue

        pairs.append({
            "uid": uid,
            "prompt": s["messages_prompt"],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "gold_verdict": gold_v,
            "decisive_category": dec,
            "chosen_source": chosen_src,
            "rejected_source": rejected_src,
            "chosen_reward": chosen_r,
            "rejected_reward": rejected_r,
        })
        stats[f"chosen={chosen_src}"] += 1
        stats[f"rejected={rejected_src}"] += 1
        stats["gold_safe" if gold_v else "gold_unsafe"] += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"\nwrote {len(pairs)} pairs -> {args.out}")
    for k, v in sorted(stats.items()):
        print(f"  {k:<28} {v}")
    if pairs:
        gaps = [p["chosen_reward"] - p["rejected_reward"] for p in pairs]
        print(f"  reward gap: mean {sum(gaps)/len(gaps):.2f}  min {min(gaps):.2f}  max {max(gaps):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
