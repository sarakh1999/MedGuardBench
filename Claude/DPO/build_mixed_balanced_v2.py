"""
Balanced v2 DPO training mix (hard precision negatives).

Combines, per category c:
  n_c = min(#drop_true_c, #add_false_hard_c, CAP) pairs of EACH direction
        (symmetric recall/precision pressure; evidence-adjacent precision
        negatives instead of the random ones that let the mixed_balanced 8B
        over-flag Cardiac/Substance)
plus all swap_hard pairs (attribution) and the 1618 single-risk profile-level
counterfactual pairs (causal grounding + verdict sensitivity).

    python Claude/DPO/build_mixed_balanced_v2.py
    -> Claude/DPO/data/mixed_balanced_v2/dpo_pairs.jsonl
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARD = HERE / "data/multicat/dpo_pairs_v2_hard.jsonl"
SINGLE_CF = HERE / "data/counterfactual_single_risk/dpo_pairs.jsonl"
OUT = HERE / "data/mixed_balanced_v2/dpo_pairs.jsonl"
CAP = 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", default=str(HARD))
    ap.add_argument("--single-cf", default=str(SINGLE_CF))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    by = defaultdict(list)
    swaps = []
    for line in open(args.hard):
        e = json.loads(line)
        if e["neg_type"] in ("swap", "swap_hard"):
            swaps.append(e)
        else:
            by[(e["neg_type"], e["changed_category"])].append(e)

    cats = sorted({c for (_, c) in by})
    picked = []
    print(f"{'category':40s} {'n/dir':>6s}")
    for c in cats:
        dt = by.get(("drop_true", c), [])
        af = by.get(("add_false_hard", c), [])
        n = min(len(dt), len(af), args.cap)
        picked += rng.sample(dt, n) + rng.sample(af, n)
        print(f"{c:40s} {n:6d}")

    single = [json.loads(line) for line in open(args.single_cf)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_ex = picked + swaps + single
    rng.shuffle(all_ex)
    with out.open("w", encoding="utf-8") as fh:
        for e in all_ex:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\nbalanced hard pairs {len(picked)} + swap_hard {len(swaps)} "
          f"+ single-risk CF {len(single)} = {len(all_ex)} -> {out}")


if __name__ == "__main__":
    main()
