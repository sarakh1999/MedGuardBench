"""
Self-play DPO pairs (round 3) from sample_selfplay_vllm.py output.

For each train prompt with at least one CORRECT self-sample and one WRONG
self-sample:
  chosen   = a correct self-sample (model's own reasoning, correct 17 bits + verdict)
  rejected = a wrong self-sample  (model's own reasoning, >=1 wrong bit or wrong verdict)
Both sides come from the same policy, so style cannot separate them; only the
category bits / verdict can. The rejected answers are the model's *real*
free-generation errors, not synthetic perturbations.

Prompts where the model is never right get chosen=gold (hardest rows; kept
but tagged and capped via --gold-frac so the gold-style shortcut stays minor).

Per-tag cap ("FP:<cat>" / "FN:<cat>" / "verdict_only") keeps the frequent-
category errors dominant without letting one channel swamp the set.

    python Claude/DPO/build_selfplay_pairs.py \
        --samples Claude/DPO/data/selfplay_8b/train_samples.jsonl \
        --out     Claude/DPO/data/selfplay_8b/dpo_pairs.jsonl
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TRAIN = REPO / "Claude/SFT/new_data_chatml_qwen_and_qwenguard/train.jsonl"


def canon(reasoning, ra, is_safe, categories):
    """Rebuild a sample in the exact gold JSON layout (reasoning, 17 bits, is_safe)."""
    return json.dumps({"reasoning": reasoning or "",
                       "risk_analysis": {c: bool(ra[c]) for c in categories},
                       "is_safe": bool(is_safe)}, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rejected-per-row", type=int, default=2,
                    help="distinct wrong patterns per prompt")
    ap.add_argument("--cap", type=int, default=400, help="max pairs per error tag")
    ap.add_argument("--gold-frac", type=float, default=0.25,
                    help="max share of pairs whose chosen is the gold answer (rows never correct)")
    ap.add_argument("--target-cats", default=None,
                    help="comma-separated categories; if set, only pairs whose wrong bits "
                         "intersect this set are kept (verdict_only pairs always kept)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    gold = {}
    for i, line in enumerate(open(args.train)):
        gold[i] = json.loads(line)["messages"]

    target = set(c.strip() for c in args.target_cats.split(",")) if args.target_cats else None
    stats = Counter()
    self_pairs, gold_pairs = [], []

    for line in open(args.samples):
        r = json.loads(line)
        i = r["idx"]
        categories = list(r["gold_risk_analysis"].keys())
        corr = [s for s in r["samples"] if s["correct"]]
        wrong = [s for s in r["samples"] if s["parsed_ok"] and not s["correct"]]
        stats["rows"] += 1
        if not wrong:
            stats["rows_all_correct_or_unparsed"] += 1
            continue
        # dedupe wrong samples by error pattern, keep the most "confident" (shortest) per pattern
        by_pat = {}
        for s in wrong:
            pat = (tuple(s["wrong_bits"]), bool(s["verdict_wrong"]))
            if pat not in by_pat or s["n_tokens"] < by_pat[pat]["n_tokens"]:
                by_pat[pat] = s
        wrong_u = list(by_pat.values())
        rng.shuffle(wrong_u)
        wrong_u = wrong_u[:args.max_rejected_per_row]

        for s in wrong_u:
            tags = [("FP:" if (s["risk_analysis"][c] and not r["gold_risk_analysis"][c]) else "FN:") + c
                    for c in s["wrong_bits"]] or ["verdict_only"]
            if target is not None and tags != ["verdict_only"] and \
                    not any(t.split(":", 1)[1] in target for t in tags):
                stats["off_target_skip"] += 1
                continue
            rejected = canon(s["reasoning"], s["risk_analysis"], s["is_safe"], categories)
            if corr:
                c = rng.choice(corr)
                chosen = canon(c["reasoning"], c["risk_analysis"], c["is_safe"], categories)
                src = "self"
            else:
                chosen = gold[i][-1]["content"]
                src = "gold"
            ex = {"neg_type": f"selfplay_{src}",
                  "changed_category": "|".join(s["wrong_bits"]) or "verdict_only",
                  "prompt": r["prompt"],
                  "chosen": [{"role": "assistant", "content": chosen}],
                  "rejected": [{"role": "assistant", "content": rejected}]}
            (self_pairs if src == "self" else gold_pairs).append((tags, ex))

    # cap gold-chosen share
    rng.shuffle(gold_pairs)
    max_gold = int(args.gold_frac / max(1e-9, 1 - args.gold_frac) * len(self_pairs))
    stats["gold_pairs_available"] = len(gold_pairs)
    gold_pairs = gold_pairs[:max_gold]

    cands = self_pairs + gold_pairs
    rng.shuffle(cands)
    kept, used = [], Counter()
    for tags, ex in cands:
        if any(used[t] < args.cap for t in tags):
            kept.append(ex)
            for t in tags:
                used[t] += 1
        else:
            stats["capped_out"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for ex in kept:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    n_self = sum(ex["neg_type"] == "selfplay_self" for ex in kept)
    print(f"wrote {len(kept)} pairs -> {out}   (self-chosen {n_self}, gold-chosen {len(kept)-n_self})")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<40} {v}")
    print("\nper-tag counts (kept):")
    for t, v in sorted(used.items(), key=lambda kv: -kv[1]):
        print(f"  {t:<48} {v}")


if __name__ == "__main__":
    main()
