"""
On-policy DPO pairs from the model's own mistakes (iterative DPO round 2).

Input: predictions of the current DPO model on the TRAIN split
(Claude/SFT/blind/eval_sft.py output) + the gold train.jsonl.

For every train row where the model got the verdict or any category bit wrong,
emit one pair:
  chosen   = gold assistant answer (exactly as in train.jsonl)
  rejected = the model's own answer, rebuilt in the identical JSON format
             (its reasoning, its 17 bits, its verdict)

These are guaranteed-hard negatives: the model assigns them high probability
by construction, and they carry its real failure modes (e.g. evidence-adjacent
Cardiac over-flagging) including the wrong reasoning text.

    python Claude/DPO/build_mined_pairs.py \
        --pred Claude/DPO/data/mined_8b/train_predictions.jsonl \
        --out  Claude/DPO/data/mined_8b/dpo_pairs.jsonl
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TRAIN = REPO / "Claude/SFT/new_data_chatml_qwen_and_qwenguard/train.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(HERE / "data/mined_8b/train_predictions.jsonl"))
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--out", default=str(HERE / "data/mined_8b/dpo_pairs.jsonl"))
    ap.add_argument("--cap", type=int, default=150,
                    help="max pairs per (direction, category) error tag, e.g. "
                         "'FP:Bleeding Risk'. The model's train errors are "
                         "concentrated (Bleeding/DDI FPs); uncapped they would "
                         "swamp the balanced answer-level set and collapse "
                         "those categories' recall the way round 1 collapsed "
                         "Cardiac precision.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    gold = {}
    for i, line in enumerate(open(args.train)):
        d = json.loads(line)
        gold[i] = d["messages"]

    stats = Counter()
    candidates = []  # (tags, example)
    for line in open(args.pred):
        p = json.loads(line)
        i = p["idx"]
        if i not in gold:
            stats["no_gold"] += 1
            continue
        msgs = gold[i]
        gold_ans = json.loads(msgs[-1]["content"])
        g_ra = gold_ans["risk_analysis"]
        g_safe = gold_ans["is_safe"]

        if not p.get("parsed_ok") or not p.get("ra_parsed_ok"):
            stats["unparsed_skip"] += 1
            continue
        m_ra = {c: bool(v) for c, v in (p["pred_risk_analysis"] or {}).items()}
        m_safe = bool(p["pred_is_safe"])
        if set(m_ra) != set(g_ra):
            stats["bad_schema_skip"] += 1
            continue
        wrong_bits = [c for c in g_ra if bool(g_ra[c]) != m_ra[c]]
        if not wrong_bits and m_safe == g_safe:
            stats["correct"] += 1
            continue

        rejected = json.dumps({
            "reasoning": p.get("pred_reasoning") or "",
            "risk_analysis": {c: m_ra[c] for c in g_ra},
            "is_safe": m_safe,
        }, indent=2, ensure_ascii=False)
        tags = [("FP:" if (m_ra[c] and not g_ra[c]) else "FN:") + c
                for c in wrong_bits] or ["verdict_only"]
        candidates.append((tags, {
            "neg_type": "mined_on_policy",
            "changed_category": "|".join(wrong_bits) if wrong_bits else "verdict_only",
            "prompt": msgs[:-1],
            "chosen": [{"role": "assistant", "content": msgs[-1]["content"]}],
            "rejected": [{"role": "assistant", "content": rejected}],
        }))

    # Per-tag cap: keep a pair only if at least one of its error tags is still
    # under the cap (multi-bit errors count against every tag they carry).
    rng.shuffle(candidates)
    kept, used = [], Counter()
    for tags, ex in candidates:
        if any(used[t] < args.cap for t in tags):
            kept.append(ex)
            for t in tags:
                used[t] += 1
                stats[t] += 1
        else:
            stats["capped_out"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for ex in kept:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"wrote {len(kept)} mined pairs (of {len(candidates)} errors, "
          f"cap {args.cap}/tag) -> {out}")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<52} {v}")


if __name__ == "__main__":
    main()
