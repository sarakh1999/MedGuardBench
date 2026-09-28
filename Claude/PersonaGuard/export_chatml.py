"""
Merge audited records from all domains and write train/val/test ChatML JSONL
in the think format (reasoning in `reasoning_content`, JSON answer in `content`),
ready for train_sft.py / train_grpo.py (and Claude/SFT/qwen_think.py).

Splits
  medication: the split of the original patient (same as the earlier experiments)
  generated domains: by pair_id hash (80/10/10)
  twins always share a split; --heldout_domains moves whole domains to test_heldout.jsonl

Test ablations (same test records, different profile rendering)
  test.jsonl              structured profile (main setting)
  test_blind.jsonl        no profile at all -> how much does personalization matter?
  test_narrative.jsonl    profile as prose instead of fields
  test_shuffled.jsonl     profile fields in random order (trigger position robustness)

Every line: {"messages": [...], "id", "pair_id", "variant", "domain", "gold": <labels JSON string>,
             "personalized", "contrast_group"}

Usage (from repo root):
  python Claude/PersonaGuard/export_chatml.py --inputs Claude/PersonaGuard/data/*.audited.jsonl
  python Claude/PersonaGuard/export_chatml.py --inputs ... --heldout_domains legal_jurisdiction,personal_safety
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from schema import load_jsonl, to_chat, validate_record  # noqa: E402

KEEP_STATUSES = {"ok", "minor", "fixed"}


def split_for(r):
    if r["domain"] == "medication" and r.get("meta", {}).get("split"):
        return r["meta"]["split"]
    x = int(hashlib.md5(r["pair_id"].encode()).hexdigest(), 16) % 100
    return "train" if x < 80 else "val" if x < 90 else "test"


def line(r, mode="structured", order=None):
    return {"messages": to_chat(r, mode, order), "id": r["id"], "pair_id": r["pair_id"],
            "variant": r["variant"], "domain": r["domain"], "gold": json.dumps(r["labels"]),
            "personalized": r.get("personalized", True), "contrast_group": r.get("contrast_group", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="*.audited.jsonl (or *.distilled.jsonl with --no_audit)")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "data", "chatml"))
    ap.add_argument("--heldout_domains", default="")
    ap.add_argument("--no_audit", action="store_true", help="accept distilled records without an audit")
    ap.add_argument("--keep_leaky", action="store_true", help="keep records whose request leaks the trigger")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = [f for p in args.inputs for f in glob.glob(p)]
    recs, dropped = [], Counter()
    for f in files:
        for r in load_jsonl(f):
            if r.get("teacher", {}).get("status") != "ok" or not r.get("reasoning"):
                dropped["no_valid_reasoning"] += 1
                continue
            if not args.no_audit and r.get("audit", {}).get("status") not in KEEP_STATUSES:
                dropped["audit_" + r.get("audit", {}).get("status", "missing")] += 1
                continue
            if not args.keep_leaky and (r.get("meta", {}).get("request_leak") or r.get("audit", {}).get("flag_request_leaks")):
                dropped["request_leaks"] += 1
                continue
            if validate_record(r):
                dropped["schema"] += 1
                continue
            recs.append(r)
    # a twin without its partner is still usable, but report it
    pairs = Counter(r["pair_id"] for r in recs)
    heldout = {d for d in args.heldout_domains.split(",") if d}
    rng = random.Random(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    buckets = {k: [] for k in ("train", "val", "test", "test_heldout")}
    for r in recs:
        buckets["test_heldout" if r["domain"] in heldout else split_for(r)].append(r)
    for name, rs in buckets.items():
        if not rs:
            continue
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(line(r), ensure_ascii=False) + "\n")
        if name.startswith("test"):
            for mode in ("blind", "narrative", "shuffled"):
                with open(os.path.join(args.out_dir, f"{name}_{mode}.jsonl"), "w", encoding="utf-8") as f:
                    for r in rs:
                        order = None
                        if mode == "shuffled":
                            order = list(r["profile"])
                            rng.shuffle(order)
                        f.write(json.dumps(line(r, "structured" if mode == "shuffled" else mode, order),
                                           ensure_ascii=False) + "\n")
    report = {
        "files": files, "kept": len(recs), "dropped": dict(dropped),
        "splits": {k: len(v) for k, v in buckets.items()},
        "by_domain": {k: dict(Counter(split_for(r) if r["domain"] not in heldout else "test_heldout"
                                      for r in recs if r["domain"] == k))
                      for k in sorted({r["domain"] for r in recs})},
        "actions": dict(Counter(r["labels"]["action"] for r in recs)),
        "complete_pairs": sum(1 for c in pairs.values() if c == 2),
        "non_personalized_controls": sum(not r.get("personalized", True) for r in recs),
    }
    with open(os.path.join(args.out_dir, "export_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
