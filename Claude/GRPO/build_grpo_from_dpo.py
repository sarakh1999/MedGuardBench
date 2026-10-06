"""
Build the GRPO training set from the mined hard set PLUS the counterfactual
twins produced for DPO.

Why: GRPO only gets a learning signal where (a) the policy's samples disagree
with each other or (b) the group is unanimously wrong and reference guidance
injects the correct trace. The counterfactual twins are hard by construction
(one factor differs from a profile the policy already knows), so they are
added without going through the sampling-based mining filter.

Sources
  1. mined set         data/<cond>/grpo_train_v2.jsonl (mine_hard_examples.py)
  2. teacher twins     Claude/DPO/data/counterfactual/pairs.jsonl
                       (LLM-edited profiles, blind teacher labels; each pair
                       carries the original and the twin, both labelled)
  3. single-risk twins Claude/new_dataset/Check_Leakage/single_risk_category_train.csv
                       (rule-generated safe twins of single-risk unsafe rows;
                       gold = safe, no teacher trace -> no reference guidance,
                       on-policy signal only; subsampled, see --single-risk-n)

Leakage guard: anything derived from a Patient ID that appears in the model's
test split (grpo_config.TEST_JSONL) is dropped. This matters because the DPO
pair files were built from the blind_v2 split, whose train/test assignment
differs from the legacy_v1 split.

Usage (no GPU):
    MGB_DATA_CONDITION=legacy_v1 python build_grpo_from_dpo.py
    MGB_DATA_CONDITION=legacy_v1 python build_grpo_from_dpo.py --single-risk-n 0
"""
import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

import grpo_config as C
from data_utils import load_scenarios, write_scenarios
from reward import canonical_category, parse_completion

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CONVERTER = REPO / "Claude/SFT/convert_csv_to_chatml_qwen_and_qwenguard.py"
CF_PAIRS = REPO / "Claude/DPO/data/counterfactual/pairs.jsonl"
SINGLE_RISK_CSV = REPO / "Claude/new_dataset/Check_Leakage/single_risk_category_train.csv"
RISK_FILE = REPO / "risk_categories.txt"

_PID_RE = re.compile(r"Patient ID:\s*(\S+)")


def load_converter():
    spec = importlib.util.spec_from_file_location("converter", CONVERTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pid_of_messages(messages):
    for m in messages:
        if m.get("role") == "user":
            mt = _PID_RE.search(m.get("content", ""))
            if mt:
                return mt.group(1)
    return None


def test_pids(test_jsonl):
    pids = set()
    with open(test_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            p = pid_of_messages(rec.get("messages", []))
            if p:
                pids.add(str(p))
    return pids


def make_scenario(uid, conv, row, categories, is_safe, reasoning, decisive, meta):
    """Build a scenario dict (load_scenarios format) from a source-style row."""
    user = conv.build_user_message(row)
    prompt = [{"role": "system", "content": conv.SYSTEM_PROMPT},
              {"role": "user", "content": user}]
    cats = {canonical_category(k): bool(v) for k, v in categories.items()}
    reference = ""
    if reasoning:
        payload = {"reasoning": reasoning,
                   "risk_analysis": {k: bool(categories.get(k, False))
                                     for k in categories},
                   "is_safe": bool(is_safe)}
        reference = json.dumps(payload, indent=2, ensure_ascii=False)
        parsed = parse_completion(reference)
        if parsed["verdict"] != bool(is_safe):
            reference = ""
    s = {
        "uid": uid,
        "messages_prompt": prompt,
        "prompt": prompt,
        "gold_verdict": bool(is_safe),
        "gold_categories": cats,
        "decisive_category": canonical_category(decisive) if decisive else None,
        "reference_completion": reference,
    }
    s.update({f"mining_{k}": v for k, v in meta.items()})
    return s


def teacher_twins(conv, excluded, seen_pids, include_originals=False):
    out, dropped = [], Counter()
    if not CF_PAIRS.exists():
        print(f"  (no teacher twins: {CF_PAIRS} missing)")
        return out, dropped
    with open(CF_PAIRS) as f:
        pairs = [json.loads(l) for l in f if l.strip()]
    for p in pairs:
        if str(p["source_pid"]) in excluded:
            dropped["twin_from_test_pid"] += 1
            continue
        if p.get("split") not in (None, "train", "val"):
            dropped["twin_split"] += 1
            continue
        for side in (("original", "twin") if include_originals else ("twin",)):
            d = p[side]
            if str(d.get("trace_valid", "True")).lower() not in ("true", "1"):
                dropped[f"{side}_trace_invalid"] += 1
                continue
            pid = str(d["row"]["Patient ID"])
            if side == "original" and pid in seen_pids:
                dropped["original_already_present"] += 1
                continue
            if pid in seen_pids:
                dropped["duplicate_twin"] += 1
                continue
            seen_pids.add(pid)
            is_safe = str(d["is_safe"]).lower() in ("true", "1")
            out.append(make_scenario(
                uid=f"cf-{side}-{pid}",
                conv=conv, row=d["row"], categories=d["categories"],
                is_safe=is_safe, reasoning=d.get("reasoning", ""),
                decisive=p["category"],
                meta={"source": f"cf_teacher_{side}", "pair_id": p["pair_id"],
                      "cf_direction": p["direction"],
                      "cf_category": p["category"]}))
    return out, dropped


def single_risk_twins(conv, excluded, seen_pids, n, seed):
    out, dropped = [], Counter()
    if n <= 0 or not SINGLE_RISK_CSV.exists():
        return out, dropped
    df = pd.read_csv(SINGLE_RISK_CSV, dtype=str, keep_default_na=False)
    cats = conv.load_risk_categories(RISK_FILE)
    cands = []
    for _, r in df.iterrows():
        if r.get("Sample_Role") != "counterfactual":
            continue
        if str(r["Source_Patient_ID"]) in excluded:
            dropped["single_from_test_pid"] += 1
            continue
        pid = str(r["Patient ID"])
        if pid in seen_pids:
            dropped["single_duplicate"] += 1
            continue
        cands.append(r)
    rng = random.Random(seed)
    rng.shuffle(cands)
    for r in cands[:n]:
        pid = str(r["Patient ID"])
        seen_pids.add(pid)
        categories = {c: False for c in cats}
        parsed = conv.parse_risk_categories(r.get("Risk_Categories"), cats)
        for k, v in parsed.items():
            categories[k] = bool(v)
        out.append(make_scenario(
            uid=f"cf-single-{pid}", conv=conv, row=r.to_dict(),
            categories=categories, is_safe=conv.parse_is_safe(r.get("Is_Safe")),
            reasoning="",  # templated trace; do not use as guidance reference
            decisive=r.get("Omitted_Risk_Category") or None,
            meta={"source": "cf_single_risk_safe",
                  "source_pid": str(r["Source_Patient_ID"]),
                  "cf_category": r.get("Omitted_Risk_Category", "")}))
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", default=str(C.MINING_CACHE_JSONL.parent / "grpo_train_v2.jsonl"))
    ap.add_argument("--out", default=str(C.MINING_CACHE_JSONL.parent / "grpo_train_guided.jsonl"))
    ap.add_argument("--single-risk-n", type=int, default=400,
                    help="number of rule-generated safe twins to add (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-originals", action="store_true",
                    help="also add the twins' source profiles when the mining "
                         "filter dropped them (they are zero-variance-correct "
                         "for the policy, so they add cost but no gradient)")
    args = ap.parse_args()

    conv = load_converter()
    excluded = test_pids(C.TEST_JSONL)
    print(f"Test split: {len(excluded)} patient IDs excluded ({Path(C.TEST_JSONL).name})")

    mined = load_scenarios(args.mined)
    seen = set()
    for s in mined:
        s.setdefault("mining_source", "mined")
        p = pid_of_messages(s["messages_prompt"])
        if p:
            seen.add(str(p))
    leaked = [s for s in mined if pid_of_messages(s["messages_prompt"]) in excluded]
    print(f"Mined set: {len(mined)} scenarios ({len(leaked)} overlap test split)")

    twins, d1 = teacher_twins(conv, excluded, seen, args.include_originals)
    singles, d2 = single_risk_twins(conv, excluded, seen, args.single_risk_n, args.seed)
    print(f"Teacher twins: {len(twins)} added  dropped={dict(d1)}")
    print(f"Single-risk safe twins: {len(singles)} added  dropped={dict(d2)}")

    all_s = mined + twins + singles
    n = write_scenarios(all_s, args.out)
    back = load_scenarios(args.out)
    assert len(back) == n, "round-trip mismatch"

    src = Counter(s.get("mining_source", "mined") for s in back)
    with_ref = sum(1 for s in back if s.get("reference_completion"))
    print(f"\nWrote {n} scenarios to {args.out}")
    print(f"  by source: {dict(src)}")
    print(f"  with reference trace (guidance-eligible): {with_ref}")
    print(f"  gold unsafe: {sum(1 for s in back if not s['gold_verdict'])}"
          f"  gold safe: {sum(1 for s in back if s['gold_verdict'])}")
    dec = Counter(s.get("decisive_category") or "(none)" for s in twins + singles)
    print("  decisive category of added twins:")
    for k, v in dec.most_common():
        print(f"    {k:<45} {v}")


if __name__ == "__main__":
    main()
