#!/usr/bin/env python3
"""Score old vs re-run (4k budget) zero-shot base files under the abstention
convention on the 729 drug-disjoint split, and compare per-category F1."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R

cats = R.cm.load_categories(R.CATS_PATH) if hasattr(R.cm, "load_categories") else \
       [l.strip() for l in open(R.CATS_PATH) if l.strip() and not l.startswith("#")]
gold_safe, gold_cat = R.load_gold(cats)
raw, keep = R.keep_mask()
keep = np.asarray(keep, dtype=bool)

JSON = json.load(open(os.path.join(R.OUT_DIR, "paper", "main_729_abstain.json")))

def score_file(path):
    ps, pc, parsed, kind, n = R.load_pred_file(path, cats)
    # align length to gold
    N = len(gold_safe)
    if n < N:
        pad = N - n
        ps = np.concatenate([ps, np.ones(pad, bool)])
        pc = np.concatenate([pc, np.zeros((pad, len(cats)), bool)])
        parsed = np.concatenate([parsed, np.zeros(pad, bool)])
    idx = np.where(keep & parsed)[0]
    n_unpars = int(np.sum(keep & ~parsed))
    s = R.score(gold_safe, gold_cat, ps, pc, idx)
    return s, n_unpars, len(idx)

jobs = {
 "q8":  ("qwen3-8b",),
 "q14": ("qwen3-14b",),
}
SB = "/fs/scratch/PCS0289/sarakhosravi/rerun4k"
for k,(key,) in jobs.items():
    old = f"Claude/Ablation/outputs/{key}/base/true/test_predictions.jsonl"
    new = f"{SB}/{key}/base/true/test_predictions.jsonl"
    so, uo, no = score_file(old)
    sn, un, nn = score_file(new)
    j = JSON[k][0]  # base arm from authoritative json
    print(f"\n===== {k} ({key}) zero-shot base, abstention/729 =====")
    print(f"{'metric':<10}{'OLD(file)':>12}{'JSON':>10}{'NEW(4k)':>12}")
    def row(name, ov, jv, nv):
        print(f"{name:<10}{ov:>12}{jv:>10}{nv:>12}")
    row("n_unpars", uo, j.get("n_unpars"), un)
    row("n_scored", no, j.get("n_scored"), nn)
    row("acc", f"{so['acc']*100:.1f}", f"{j['acc']:.1f}", f"{sn['acc']*100:.1f}")
    row("recall", f"{so['recall']*100:.1f}", f"{j['rec']:.1f}", f"{sn['recall']*100:.1f}")
    row("fpr", f"{so['fpr']*100:.1f}", f"{j['fpr']:.1f}", f"{sn['fpr']*100:.1f}")
    row("micro_f1", f"{so['micro_f1']*100:.1f}", f"{j['mic']:.1f}", f"{sn['micro_f1']*100:.1f}")
    row("macro_f1", f"{so['macro_f1']*100:.1f}", f"{j['mac']:.1f}", f"{sn['macro_f1']*100:.1f}")
    row("exact", f"{so['exact_match']*100:.1f}", f"{j.get('em')}", f"{sn['exact_match']*100:.1f}")
    # per-category new values
    print("  per-category F1 (NEW):")
    for c, f in zip(cats, sn['per_cat_f1']):
        print(f"    {c:<42s} {f*100:5.1f}")
