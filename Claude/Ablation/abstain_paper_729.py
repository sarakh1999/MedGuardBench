#!/usr/bin/env python3
"""Authoritative abstention recompute for every paper cell on the 729 split.

Convention change: a completion with no parseable SAFE/UNSAFE verdict is counted
as ``unparsable'' and dropped from every metric denominator (it is NOT scored as
Safe). Each cell additionally carries ``n_unpars`` and ``unpars`` (the % of the
729 rows that were unparsable), which the tables report in a dedicated column.

We reuse the file->cell mappings already recorded by emit_paper_729.py in
report/recompute_729/paper/{main_729_used,np_729,profile_729}.json, and rescore
every referenced file with one scorer (recompute_729.load_pred_file + score),
excluding unparsable rows. Outputs *_abstain.json next to the originals.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute_729 as R

PAPER = os.path.join(R.OUT_DIR, "paper")
CATS = R.cm.load_risk_categories(R.CATS_PATH)
_RAW, KEEP = R.keep_mask()
GOLD_SAFE, GOLD_CAT = R.load_gold(CATS)
KEPT = np.flatnonzero(KEEP)
PRESC_COLS = np.array([c in R.PRESCRIPTION_CATS for c in CATS])
_UNSAFE = ~GOLD_SAFE
_PRESC = _UNSAFE & GOLD_CAT[:, PRESC_COLS].any(axis=1)
_PROFILE_ONLY = _UNSAFE & ~_PRESC


def P(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(100 * float(x) + 1e-9, 1)


def resolve_jsonl(rel):
    """Map a cell 'file' (metrics json or jsonl) to its prediction jsonl."""
    if rel.endswith(".jsonl"):
        return os.path.join(R.ROOT, rel)
    cand = R.jsonl_for_metrics(rel)
    if cand is None:
        raise SystemExit(f"no jsonl for {rel}")
    return cand


def load_any(path):
    """Like R.load_pred_file but tolerates files with line-order (no 'idx') rows."""
    with open(path) as f:
        has_idx = "idx" in (f.readline() or "")
    if has_idx:
        return R.load_pred_file(path, CATS)
    from reward import parse_completion
    rows = {}
    kind = None
    with open(path) as f:
        for nline, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            i = int(r["idx"]) if "idx" in r else nline
            rows[i] = r
            if kind is None:
                kind = "completion" if ("completion" in r and "pred_is_safe" not in r) else "schema"
    n = max(rows) + 1
    pred_safe = np.ones(n, dtype=bool)
    pred_cat = np.zeros((n, len(CATS)), dtype=bool)
    parsed = np.zeros(n, dtype=bool)
    for i, r in rows.items():
        if kind == "schema":
            ok = bool(r.get("parsed_ok", True))
            safe = bool(r["pred_is_safe"])
            if not ok:
                lg = R.cm.parse_llamaguard(r.get("raw_response", ""), CATS)
                if lg:
                    safe, ok = lg[1], True
            parsed[i] = ok
            pred_safe[i] = safe
            ra, _ = R.cm.pred_ra_from_record(r, CATS)
            pred_cat[i] = [bool(ra[c]) for c in CATS]
        else:
            pc = parse_completion(r.get("completion", ""))
            parsed[i] = bool(pc["parse_ok"])
            v = pc["verdict"]
            pred_safe[i] = True if v is None else bool(v)
            cc = pc["categories"] or {}
            pred_cat[i] = [bool(cc.get(c, False)) for c in CATS]
    return pred_safe, pred_cat, parsed, kind, n


def cell_abstain(rel):
    path = resolve_jsonl(rel)
    pred_safe, pred_cat, parsed, kind, n = load_any(path)
    if n < len(KEEP):
        raise SystemExit(f"short file {rel} n={n}")
    parsed_keep = parsed[KEPT]
    idx = KEPT[parsed_keep]                       # kept AND parsed
    n_unpars = int((~parsed_keep).sum())
    if len(idx) == 0:
        out = {k: None for k in ("acc", "rec", "fpr", "mic", "mac", "em", "prec",
                                 "g_profile", "g_presc", "g_safe")}
        out.update({"n_unpars": n_unpars, "unpars": P(n_unpars / len(KEPT)),
                    "n_scored": 0, "file": rel})
        for c in CATS:
            out[c] = None
        return out
    s = R.score(GOLD_SAFE, GOLD_CAT, pred_safe, pred_cat, idx)

    def grp(mask_full):
        m = mask_full[idx]
        if not m.any():
            return None
        return float(np.mean(pred_safe[idx][m] == GOLD_SAFE[idx][m]))

    out = {
        "acc": P(s["acc"]), "rec": P(s["recall"]), "fpr": P(s["fpr"]),
        "mic": P(s["micro_f1"]), "mac": P(s["macro_f1"]), "em": P(s["exact_match"]),
        "prec": P(s["precision"]),
        "g_profile": P(grp(_PROFILE_ONLY)), "g_presc": P(grp(_PRESC)), "g_safe": P(grp(GOLD_SAFE)),
        "n_unpars": n_unpars, "unpars": P(n_unpars / len(KEPT)), "n_scored": int(len(idx)),
        "file": rel,
    }
    for j, c in enumerate(CATS):
        out[c] = P(s["per_cat_f1"][j])
    return out


def redo_main():
    used = json.load(open(os.path.join(PAPER, "main_729_used.json")))
    new = {}
    for model, arms in used.items():
        new[model] = []
        for a in arms:
            if not a or "file" not in a:
                new[model].append(None)
                continue
            new[model].append(cell_abstain(a["file"]))
    json.dump(new, open(os.path.join(PAPER, "main_729_abstain.json"), "w"), indent=1)
    return new


def redo_np():
    d = json.load(open(os.path.join(PAPER, "np_729.json")))
    new_rows = {}
    for key, r in d["rows"].items():
        new_rows[key] = cell_abstain(r["file"])
    json.dump({"rows": new_rows, "boot": d.get("boot"), "drops": d.get("drops")},
              open(os.path.join(PAPER, "np_729_abstain.json"), "w"), indent=1)
    return new_rows


def redo_profile():
    d = json.load(open(os.path.join(PAPER, "profile_729.json")))
    new = {}
    for key, r in d.items():
        new[key] = cell_abstain(r["file"])
    json.dump(new, open(os.path.join(PAPER, "profile_729_abstain.json"), "w"), indent=1)
    return new


def main():
    m = redo_main()
    npr = redo_np()
    pr = redo_profile()
    print("MAIN cells:", sum(len(v) for v in m.values()))
    print("NP cells:", len(npr), " PROFILE cells:", len(pr))
    # quick report of the biggest unparsable rates
    allcells = []
    for model, arms in m.items():
        for a in arms:
            if a:
                allcells.append(("main/" + model, a))
    for k, a in npr.items():
        allcells.append(("np/" + k, a))
    for k, a in pr.items():
        allcells.append(("prof/" + k, a))
    allcells.sort(key=lambda t: (t[1]["unpars"] or 0), reverse=True)
    print("\nhighest unparsable rates among paper cells:")
    for tag, a in allcells[:20]:
        print(f"  {tag:34s} unpars={a['unpars']:5.1f}% n_unpars={a['n_unpars']:3d} "
              f"acc={a['acc']} fpr={a['fpr']} rec={a['rec']}")


if __name__ == "__main__":
    main()
