"""
Paired comparison of two prediction files on the same test rows.

Use it to compare inference-time unrolling against the plain SFT predictions
of the SAME checkpoint, or any two prediction jsonl files that share `idx`
over the same ground-truth ChatML test set.

    python context_unrolling/compare_paired.py \
        --a Claude/SFT/new_outputs/Qwen3-4B-Instruct/test_predictions_w_schema.jsonl --a-name "SFT direct" \
        --b context_unrolling/outputs/infer_ckpt950/given.jsonl --b-name "SFT + unrolled (given)" \
        --gt Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl

Shard files can be merged first with --merge:

    python context_unrolling/compare_paired.py --merge context_unrolling/outputs/infer_ckpt950/given_shard*.jsonl \
        --out context_unrolling/outputs/infer_ckpt950/given.jsonl

Reports, on the intersection of idx present in both files:
  verdict accuracy / balanced accuracy / FNR / FPR, McNemar-style flip counts
  category macro-F1, micro-F1, samples-F1 with paired bootstrap CIs on the delta
  per-category F1 for both, with the delta
"""

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import unroll_config as C  # noqa: E402

CATS = C.RISK_CATEGORIES


def load_preds(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "idx" in r:
                out[int(r["idx"])] = r
    return out


def extract_json_dict(text):
    if not text:
        return None
    for noise in ("</tool_call>", "<tool_call>", "<think>", "</think>"):
        text = text.replace(noise, "")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group())
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            msg = next((m["content"] for m in ex["messages"] if m["role"] == "assistant"), "")
            d = extract_json_dict(msg)
            if d and isinstance(d.get("risk_analysis"), dict):
                ra = {c: bool(d["risk_analysis"].get(c, d["risk_analysis"].get(c.replace("-", "\u2013"), False))) for c in CATS}
                gt[i] = {"is_safe": bool(d.get("is_safe")), "ra": ra}
    return gt


def norm_ra(r):
    ra = r.get("pred_risk_analysis") or {}
    return {c: bool(ra.get(c, ra.get(c.replace("-", "\u2013"), False))) for c in CATS}


# ------------------------------------------------------------------ metrics
def verdict_stats(P, gt, idxs):
    tp = fp = tn = fn = 0
    for i in idxs:
        unsafe_gt = not gt[i]["is_safe"]
        unsafe_pr = not P[i]["pred_is_safe"]
        if unsafe_gt and unsafe_pr:
            tp += 1
        elif unsafe_gt and not unsafe_pr:
            fn += 1
        elif not unsafe_gt and unsafe_pr:
            fp += 1
        else:
            tn += 1
    acc = (tp + tn) / max(len(idxs), 1)
    fnr = fn / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    bal = ((tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1))) / 2
    return {"accuracy": acc, "balanced_accuracy": bal, "FNR_missed_unsafe": fnr, "FPR_over_block": fpr,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def macro_f1(P, gt, idxs):
    fs = []
    for c in CATS:
        tp = fp = fn = 0
        for i in idxs:
            y = gt[i]["ra"][c]
            p = norm_ra(P[i])[c]
            tp += y and p
            fp += (not y) and p
            fn += y and (not p)
        if tp + fn == 0:
            continue
        fs.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(fs) / len(fs) if fs else float("nan")


def micro_f1(P, gt, idxs):
    tp = fp = fn = 0
    for i in idxs:
        pr = norm_ra(P[i])
        for c in CATS:
            y, p = gt[i]["ra"][c], pr[c]
            tp += y and p
            fp += (not y) and p
            fn += y and (not p)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def samples_f1(P, gt, idxs):
    tot = 0.0
    for i in idxs:
        pr = norm_ra(P[i])
        ps = {c for c in CATS if pr[c]}
        gs = {c for c in CATS if gt[i]["ra"][c]}
        if not ps and not gs:
            tot += 1.0
        else:
            tp = len(ps & gs)
            tot += 0.0 if tp == 0 else 2 * tp / (len(ps) + len(gs))
    return tot / max(len(idxs), 1)


def per_cat(P, gt, idxs):
    out = {}
    for c in CATS:
        tp = fp = fn = 0
        for i in idxs:
            y = gt[i]["ra"][c]
            p = norm_ra(P[i])[c]
            tp += y and p
            fp += (not y) and p
            fn += y and (not p)
        out[c] = {"support": tp + fn, "f1": (float("nan") if tp + fn == 0 else (0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)))}
    return out


def paired_boot(fn, PA, PB, gt, idxs, n=2000, seed=0):
    rnd = random.Random(seed)
    d = []
    for _ in range(n):
        s = [rnd.choice(idxs) for _ in idxs]
        d.append(fn(PB, gt, s) - fn(PA, gt, s))
    d.sort()
    return d[int(0.025 * n)], d[int(0.975 * n)], sum(x > 0 for x in d) / n


# ------------------------------------------------------------------ main
def merge(patterns, out):
    rows = {}
    for pat in patterns:
        for p in glob.glob(pat):
            rows.update(load_preds(p))
    with open(out, "w", encoding="utf-8") as f:
        for i in sorted(rows):
            f.write(json.dumps(rows[i], ensure_ascii=False, default=str) + "\n")
    print(f"merged {len(rows)} rows -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merge", nargs="*", help="glob(s) of shard jsonl files to merge into --out")
    ap.add_argument("--out", default=None)
    ap.add_argument("--a", help="baseline predictions jsonl")
    ap.add_argument("--b", help="comparison predictions jsonl")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--gt", default=str(C.PROJECT_ROOT / "Claude/SFT/new_data_chatml_qwen_and_qwenguard/test.jsonl"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    if args.merge:
        if not args.out:
            sys.exit("--merge needs --out")
        merge(args.merge, args.out)
        if not (args.a and args.b):
            return

    if not (args.a and args.b):
        sys.exit("--a and --b are required")

    gt = load_gt(args.gt)
    PA, PB = load_preds(args.a), load_preds(args.b)
    idxs = sorted(set(PA) & set(PB) & set(gt))
    print(f"{args.a_name}: {len(PA)} rows | {args.b_name}: {len(PB)} rows | paired on {len(idxs)} rows\n")

    A, B = args.a_name, args.b_name
    w = max(len(A), len(B), 12) + 2

    print("VERDICT")
    va, vb = verdict_stats(PA, gt, idxs), verdict_stats(PB, gt, idxs)
    print(f"{'':<28s}{A:>{w}s}{B:>{w}s}{'delta':>10s}")
    for k in ("accuracy", "balanced_accuracy", "FNR_missed_unsafe", "FPR_over_block"):
        print(f"{k:<28s}{va[k]:>{w}.3f}{vb[k]:>{w}.3f}{vb[k]-va[k]:>+10.3f}")
    ok = lambda P, i: P[i]["pred_is_safe"] == gt[i]["is_safe"]  # noqa: E731
    b_only = sum(ok(PB, i) and not ok(PA, i) for i in idxs)
    a_only = sum(ok(PA, i) and not ok(PB, i) for i in idxs)
    both_wrong = sum((not ok(PA, i)) and (not ok(PB, i)) for i in idxs)
    lo, hi, p = paired_boot(lambda P, g, s: verdict_stats(P, g, s)["accuracy"], PA, PB, gt, idxs, args.n_boot)
    print(f"  {B} right / {A} wrong: {b_only}   {A} right / {B} wrong: {a_only}   both wrong: {both_wrong}")
    print(f"  paired bootstrap d(accuracy) 95% CI [{lo:+.3f}, {hi:+.3f}]  P(d>0)={p:.2f}")

    print("\nCATEGORIES (personalization)")
    print(f"{'':<28s}{A:>{w}s}{B:>{w}s}{'delta':>10s}{'95% CI':>20s}{'P(d>0)':>8s}")
    for name, fn in (("macro-F1", macro_f1), ("micro-F1", micro_f1), ("samples-F1", samples_f1)):
        a, b = fn(PA, gt, idxs), fn(PB, gt, idxs)
        lo, hi, p = paired_boot(fn, PA, PB, gt, idxs, args.n_boot)
        print(f"{name:<28s}{a:>{w}.3f}{b:>{w}.3f}{b-a:>+10.3f}{f'[{lo:+.3f},{hi:+.3f}]':>20s}{p:>8.2f}")

    print("\nPER-CATEGORY F1")
    pa, pb = per_cat(PA, gt, idxs), per_cat(PB, gt, idxs)
    print(f"{'category':<40s}{'supp':>5s}{A:>{w}s}{B:>{w}s}{'delta':>8s}")
    for c in CATS:
        if pa[c]["support"] == 0:
            continue
        print(f"{c:<40s}{pa[c]['support']:>5d}{pa[c]['f1']:>{w}.3f}{pb[c]['f1']:>{w}.3f}{pb[c]['f1']-pa[c]['f1']:>+8.3f}")

    # parse health and cost
    print("\nHEALTH / COST")
    for name, P in ((A, PA), (B, PB)):
        n = len(idxs)
        unp = sum(not P[i].get("parsed_ok", True) for i in idxs)
        toks = sum(P[i].get("n_generated_tokens", 0) for i in idxs) / n
        secs = sum(P[i].get("gen_seconds", 0) for i in idxs) / n
        print(f"  {name:<{w}s} unparseable {unp}/{n}   mean tokens {toks:.0f}   mean sec {secs:.1f}")


if __name__ == "__main__":
    main()
