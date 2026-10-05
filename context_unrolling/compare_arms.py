"""
Collate compute_metrics.py outputs from the smoke-test arms into one table.

    python context_unrolling/compare_arms.py
    python context_unrolling/compare_arms.py --root context_unrolling/outputs/smoke

Reads <root>/<arm>/test_predictions_metrics.json for every arm present and
prints verdict accuracy, verdict macro-F1, category macro-F1, category
micro-F1, exact-match, parse health and generation cost side by side, plus a
per-category F1 table. Bootstrap 95% CIs are shown where compute_metrics.py
stored them.
"""

import argparse
import json
from pathlib import Path

ARM_ORDER = [
    "sft_baseline",
    "sft+given_ctx_only",
    "sft+1ep_long",
    "sft+1ep_unrolled_long_assistant",
    "sft+1ep_unrolled_long_user",
    "direct",
    "long",
    "unrolled__assistant__patient+prescription",
    "unrolled_long__assistant__patient+prescription",
    "unrolled__user",
    "unrolled_long__user",
]

SHORT = {
    "sft_baseline": "SFT (ckpt-950)",
    "sft+given_ctx_only": "SFT + ctx at inference",
    "sft+1ep_long": "SFT +1ep long (control)",
    "sft+1ep_unrolled_long_assistant": "SFT +1ep unrolled (self)",
    "sft+1ep_unrolled_long_user": "SFT +1ep unrolled (given)",
    "direct": "direct",
    "long": "long (current SFT)",
    "unrolled__assistant__patient+prescription": "unrolled",
    "unrolled_long__assistant__patient+prescription": "unrolled+long",
    "unrolled__user": "unrolled (given)",
    "unrolled_long__user": "unrolled+long (given)",
}


def ci(m, key):
    b = (m.get("bootstrap_95ci") or {}).get(key)
    if isinstance(b, dict) and "lo" in b and "hi" in b:
        return f"[{b['lo']:.3f},{b['hi']:.3f}]"
    if isinstance(b, (list, tuple)) and len(b) == 2:
        return f"[{b[0]:.3f},{b[1]:.3f}]"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent / "outputs" / "smoke"))
    args = ap.parse_args()
    root = Path(args.root)

    arms = {}
    for d in sorted(root.iterdir()) if root.exists() else []:
        mfile = d / "test_predictions_metrics.json"
        if mfile.exists():
            arms[d.name] = json.load(open(mfile))
    if not arms:
        print(f"no metrics under {root}")
        return
    order = [a for a in ARM_ORDER if a in arms] + [a for a in arms if a not in ARM_ORDER]

    def row(label, fn, fmt="{:.3f}"):
        cells = []
        for a in order:
            try:
                v = fn(arms[a])
                cells.append(fmt.format(v) if isinstance(v, (int, float)) else str(v))
            except (KeyError, TypeError):
                cells.append("-")
        print(f"{label:<34s}" + "".join(f"{c:>22s}" for c in cells))

    print(f"{'':<34s}" + "".join(f"{SHORT.get(a, a)[:21]:>22s}" for a in order))
    print("-" * (34 + 22 * len(order)))
    row("n scored", lambda m: m["n_scored"], "{:d}")
    row("verdict accuracy", lambda m: m["verdict"]["accuracy"])
    row("  95% CI", lambda m: ci(m, "accuracy"), "{}")
    row("verdict balanced acc", lambda m: m["verdict"]["balanced_accuracy"])
    row("verdict macro-F1", lambda m: m["verdict"]["macro_f1"])
    row("missed-unsafe rate (FNR)", lambda m: m["verdict"]["false_negative_rate (missed unsafe)"])
    row("over-block rate (FPR)", lambda m: m["verdict"]["false_positive_rate (over-blocking)"])
    print()
    row("category macro-F1", lambda m: m["categories"]["macro_f1"])
    row("  95% CI", lambda m: ci(m, "category_macro_f1"), "{}")
    row("category micro-F1", lambda m: m["categories"]["micro_f1"])
    row("exact match (all 17)", lambda m: m["categories"]["exact_match (subset accuracy)"])
    row("samples-F1", lambda m: m["categories"]["samples_f1"])
    print()
    row("verdict parse rate", lambda m: m["verdict_parse_rate"])
    row("risk_analysis parse rate", lambda m: m["risk_analysis_parse_rate"])
    row("verdict/category inconsistency", lambda m: m["verdict_category_inconsistency_rate"])
    row("mean generated tokens", lambda m: m["mean_generated_tokens"], "{:.0f}")
    row("mean gen seconds", lambda m: m["mean_gen_seconds"], "{:.1f}")

    print("\nper-category F1 (support = positives in the scored subset)")
    cats = {}
    for a in order:
        for c in arms[a].get("per_category", []):
            cats.setdefault(c["category"], {})[a] = c
    print(f"{'category':<40s}{'supp':>5s}" + "".join(f"{SHORT.get(a, a)[:14]:>16s}" for a in order))
    for name, per in cats.items():
        first = next(iter(per.values()))
        supp = first.get("support_gt", first.get("TP", 0) + first.get("FN", 0))
        cells = "".join(f"{per[a]['f1']:>16.3f}" if a in per else f"{'-':>16s}" for a in order)
        print(f"{name:<40s}{supp:>5d}{cells}")


if __name__ == "__main__":
    main()
