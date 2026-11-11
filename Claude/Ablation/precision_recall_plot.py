#!/usr/bin/env python3
"""Precision-vs-recall dot plot of every model x arm we have on the true-profile test set (754 scenarios).

Precision / recall are for the *unsafe* class (the guardrail's positive class). One parser
(reward.parse_completion) for every file; unparsed completions count as "safe" (as in the evals).

Sources, in priority order per (model, arm):
  1. Claude/Ablation/outputs/<model>/<arm>/true/test_predictions.jsonl         (vLLM greedy, same engine for all arms)
  2. arm-specific run dirs (HF greedy): SFT new_outputs, context_unrolling ontop_*, GRPO final/eval, DPO outputs.
Non-personalised SFT (trained and tested without profiles) is drawn as hollow markers for reference.

Outputs: Claude/Ablation/report/fig_precision_recall.png, precision_recall_table.md
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from personalization_report import ROOT, load_gold, load_preds, RISK_CATEGORIES, OUT_DIR  # noqa: E402

A = os.path.join(ROOT, "Claude/Ablation/outputs")
S = os.path.join(ROOT, "Claude/SFT/new_outputs")
C = os.path.join(ROOT, "context_unrolling/outputs")
G = os.path.join(ROOT, "Claude/GRPO/outputs")
D = os.path.join(ROOT, "Claude/DPO/outputs/partial_miss_audited")
B = os.path.join(ROOT, "Claude/Base_Models/outputs_blind_v2")

# model key -> (display name, SFT run dir, CU ontop dir, GRPO run prefix, DPO dir, base dir)
MODELS = {
    "qwen3-4b":      ("Qwen3-4B",          "Qwen3-4B-Instruct",  "ontop",              "legacy_v1",                "qwen3-4b", "Qwen3-4B-Instruct"),
    "qwen3-8b":      ("Qwen3-8B",          "Qwen3-8B-Instruct",  "ontop_8b",           "legacy_v1-qwen3-8b",       "qwen3-8b", None),
    "qwen3-14b":     ("Qwen3-14B",         "Qwen3-14B-Instruct", "ontop_14b",          "legacy_v1-qwen3-14b",      None,       None),
    "qwen3guard-4b": ("Qwen3Guard-4B",     "Qwen3Guard-Gen-4B",  "ontop_guard",        "legacy_v1-qwen3guard-4b",  None,       "Qwen3Guard-Gen-4B"),
    "qwen3guard-8b": ("Qwen3Guard-8B",     "Qwen3Guard-Gen-8B",  "ontop_guard8b",      "legacy_v1-qwen3guard-8b",  None,       None),
    "shieldgemma-2b": ("ShieldGemma-2B",   "ShieldGemma-2B",     "ontop_shieldgemma2b", "legacy_v1-shieldgemma-2b", None,      None),
    "shieldgemma-9b": ("ShieldGemma-9B",   "ShieldGemma-9B",     "ontop_shieldgemma9b", "legacy_v1-shieldgemma-9b", None,      None),
    "llamaguard2-8b": ("LlamaGuard-2-8B",  "Llama-Guard-2-8B",   "ontop_llamaguard2",  "legacy_v1-llamaguard2-8b", None,       None),
    "llamaguard-7b": ("LlamaGuard-7B",     "LlamaGuard-7b",      "ontop_llamaguard7b", None,                       None,       None),
    "llama31-8b":    ("Llama-3.1-8B",      "Llama-3.1-8B",       "ontop_llama31",      "legacy_v1-llama31-8b",     None,       None),
    "wildguard-7b":  ("WildGuard-7B",      "WildGuard-7B",       "ontop_wildguard7b",  "legacy_v1-wildguard-7b",   None,       None),
    "medgemma-4b":   ("MedGemma-4B",       "MedGemma-4B-it",     "ontop_medgemma4b",   "legacy_v1-medgemma-4b",    None,       None),
    "llama3.3-70b":  ("Llama-3.3-70B (0-shot)", None,            None,                 None,                       None,       None),
}
NOPROFILE_SFT = {"qwen3-4b": "Qwen3-4B-noprofile", "qwen3-8b": "Qwen3-8B-noprofile", "qwen3-14b": "Qwen3-14B-noprofile",
                 "qwen3guard-4b": "Qwen3Guard-4B-noprofile", "qwen3guard-8b": "Qwen3Guard-8B-noprofile",
                 "shieldgemma-2b": "ShieldGemma-2B-noprofile"}
# Non-personalised CU / GRPO arms listed in the paper's *-NP table (trained and tested without profiles).
NOPROFILE_CU = {"qwen3-4b": "ontop_noprofile", "qwen3-14b": "ontop_14b_noprofile",
                "qwen3guard-4b": "ontop_guard_noprofile", "qwen3guard-8b": "ontop_guard8b_noprofile"}
NOPROFILE_GRPO = {"qwen3-4b": "legacy_v1-noprofile", "qwen3-14b": "legacy_v1-qwen3-14b-noprofile",
                  "qwen3guard-4b": "legacy_v1-qwen3guard-4b-noprofile"}
ARMS = ["base", "sft", "dpo", "cu", "grpo"]
NP_ARMS = ["sft-noprofile", "cu-noprofile", "grpo-noprofile"]
ARM_LABEL = {"base": "zero-shot (base)", "sft": "SFT", "dpo": "DPO", "cu": "CU v4pro", "grpo": "guided GRPO",
             "sft-noprofile": "non-personalised SFT\n(no profile at train & test)",
             "cu-noprofile": "non-personalised CU\n(no profile at train & test)",
             "grpo-noprofile": "non-personalised GRPO\n(no profile at train & test)"}
ARM_COL = {"base": "#9e9e9e", "sft": "#1f77b4", "dpo": "#9467bd", "cu": "#2ca02c", "grpo": "#d62728",
           "sft-noprofile": "#ff7f0e", "cu-noprofile": "#2ca02c", "grpo-noprofile": "#d62728"}


def first(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def find_runs():
    from audit_729 import table1_files
    global T1
    T1 = table1_files()
    runs = {}
    for key, (_, sft, cu, grpo, dpo, base) in MODELS.items():
        cand = {
            "base": [f"{A}/{key}/base/true/test_predictions.jsonl", base and f"{B}/{base}/test_predictions_base.jsonl"],
            "sft": [f"{A}/{key}/sft/true/test_predictions.jsonl", sft and f"{S}/{sft}/test_predictions_sft.jsonl",
                    sft and f"{S}/{sft}/test_predictions_w_schema.jsonl"],
            "dpo": [dpo and f"{D}/{dpo}/test_predictions_dpo.jsonl"],
            "cu": [f"{A}/{key}/cu/true/test_predictions.jsonl", cu and f"{C}/{cu}/sft+1ep_unrolled_v4pro/test_predictions.jsonl"],
            "grpo": [f"{A}/{key}/grpo/true/test_predictions.jsonl",
                     grpo and f"{G}/{grpo}/grpo-guided-v3/final_selected/eval/test_predictions.jsonl",
                     grpo and f"{G}/{grpo}/grpo-guided-v2/final/eval/test_predictions.jsonl"],
        }
        for arm, paths in cand.items():
            p = first(*paths)
            if p:
                runs[(key, arm)] = p
        for arm in ("base", "sft", "cu", "grpo"):
            if (key, arm) in T1:
                runs[(key, arm)] = T1[(key, arm)]
        if key in NOPROFILE_SFT:
            p = first(f"{S}/{NOPROFILE_SFT[key]}/test_predictions_sft.jsonl")
            if p:
                runs[(key, "sft-noprofile")] = p
        if key in NOPROFILE_CU:
            p = first(f"{C}/{NOPROFILE_CU[key]}/sft+1ep_unrolled_v4pro/test_predictions.jsonl")
            if p:
                runs[(key, "cu-noprofile")] = p
        if key in NOPROFILE_GRPO:
            p = first(f"{G}/{NOPROFILE_GRPO[key]}/grpo-guided-v2/final/eval/test_predictions.jsonl")
            if p:
                runs[(key, "grpo-noprofile")] = p
    return runs


def pr(gv, pv):
    unsafe_g, unsafe_p = ~gv, ~pv
    tp = np.sum(unsafe_g & unsafe_p); fp = np.sum(~unsafe_g & unsafe_p); fn = np.sum(unsafe_g & ~unsafe_p); tn = np.sum(~unsafe_g & ~unsafe_p)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9), fpr=fp / max(fp + tn, 1),
                acc=(tp + tn) / len(gv), bal_acc=0.5 * (rec + tn / max(fp + tn, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gold = load_gold("true"); n_full = len(gold)
    gv = np.array([g[0] for g in gold]); gc = np.array([[g[1][c] for c in RISK_CATEGORIES] for g in gold])
    import recompute_729 as R
    _, keep = R.keep_mask()
    idx = np.flatnonzero(keep)
    n = len(idx)
    print(f"precision-recall on the {n}-row drug-disjoint test split")
    runs = find_runs()
    rows = []
    from audit_729 import load_any
    for (key, arm), p in sorted(runs.items()):
        pv, pc, parsed, _, n_rows = load_any(p, RISK_CATEGORIES)
        if n_rows < n_full:
            print(f"  skip {key}/{arm}: {n_rows}/{n_full} rows ({p})"); continue
        if not parsed[idx].any():
            print(f"  skip {key}/{arm}: no parsable verdicts ({p})"); continue
        gv_i, gc_i, pv_i, pc_i = gv[idx], gc[idx], pv[idx], pc[idx]
        m = pr(gv_i, pv_i)
        ctp = np.sum(gc_i & pc_i, axis=0); cfp = np.sum(~gc_i & pc_i, axis=0); cfn = np.sum(gc_i & ~pc_i, axis=0)
        f1c = 2 * ctp / np.maximum(2 * ctp + cfp + cfn, 1); present = (ctp + cfn) > 0
        m["macro_f1"] = f1c[present].mean(); m["unparsed"] = int((~parsed[idx]).sum())
        m["engine"] = "vLLM" if p.startswith(A) else "HF"
        rows.append((key, arm, m, os.path.relpath(p, ROOT)))

    # ------------------------------------------------------------ figure
    models = [k for k in MODELS if any(r[0] == k for r in rows)]
    markers = dict(zip(models, "osD^vP*Xh<>pd8"))
    XLIM, YLIM = (80, 100.5), (55, 101)
    fig, ax = plt.subplots(figsize=(10.5, 6.6))

    def draw(a, pts, iso_labels=True, size=1.0):
        # iso-F1 curves
        rr = np.linspace(0.3, 1.0, 600)
        for f in (0.80, 0.85, 0.90, 0.95):
            pp = f * rr / np.maximum(2 * rr - f, 1e-9); ok = (pp <= 1) & (pp >= 0.3)
            a.plot(100 * rr[ok], 100 * pp[ok], color="lightgrey", lw=0.8, ls="--", zorder=1)
            if iso_labels:
                a.text(100 * rr[ok][-1] - 0.3, 100 * pp[ok][-1] + 0.4, f"F1={f:.2f}", fontsize=7, color="grey", ha="right")
        for key, arm, m, _ in pts:
            hollow = arm in NP_ARMS
            a.scatter(100 * m["recall"], 100 * m["precision"], s=size * (95 if not hollow else 80), marker=markers[key],
                      facecolor="none" if hollow else ARM_COL[arm], edgecolor=ARM_COL[arm] if hollow else "k",
                      linewidth=1.4 if hollow else 0.5, zorder=3)

    draw(ax, rows)
    best = max((r for r in rows if r[1] not in NP_ARMS), key=lambda r: r[2]["f1"])
    ax.annotate(f"best F1: {MODELS[best[0]][0]} {ARM_LABEL[best[1]]}\nP={100*best[2]['precision']:.1f} R={100*best[2]['recall']:.1f}",
                xy=(100 * best[2]["recall"], 100 * best[2]["precision"]), xytext=(80.6, 99), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="k", lw=0.8), ha="left", va="top")

    # points outside the window (e.g. Llama-Guard-7B zero-shot calls almost everything safe)
    off = [r for r in rows if not (XLIM[0] <= 100 * r[2]["recall"] <= XLIM[1] and YLIM[0] <= 100 * r[2]["precision"] <= YLIM[1])]
    if off:
        txt = "outside window:\n" + "\n".join(f"{MODELS[k][0]} {ARM_LABEL[a].splitlines()[0]}: P={100*m['precision']:.1f} R={100*m['recall']:.1f}" for k, a, m, _ in off)
        ax.text(1.02, 0.0, txt, transform=ax.transAxes, fontsize=6.8, ha="left", va="bottom", color="#333333")

    h1 = [ax.scatter([], [], color=ARM_COL[a], s=60, label=ARM_LABEL[a]) for a in ARMS if any(r[1] == a for r in rows)]
    for a in NP_ARMS:
        if any(r[1] == a for r in rows):
            h1.append(ax.scatter([], [], facecolor="none", edgecolor=ARM_COL[a], linewidth=1.4, s=60, label=ARM_LABEL[a]))
    leg1 = ax.legend(handles=h1, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="arm (colour)", title_fontsize=8, frameon=False)
    ax.add_artist(leg1)
    h2 = [ax.scatter([], [], marker=markers[k], color="w", edgecolor="k", s=60, label=MODELS[k][0]) for k in models]
    ax.legend(handles=h2, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 0.60), title="backbone (marker)", title_fontsize=8, frameon=False)
    ax.set_xlabel("recall on unsafe prescriptions (%)"); ax.set_ylabel("precision of 'unsafe' verdicts (%)")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.grid(alpha=0.3)
    ax.set_title("Precision vs recall, every backbone x arm, true-profile test set (n=%d)" % n, fontsize=10)
    fig.tight_layout(); out = os.path.join(args.out_dir, "fig_precision_recall.png"); fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)

    # ------------------------------------------------------------ table
    rows.sort(key=lambda r: -r[2]["f1"])
    md = ["# Precision / recall of the unsafe class — all models x arms, true-profile test (n=%d), sorted by F1\n" % n,
          "| # | backbone | arm | precision | recall | F1 | FPR | acc | bal-acc | cat macro-F1 | unparsed | engine | file |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, (key, arm, m, p) in enumerate(rows, 1):
        md.append(f"| {i} | {MODELS[key][0]} | {ARM_LABEL[arm]} | {100*m['precision']:.1f} | {100*m['recall']:.1f} | **{100*m['f1']:.1f}** | {100*m['fpr']:.1f} | "
                  f"{100*m['acc']:.1f} | {100*m['bal_acc']:.1f} | {100*m['macro_f1']:.1f} | {m['unparsed']} | {m['engine']} | `{p}` |")
    md.append("\n## Best arm per backbone (by F1)\n")
    md.append("| backbone | best arm | F1 | precision | recall | runner-up |"); md.append("|---|---|---|---|---|---|")
    for key in models:
        rs = sorted([r for r in rows if r[0] == key and r[1] not in NP_ARMS], key=lambda r: -r[2]["f1"])
        if rs:
            ru = f"{ARM_LABEL[rs[1][1]]} ({100*rs[1][2]['f1']:.1f})" if len(rs) > 1 else "—"
            md.append(f"| {MODELS[key][0]} | {ARM_LABEL[rs[0][1]]} | {100*rs[0][2]['f1']:.1f} | {100*rs[0][2]['precision']:.1f} | {100*rs[0][2]['recall']:.1f} | {ru} |")
    with open(os.path.join(args.out_dir, "precision_recall_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\nwrote {out} and precision_recall_table.md")


if __name__ == "__main__":
    main()
