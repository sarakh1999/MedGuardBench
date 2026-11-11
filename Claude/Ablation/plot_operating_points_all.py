#!/usr/bin/env python3
"""Figure 1 layout (recall on unsafe vs FPR on safe) for every backbone we have.

Same systems and colours as fig_operating_points.png in personalization_report.py:
  zero-shot, no profile            (base, noprofile)
  zero-shot, with profile          (base, true)
  non-personalized SFT             (sft-noprofile, nonpersonalized)   trained & tested without profiles
  personalized SFT, profile withheld (sft, noprofile)
  personalized SFT / CU / GRPO     (arm, true)
Marker = backbone. Runs with no parsable verdict are dropped; points below the
window are listed in a note instead of being cropped silently.

Outputs (Claude/Ablation/report/):
  fig_operating_points_all.png     colour = system, marker = backbone, shaded target / over-blocking zones
  fig_operating_points_simple.png  colour = profile at inference vs not (zero-shot grey), one marker
  fig_operating_points_all.csv
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from personalization_report import ROOT, OUT_DIR, NOPROFILE_SFT, load_gold, load_preds, metrics, RISK_CATEGORIES  # noqa: E402

ARMS = ["base", "sft", "cu", "grpo"]
DISPLAY = {
    "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B", "qwen3-14b": "Qwen3-14B",
    "qwen3guard-4b": "Qwen3Guard-4B", "qwen3guard-8b": "Qwen3Guard-8B",
    "llama31-8b": "Llama-3.1-8B", "llamaguard2-8b": "Llama-Guard-2-8B", "llamaguard-7b": "Llama-Guard-7B",
    "shieldgemma-2b": "ShieldGemma-2B", "shieldgemma-9b": "ShieldGemma-9B",
    "wildguard-7b": "WildGuard-7B", "medgemma-4b": "MedGemma-4B", "llama3.3-70b": "Llama-3.3-70B",
}
MODEL_ORDER = list(DISPLAY)
MARKERS = dict(zip(MODEL_ORDER, "osD^vP*Xh<>pd"))
SYS_ORDER = ["zero-shot, no profile", "zero-shot, with profile", "non-personalized SFT (trained w/o profiles)",
             "personalized SFT, profile withheld", "personalized SFT", "personalized CU", "personalized GRPO"]
SYS_COL = {"zero-shot, no profile": "#bbbbbb", "zero-shot, with profile": "#777777",
           "non-personalized SFT (trained w/o profiles)": "#E1812C", "personalized SFT, profile withheld": "#F2C572",
           "personalized SFT": "#4C72B0", "personalized CU": "#55A868", "personalized GRPO": "#C44E52"}
XLIM, YLIM = (0, 85), (60, 101)
TARGET = dict(recall=95, fpr=20)   # target box: recall >= 95, FPR <= 20
OVERBLOCK_FPR = 40                 # over-blocking band: FPR >= 40

# simplified grouping: does the model see the profile when it decides?
GROUP = {
    "zero-shot, no profile": "zero-shot",
    "zero-shot, with profile": "zero-shot",
    "non-personalized SFT (trained w/o profiles)": "no profile at inference",
    "personalized SFT, profile withheld": "no profile at inference",
    "personalized SFT": "profile at inference",
    "personalized CU": "profile at inference",
    "personalized GRPO": "profile at inference",
}
GROUP_ORDER = ["zero-shot", "no profile at inference", "profile at inference"]
GROUP_COL = {"zero-shot": "#9a9a9a", "no profile at inference": "#E1812C", "profile at inference": "#4C72B0"}
GROUP_LABEL = {
    "zero-shot": "zero-shot (with or without profile)",
    "no profile at inference": "trained, no profile at inference\n(non-personalized SFT; personalized SFT with profile withheld)",
    "profile at inference": "trained, profile at inference\n(personalized SFT / CU / GRPO)",
}


def draw_zones(ax, plt):
    """Target box (top-left) and over-blocking band (right), behind the points."""
    ax.add_patch(plt.Rectangle((XLIM[0], TARGET["recall"]), TARGET["fpr"] - XLIM[0], YLIM[1] - TARGET["recall"],
                               facecolor="#55A868", alpha=0.10, edgecolor="#55A868", lw=0.8, ls="--", zorder=0))
    ax.add_patch(plt.Rectangle((OVERBLOCK_FPR, YLIM[0]), XLIM[1] - OVERBLOCK_FPR, YLIM[1] - YLIM[0],
                               facecolor="#E1812C", alpha=0.08, edgecolor="#E1812C", lw=0.8, ls="--", zorder=0))
    ax.text(XLIM[0] + 0.6, YLIM[1] - 0.4, f"target: recall $\\geq$ {TARGET['recall']}, FPR $\\leq$ {TARGET['fpr']}",
            fontsize=7, color="#2e6b3e", ha="left", va="top")
    ax.text(XLIM[1] - 0.8, YLIM[1] - 0.4, f"over-blocking: FPR $\\geq$ {OVERBLOCK_FPR}%",
            fontsize=7.5, color="#9a4f12", ha="right", va="top")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-zones", action="store_true", help="omit the shaded target / over-blocking zones and cluster labels")
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gold = load_gold("true"); n = len(gold)
    gv = np.array([g[0] for g in gold]); gc = np.array([[g[1][c] for c in RISK_CATEGORIES] for g in gold])
    import recompute_729 as R
    from rescore_regen_base_729 import DRIFT_ROWS
    _, keep = R.keep_mask()
    keep = np.asarray(keep, dtype=bool).copy()
    keep[DRIFT_ROWS] = False
    idx = np.flatnonzero(keep)
    print(f"scoring the {len(idx)}-row drug-disjoint test split (dropped {n - len(idx)} cross-split rows)")

    runs = {}
    for p in glob.glob(os.path.join(ROOT, "Claude/Ablation/outputs/*/*/*/test_predictions.jsonl")):
        model, arm, cond = p.split(os.sep)[-4:-1]
        if arm in ARMS and cond in ("true", "noprofile"):
            runs[(model, arm, cond)] = p
    for run, model in NOPROFILE_SFT.items():
        p = os.path.join(ROOT, "Claude/SFT/new_outputs", run, "test_predictions_sft.jsonl")
        if os.path.exists(p):
            runs[(model, "sft-noprofile", "nonpersonalized")] = p
    from audit_729 import DIRKEY, load_any
    for k in DIRKEY.values():
        runs.pop((k, "base", "true"), None)  # Table 1 cells come from the authoritative scores below

    res = {}
    for k, p in sorted(runs.items()):
        pv, pc, parsed, _, n_rows = load_any(p, RISK_CATEGORIES)
        if n_rows < n:
            print(f"  skip {k}: {n_rows}/{n} rows"); continue
        if not parsed.any():
            print(f"  skip {k}: no parsable verdicts"); continue
        res[k] = metrics(gv, gc, pv, pc, idx[parsed[idx]])  # unparsable rows excluded, as in the tables

    # Truncated prediction files: fall back to the Table 7 (profile perturbation) scores
    prof = json.load(open(os.path.join(R.OUT_DIR, "paper", "profile_729_abstain.json")))
    for k in runs:
        c = prof.get("|".join(k))
        if k not in res and c and c.get("rec") is not None:
            res[k] = {"recall": c["rec"] / 100, "fpr": c["fpr"] / 100, "acc": c["acc"] / 100, "macro_f1": c["mac"] / 100}
            print(f"  {k}: using Table 7 scores")

    # Table 1 cells: same scores as the typeset table (some prediction files are no longer on disk)
    main = json.load(open(os.path.join(R.OUT_DIR, "paper", "main_729_abstain.json")))
    for key, c in json.load(open(os.path.join(R.OUT_DIR, "paper", "regen_base_729_abstain.json"))).items():
        main[key][0] = c
    main["lg2"][0] = {"rec": 97.4, "fpr": 80.3, "acc": 60.1, "mac": 0.0}  # outputs/llamaguard2-8b/base/true
    for key, arms in main.items():
        for arm, c in zip(ARMS, arms):
            if c and c.get("rec") is not None:
                res[(DIRKEY[key], arm, "true")] = {"recall": c["rec"] / 100, "fpr": c["fpr"] / 100,
                                                   "acc": c["acc"] / 100, "macro_f1": c["mac"] / 100}

    pts = []  # (model, system, metrics)
    for model in MODEL_ORDER:
        systems = []
        if (model, "base", "noprofile") in res: systems.append(("zero-shot, no profile", (model, "base", "noprofile")))
        if (model, "base", "true") in res: systems.append(("zero-shot, with profile", (model, "base", "true")))
        if (model, "sft-noprofile", "nonpersonalized") in res: systems.append(("non-personalized SFT (trained w/o profiles)", (model, "sft-noprofile", "nonpersonalized")))
        if (model, "sft", "noprofile") in res: systems.append(("personalized SFT, profile withheld", (model, "sft", "noprofile")))
        for arm, name in (("sft", "personalized SFT"), ("cu", "personalized CU"), ("grpo", "personalized GRPO")):
            if (model, arm, "true") in res: systems.append((name, (model, arm, "true")))
        if len(systems) < 2:
            continue
        for name, k in systems:
            pts.append((model, name, res[k]))

    out_csv = os.path.join(OUT_DIR, "fig_operating_points_all.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["backbone", "system", "recall_unsafe", "fpr_safe", "acc", "macro_f1"])
        for model, name, m in pts:
            w.writerow([DISPLAY[model], name, f"{100*m['recall']:.1f}", f"{100*m['fpr']:.1f}", f"{100*m['acc']:.1f}", f"{100*m['macro_f1']:.1f}"])
    print(f"{'backbone':18} {'system':44} {'recall':>6} {'FPR':>6}")
    for model, name, m in pts:
        print(f"{DISPLAY[model]:18} {name:44} {100*m['recall']:6.1f} {100*m['fpr']:6.1f}")
    print(f"\n{len(pts)} points, {len({p[0] for p in pts})} backbones")

    inside = [p for p in pts if XLIM[0] <= 100 * p[2]["fpr"] <= XLIM[1] and YLIM[0] <= 100 * p[2]["recall"] <= YLIM[1]]
    off = [p for p in pts if p not in inside]

    # FPR range per group, printed so the caption can quote it
    for g in GROUP_ORDER:
        f = [100 * m["fpr"] for mo, na, m in pts if GROUP[na] == g]
        print(f"{g:26} FPR {min(f):5.1f}–{max(f):5.1f}  median {np.median(f):5.1f}  (n={len(f)})")

    def off_note(ax):
        if not off:
            return
        txt = "below window (near-zero recall = flags almost nothing):\n" + "\n".join(
            f"{DISPLAY[mo]} {na}: recall {100*m['recall']:.1f}, FPR {100*m['fpr']:.1f}" for mo, na, m in off)
        ax.text(0.99, 0.06 if not args.no_zones else 0.02, txt, transform=ax.transAxes, fontsize=6.6, ha="right", va="bottom",
                color="#333333", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

    def axes_common(ax):
        ax.axhline(95, color="grey", ls=":", lw=0.8); ax.text(TARGET["fpr"] + 0.8, 95.5, "95 % recall", fontsize=8, color="grey")
        ax.set_xlabel("false-positive rate on safe prescriptions (over-blocking, %)")
        ax.set_ylabel("recall on unsafe prescriptions (%)")
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.grid(alpha=0.3); ax.set_axisbelow(True)

    def group_stats(g):
        f = [100 * m["fpr"] for mo, na, m in pts if GROUP[na] == g]
        return min(f), max(f), float(np.median(f))

    def cluster_labels(ax):
        # Group descriptions, not pointers to individual runs: no arrows.
        lo, hi, med = group_stats("profile at inference")
        ax.text(24.5, 80.5,
                "blue / green / red: trained models that see the\n"
                f"patient profile when they decide\n"
                f"FPR {lo:.0f}–{hi:.0f}%, median {med:.0f}%",
                fontsize=7.8, color="#2e4a7a", ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#4C72B0", lw=0.6, alpha=0.9))
        lo, hi, med = group_stats("no profile at inference")
        ax.text(41.5, 77.5,
                "orange / gold: trained models that do NOT see the\n"
                "profile when they decide (non-personalized SFT;\n"
                "personalized SFT with the profile withheld)\n"
                f"FPR {lo:.0f}–{hi:.0f}%, median {med:.0f}%",
                fontsize=7.8, color="#9a4f12", ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E1812C", lw=0.6, alpha=0.9))

    # ---------------------------------------------------------------- full figure: colour = system, marker = backbone
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    if not args.no_zones:
        draw_zones(ax, plt)
    for model, name, m in inside:
        ax.scatter(100 * m["fpr"], 100 * m["recall"], s=90, marker=MARKERS[model], color=SYS_COL[name],
                   edgecolor="k", linewidth=0.5, zorder=3)
    axes_common(ax)
    if not args.no_zones:
        cluster_labels(ax)
    ax.set_title("Operating points, all backbones: without the profile at inference, safety is only reachable by over-blocking", fontsize=10)
    off_note(ax)
    present_models = [m for m in MODEL_ORDER if any(p[0] == m for p in pts)]
    h1 = [ax.scatter([], [], color=SYS_COL[s], s=60, label=s) for s in SYS_ORDER if any(p[1] == s for p in pts)]
    leg1 = ax.legend(handles=h1, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="system (colour)", title_fontsize=8, frameon=False)
    ax.add_artist(leg1)
    h2 = [ax.scatter([], [], marker=MARKERS[m], color="w", edgecolor="k", s=60, label=DISPLAY[m]) for m in present_models]
    ax.legend(handles=h2, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 0.62), title="backbone (marker)", title_fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_operating_points_all.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")

    # ---------------------------------------------------------------- single-column version (ACL \columnwidth ~ 3.03in)
    # Same data and colours. Legends sit inside the empty lower-right area; the
    # below-window runs are described in the caption instead of a note.
    fig, ax = plt.subplots(figsize=(3.3, 3.5))
    if not args.no_zones:
        draw_zones(ax, plt)
        for t in ax.texts:
            t.set_fontsize(5.2)
    for model, name, m in inside:
        ax.scatter(100 * m["fpr"], 100 * m["recall"], s=26, marker=MARKERS[model], color=SYS_COL[name],
                   edgecolor="k", linewidth=0.35, zorder=3)
    ax.axhline(95, color="grey", ls=":", lw=0.6); ax.text(26, 95.6, "95 % recall", fontsize=5.2, color="grey")
    ax.set_xlabel("false-positive rate on safe prescriptions (%)", fontsize=7, labelpad=2)
    ax.set_ylabel("recall on unsafe prescriptions (%)", fontsize=7, labelpad=2)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.grid(alpha=0.3, lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    h1 = [ax.scatter([], [], color=SYS_COL[s], s=18, edgecolor="k", linewidth=0.35, label=s.replace(" (trained w/o profiles)", ""))
          for s in SYS_ORDER if any(p[1] == s for p in pts)]
    leg1 = ax.legend(handles=h1, fontsize=4.9, loc="lower right", bbox_to_anchor=(1.0, 0.0), frameon=True, framealpha=0.92,
                     edgecolor="#cccccc", borderpad=0.4, labelspacing=0.3, handletextpad=0.3, handlelength=1.0,
                     title="system (colour)", title_fontsize=5.3)
    leg1.get_frame().set_linewidth(0.4)
    ax.add_artist(leg1)
    h2 = [ax.scatter([], [], marker=MARKERS[m], color="w", edgecolor="k", s=18, linewidth=0.5, label=DISPLAY[m]) for m in present_models]
    leg2 = ax.legend(handles=h2, fontsize=4.9, loc="lower right", bbox_to_anchor=(1.0, 0.235), ncol=2, frameon=True,
                     framealpha=0.92, edgecolor="#cccccc", borderpad=0.4, labelspacing=0.3, handletextpad=0.3,
                     columnspacing=0.8, handlelength=1.0, title="backbone (marker)", title_fontsize=5.3)
    leg2.get_frame().set_linewidth(0.4)
    fig.tight_layout(pad=0.3)
    out = os.path.join(OUT_DIR, "fig_operating_points_all_col.png")
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print(f"wrote {out}")

    # ---------------------------------------------------------------- simplified figure: colour = profile at inference or not
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    draw_zones(ax, plt)
    for g in GROUP_ORDER:  # draw grey first so the coloured groups sit on top
        sel = [p for p in inside if GROUP[p[1]] == g]
        ax.scatter([100 * m["fpr"] for _, _, m in sel], [100 * m["recall"] for _, _, m in sel], s=62,
                   color=GROUP_COL[g], edgecolor="k", linewidth=0.5, zorder=3, alpha=0.9)
    axes_common(ax)
    cluster_labels(ax)
    ax.set_title("What moves a guardrail to the safe region is the profile at decision time", fontsize=10)
    off_note(ax)
    h = [ax.scatter([], [], color=GROUP_COL[g], edgecolor="k", linewidth=0.5, s=55, label=GROUP_LABEL[g]) for g in GROUP_ORDER]
    # legend sits in the empty lower-right area, above the off-window note
    ax.legend(handles=h, fontsize=7.0, loc="lower right", bbox_to_anchor=(1.0, 0.145), frameon=True, framealpha=0.95,
              title=f"{len(pts)} runs, {len(present_models)} backbones; each point = one backbone × system", title_fontsize=7.2)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_operating_points_simple.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
