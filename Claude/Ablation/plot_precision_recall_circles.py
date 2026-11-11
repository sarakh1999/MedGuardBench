#!/usr/bin/env python3
"""Precision (x) vs recall (y) for every backbone and training arm run so far.

Same reading as the operating-points figure: recall of unsafe prescriptions is
on the y-axis. The x-axis is precision of the unsafe verdict instead of FPR.
Each result is an unfilled dashed circle. Circle radius identifies the backbone
(replacing the marker shapes on the operating-points plot); colour identifies
the arm.

One point per (backbone, arm), at the arm's intended prompt:
  - profile-trained arms and zero-shot: true patient profile
  - no-profile-trained arms (*-NP): non-personalized prompt
When the uniform ablation harness and an arm-native eval both exist, the
harness number is kept. DPO seed replicates are not separate arms.
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(ROOT, "Claude/Ablation/report/all_results_general.csv")
OUT_PNG = os.path.join(ROOT, "Claude/Ablation/report/fig_precision_recall_circles.png")
OUT_CSV = os.path.join(ROOT, "Claude/Ablation/report/fig_precision_recall_circles.csv")

CANON = {
    "qwen3-4b": "Qwen3-4B", "Qwen3-4B": "Qwen3-4B", "Qwen3-4B-Instruct": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B", "Qwen3-8B": "Qwen3-8B", "Qwen3-8B-Instruct": "Qwen3-8B",
    "qwen3-14b": "Qwen3-14B", "Qwen3-14B": "Qwen3-14B", "Qwen3-14B-Instruct": "Qwen3-14B",
    "qwen3guard-4b": "Qwen3Guard-4B", "Qwen3Guard-4B": "Qwen3Guard-4B", "Qwen3Guard-Gen-4B": "Qwen3Guard-4B",
    "qwen3guard-8b": "Qwen3Guard-8B", "Qwen3Guard-8B": "Qwen3Guard-8B",
    "shieldgemma-2b": "ShieldGemma-2B", "ShieldGemma-2B": "ShieldGemma-2B",
    "shieldgemma-9b": "ShieldGemma-9B", "ShieldGemma-9B": "ShieldGemma-9B",
    "llamaguard2-8b": "Llama-Guard-2-8B", "Llama-Guard-2-8B": "Llama-Guard-2-8B",
    "llamaguard-7b": "LlamaGuard-7B", "LlamaGuard-7b": "LlamaGuard-7B",
    "llama31-8b": "Llama-3.1-8B", "Llama-3.1-8B": "Llama-3.1-8B",
    "wildguard-7b": "WildGuard-7B", "WildGuard-7B": "WildGuard-7B",
    "medgemma-4b": "MedGemma-4B", "MedGemma-4B-it": "MedGemma-4B",
    "llama3.3-70b": "Llama-3.3-70B",
}

# Radius in points. Ordered by parameter count; same-size backbones still
# get distinct radii so every backbone is identifiable.
MODEL_RADIUS = {
    "ShieldGemma-2B": 4.2,
    "MedGemma-4B": 5.2,
    "Qwen3-4B": 6.2,
    "Qwen3Guard-4B": 7.2,
    "LlamaGuard-7B": 8.2,
    "WildGuard-7B": 9.2,
    "Llama-3.1-8B": 10.2,
    "Llama-Guard-2-8B": 11.2,
    "Qwen3-8B": 12.2,
    "Qwen3Guard-8B": 13.2,
    "ShieldGemma-9B": 14.2,
    "Qwen3-14B": 15.6,
    "Llama-3.3-70B": 17.2,
}

ARM_MAP = {
    "base": "zero-shot", "Base": "zero-shot", "base_greedy": "zero-shot",
    "sft": "SFT", "SFT": "SFT", "SFT-reeval": "SFT",
    "SFT-NP": "SFT-NP", "SFT-reeval-NP": "SFT-NP",
    "cu": "CU", "CU": "CU",
    "CU-ctrl-1ep": "CU-ctrl", "CU-given-usr": "CU-given-usr", "CU-long-usr": "CU-long-usr",
    "CU-NP": "CU-NP",
    "grpo": "GRPO", "GRPO": "GRPO",
    "GRPO-NP": "GRPO-NP",
    "GRPO-grpo-guided-v2-noguide": "GRPO-noguide",
    "GRPO-grpo-legacy-cat-lr2e-5": "GRPO-legacy-cat",
    "GRPO-grpo-legacy-lr5e-6": "GRPO-legacy-lr",
    "DPO-partial_miss_audited": "DPO",
    "DPO-counterfactual_single_risk": "DPO-cf",
    "DPO-mixed_balanced": "DPO-mixed",
    "DPO-round2": "DPO-r2",
    "DPO-round3_teacher": "DPO-r3",
    "DPO-selfplay": "DPO-selfplay",
}

ARM_LABEL = {
    "zero-shot": "zero-shot",
    "SFT": "SFT",
    "SFT-NP": "SFT, no profile",
    "CU": "CU",
    "CU-ctrl": "CU control (1 ep)",
    "CU-given-usr": "CU given-context",
    "CU-long-usr": "CU long user",
    "CU-NP": "CU, no profile",
    "GRPO": "guided GRPO",
    "GRPO-v3": "guided GRPO v3",
    "GRPO-noguide": "GRPO, no guide",
    "GRPO-legacy-cat": "GRPO legacy (cat)",
    "GRPO-legacy-lr": "GRPO legacy (lr)",
    "GRPO-NP": "GRPO, no profile",
    "DPO": "DPO partial-miss",
    "DPO-cf": "DPO counterfactual",
    "DPO-mixed": "DPO mixed",
    "DPO-r2": "DPO round 2",
    "DPO-r3": "DPO round 3",
    "DPO-selfplay": "DPO self-play",
}

ARM_COLOR = {
    "zero-shot": "#6e6e6e",
    "SFT": "#1f77b4",
    "SFT-NP": "#6baed6",
    "CU": "#2ca02c",
    "CU-ctrl": "#74c476",
    "CU-given-usr": "#17becf",
    "CU-long-usr": "#bcbd22",
    "CU-NP": "#8c564b",
    "GRPO": "#d62728",
    "GRPO-v3": "#e377c2",
    "GRPO-noguide": "#fb6a4a",
    "GRPO-legacy-cat": "#a6761d",
    "GRPO-legacy-lr": "#f768a1",
    "GRPO-NP": "#ff7f0e",
    "DPO": "#9467bd",
    "DPO-cf": "#c5b0d5",
    "DPO-mixed": "#6a3d9a",
    "DPO-r2": "#8c6bb1",
    "DPO-r3": "#b15928",
    "DPO-selfplay": "#e7298a",
}

SOURCE_PRIO = {
    "ablation-harness": 0,
    "sft-native": 1,
    "cu-native": 2,
    "grpo-native": 3,
    "dpo-native": 4,
    "base-native": 5,
}


def intended_condition(arm):
    return "nonpersonalized" if arm.endswith("-NP") else "true"


def skip_model(raw):
    return ("seed" in raw) or ("backup" in raw) or ("broken" in raw)


def add_dashed_circle(ax, x, y, radius_pt, color, zorder):
    # dpi_scale_trans takes inches. A point is 1/72 inch.
    circ = Circle(
        (0, 0), radius_pt / 72.0, fill=False, linestyle=(0, (3.2, 1.6)),
        edgecolor=color, linewidth=1.15, zorder=zorder,
    )
    circ.set_transform(
        ax.figure.dpi_scale_trans + mtransforms.ScaledTranslation(x, y, ax.transData)
    )
    circ.set_clip_path(ax.patch)
    ax.add_patch(circ)


def load_points():
    rows = list(csv.DictReader(open(CSV_PATH)))
    # Selected ShieldGemma-2B GRPO checkpoint lives outside the collector glob.
    v3 = os.path.join(
        ROOT,
        "Claude/GRPO/outputs/legacy_v1-shieldgemma-2b/grpo-guided-v3/final_selected/eval/eval_metrics.json",
    )
    if os.path.exists(v3):
        g = json.load(open(v3))["grpo"]
        rows.append({
            "model": "shieldgemma-2b", "arm": "GRPO-v3", "condition": "true",
            "source": "grpo-native", "n": g["n"],
            "precision_unsafe": g["precision_unsafe"], "recall_unsafe": g["recall_unsafe"],
            "f1_unsafe": g["f1_unsafe"], "fpr": g["fpr"],
            "file": os.path.relpath(v3, ROOT),
        })
        ARM_MAP["GRPO-v3"] = "GRPO-v3"

    best = {}
    unknown = set()
    for r in rows:
        raw = r["model"]
        if skip_model(raw) or raw not in CANON:
            continue
        arm = ARM_MAP.get(r["arm"])
        if arm is None:
            unknown.add(r["arm"])
            continue
        if r["condition"] != intended_condition(arm):
            continue
        prec = float(r["precision_unsafe"])
        rec = float(r["recall_unsafe"])
        item = {
            "model": CANON[raw], "arm": arm,
            "precision": prec, "recall": rec,
            "f1": float(r["f1_unsafe"]) if r.get("f1_unsafe") not in (None, "") else 2 * prec * rec / max(prec + rec, 1e-9),
            "fpr": float(r["fpr"]) if r.get("fpr") not in (None, "") else None,
            "source": r["source"], "n": int(float(r["n"])), "file": r["file"],
        }
        key = (item["model"], item["arm"])
        prev = best.get(key)
        if prev is None or SOURCE_PRIO[item["source"]] < SOURCE_PRIO[prev["source"]]:
            best[key] = item
    if unknown:
        print("unmapped arms (not plotted):", sorted(unknown))
    return list(best.values())


def main():
    points = load_points()
    points.sort(key=lambda p: (-p["f1"], p["model"], p["arm"]))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "arm", "precision", "recall", "f1", "fpr", "source", "n", "file"])
        w.writeheader()
        for p in points:
            w.writerow({k: (f"{p[k]:.4f}" if k in ("precision", "recall", "f1", "fpr") and p[k] is not None else p[k]) for k in w.fieldnames})

    print(f"{'backbone':22} {'arm':22} {'P':>6} {'R':>6} {'F1':>6}  source")
    for p in points:
        print(f"{p['model']:22} {ARM_LABEL[p['arm']]:22} {100*p['precision']:6.1f} {100*p['recall']:6.1f} {100*p['f1']:6.1f}  {p['source']}")
    print(f"\n{len(points)} points, {len({p['model'] for p in points})} backbones, {len({p['arm'] for p in points})} arms")

    # Keep the axes on the same window as the operating-points figure.
    # Anything below 60% recall is drawn as a callout so it is not cropped silently.
    outliers = [p for p in points if 100 * p["recall"] < 60 or 100 * p["precision"] < 55]
    shown = [p for p in points if p not in outliers]

    fig, ax = plt.subplots(figsize=(11.6, 7.6))
    for p in sorted(shown, key=lambda q: -MODEL_RADIUS[q["model"]]):
        add_dashed_circle(
            ax, 100 * p["precision"], 100 * p["recall"], MODEL_RADIUS[p["model"]],
            ARM_COLOR[p["arm"]], zorder=3,
        )
    ax.scatter(
        [100 * p["precision"] for p in shown],
        [100 * p["recall"] for p in shown],
        s=11, c=[ARM_COLOR[p["arm"]] for p in shown], zorder=5, linewidths=0,
    )

    ax.axhline(95, color="grey", ls=":", lw=0.8, zorder=1)
    ax.text(56.2, 95.6, "95% recall", fontsize=8, color="grey")
    ax.set_xlabel("precision of unsafe verdicts (%)")
    ax.set_ylabel("recall on unsafe prescriptions (%)")
    ax.set_xlim(55, 101)
    ax.set_ylim(60, 101.5)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(
        "Precision vs recall, every backbone and arm (n=754)\n"
        "dashed circle radius = backbone, colour = arm; top-right is better",
        fontsize=11,
    )

    if outliers:
        lines = ["below this window:"]
        for p in sorted(outliers, key=lambda q: q["recall"]):
            lines.append(f"{p['model']} {ARM_LABEL[p['arm']]}:  P={100*p['precision']:.1f}  R={100*p['recall']:.1f}")
        ax.text(
            0.01, 0.01, "\n".join(lines), transform=ax.transAxes, fontsize=7.5,
            va="bottom", ha="left", color="#333333",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.92),
        )

    arm_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=7,
               markerfacecolor="none", markeredgecolor=ARM_COLOR[a], markeredgewidth=1.4,
               label=ARM_LABEL[a])
        for a in ARM_LABEL if any(p["arm"] == a for p in points)
    ]
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=2 * MODEL_RADIUS[m],
               markerfacecolor="none", markeredgecolor="black", markeredgewidth=0.9,
               label=m)
        for m in MODEL_RADIUS if any(p["model"] == m for p in points)
    ]
    leg1 = ax.legend(
        handles=arm_handles, title="arm (colour)", fontsize=7.2, title_fontsize=8,
        loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=False,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=model_handles, title="backbone (radius)", fontsize=7.2, title_fontsize=8,
        loc="upper left", bbox_to_anchor=(1.01, 0.42), frameon=False,
        labelspacing=1.15, handletextpad=1.4, borderpad=0.4,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
