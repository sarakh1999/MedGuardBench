#!/usr/bin/env python3
"""Redraw the two paper radar figures from the numbers printed in the paper's tables.

  personalization_effect_radar.pdf  test-time profile perturbation (Table 5 cells,
                                    report/recompute_729/paper/table5_all_729.json)
  radar.pdf                         per-category F1 of SFT / CU / GRPO (Table 3,
                                    read from the paper's per_category_F1_Table.tex)

Both are sized for one ACL column (3.03in) so fonts print at their nominal size.
"""
import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T5 = os.path.join(HERE, "report/recompute_729/paper/table5_all_729.json")
T3 = os.path.expanduser("~/acl-paper/ARR/Results/per_category_F1_Table.tex")

P_MODELS = [("qwen3-4b", "Qwen3-4B"), ("qwen3-8b", "Qwen3-8B"),
            ("qwen3guard-4b", "Qwen3Guard-4B"), ("qwen3guard-8b", "Qwen3Guard-8B")]
P_ARMS = [("base", "Zero-Shot"), ("sft", "Pers. SFT"), ("cu", "Pers. CU"), ("grpo", "Pers. GRPO")]
P_CONDS = [("true", "True Profile", "#4E79A7"), ("noprofile", "Blank Profile", "#76B7B2"),
           ("shuffled", "Shuffled Profile", "#E8879C"), ("nonpersonalized", "No Patient Context", "#F2C14E")]
P_AXES = [("acc", "Acc"), ("rec", "Rec"), ("fpr", "100$-$FPR"), ("mic", "Mic-F1"), ("mac", "Mac-F1")]

C_MODELS = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3Guard-4B", "Qwen3Guard-8B", "WildGuard-7B",
            "Llama-3.1-8B", "Llama-Guard-7B", "Llama-Guard-2-8B", "ShieldGemma-2B", "ShieldGemma-9B", "MedGemma-4B"]
C_ARMS = [("SFT", "#4E79A7"), ("CU", "#F2C14E"), ("GRPO", "#E15759")]
C_SHORT = {"Drug-Drug Interaction": "DDI", "Cardiac Impairment": "Card", "Dosage & Toxicity": "Dose", "Age": "Age",
           "Allergy & Adverse Drug Reaction": "Alrg", "Hepatic Impairment": "Hep", "Renal Impairment": "Renal",
           "Alcohol Use": "Alc", "Drug-Food Interaction": "DFI", "Pregnancy & Breastfeeding": "Preg",
           "Bleeding": "Bleed", "Infection": "Infec", "Respiratory Impairment": "Resp", "Substance Use": "Subst",
           "Weight/BMI": "Wt", "Tobacco Use": "Tob", "Caffeine Intake": "Caf"}


def polar_grid(ax, labels, fs, rmax=100, label_pad=1.0):
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(0, rmax); ax.set_yticks([25, 50, 75]); ax.set_yticklabels([])
    ax.set_xticks(ang); ax.set_xticklabels([])
    for a, lab in zip(ang, labels):
        ax.text(a, rmax * (1.0 + 0.16 * label_pad), lab, fontsize=fs, ha="center", va="center")
    ax.grid(color="#cccccc", lw=0.35); ax.spines["polar"].set_linewidth(0.4); ax.spines["polar"].set_color("#999999")
    return ang


def closed(ang, vals):
    return np.r_[ang, ang[:1]], np.r_[vals, vals[:1]]


def perturbation_radar(out, plt):
    t5 = json.load(open(T5))
    fig, axes = plt.subplots(len(P_MODELS), len(P_ARMS), figsize=(3.35, 3.75), subplot_kw=dict(polar=True))
    for i, (mk, mlab) in enumerate(P_MODELS):
        for j, (ak, alab) in enumerate(P_ARMS):
            ax = axes[i, j]
            ang = polar_grid(ax, [a[1] for a in P_AXES], fs=5.0, label_pad=1.7)
            for ck, _, col in P_CONDS:
                c = t5[f"{mk}|{ak}|{ck}"]
                v = np.array([100 - c[k] if k == "fpr" else c[k] for k, _ in P_AXES], dtype=float)
                x, y = closed(ang, v)
                lw = 0.9 if ck == "true" else 0.7
                ax.plot(x, y, color=col, lw=lw, zorder=3 if ck == "true" else 2)
                if ck == "true":
                    ax.fill(x, y, color=col, alpha=0.10)
            if i == 0:
                ax.set_title(alab, fontsize=6.3, fontweight="bold", pad=9)
            if j == 0:
                ax.text(-0.62, 0.5, mlab, transform=ax.transAxes, rotation=90, fontsize=6.3,
                        fontweight="bold", ha="center", va="center")
    handles = [plt.Line2D([], [], color=col, lw=1.2, label=lab) for _, lab, col in P_CONDS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=5.6, frameon=False,
               bbox_to_anchor=(0.53, -0.005), handlelength=1.4, columnspacing=0.9, handletextpad=0.4)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.07, wspace=0.85, hspace=0.62)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("wrote", out)


def read_table3():
    tex = open(T3).read()
    blocks = re.split(r"\\begin\{tabular\}", tex)[1:]
    res = {}
    for b in blocks:
        models = re.findall(r"\\multicolumn\{4\}\{c\|?\}\{\\textbf\{([^}]*)\}\}", b)
        for line in b.splitlines():
            cells = [c.strip() for c in re.split(r"(?<!\\)&", line.rstrip("\\ "))]
            name = cells[0].replace("\\&", "&").replace("Allergy & ADR", "Allergy & Adverse Drug Reaction")
            if name not in C_SHORT:
                continue
            vals = [float(re.sub(r"\\textbf\{([^}]*)\}", r"\1", c)) for c in cells[1:]]
            assert len(vals) == 4 * len(models), line
            for k, m in enumerate(models):
                base, sft, cu, grpo = vals[4 * k:4 * k + 4]
                res.setdefault(m, {})[name] = dict(SFT=sft, CU=cu, GRPO=grpo)
    return res


def category_radar(out, plt):
    t3 = read_table3()
    cats = list(C_SHORT)
    fig, axes = plt.subplots(3, 4, figsize=(3.35, 2.95), subplot_kw=dict(polar=True))
    for ax, m in zip(axes.flat, C_MODELS):
        ang = polar_grid(ax, [C_SHORT[c] for c in cats], fs=3.9, label_pad=1.0)
        for arm, col in C_ARMS:
            x, y = closed(ang, np.array([t3[m][c][arm] for c in cats]))
            ax.plot(x, y, color=col, lw=0.65)
            ax.fill(x, y, color=col, alpha=0.06)
        ax.set_title(m, fontsize=5.6, fontweight="bold", pad=8.5)
    handles = [plt.Line2D([], [], color=col, lw=1.2, label=arm) for arm, col in C_ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=5.8, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.6)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.07, wspace=0.55, hspace=0.62)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "report"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    perturbation_radar(os.path.join(args.out_dir, "personalization_effect_radar.pdf"), plt)
    category_radar(os.path.join(args.out_dir, "radar.pdf"), plt)


if __name__ == "__main__":
    main()
