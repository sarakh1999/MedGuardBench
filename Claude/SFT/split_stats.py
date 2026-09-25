"""Cross-split comparison and per-risk-category safe/unsafe counts for each split.

`Is_Safe` is exactly the negation of "any risk category flagged" -- verified here on every
row, and asserted, so a category-vs-Is_Safe cross-tab would be degenerate: every positive
of every category sits in an unsafe sample. The meaningful per-category breakdown is
therefore the binary class balance the multi-label model actually trains on: how many
samples have that risk flagged (unsafe on that axis) against how many do not (safe on
that axis).

Prevalence drift between splits is unavoidable under a drug-disjoint partition, since a
category's positives are concentrated in particular drugs. Each category gets a
chi-square test of independence between split and flag; with 3 splits and 2 outcomes
df = 2, for which the p-value is exactly exp(-chi2 / 2), so no scipy is needed.

  outputs -> data_splits/risk_balance_by_split.csv
             data_splits/SPLIT_BALANCE.md
             plots/split_safe_unsafe.png
             plots/split_prevalence_compare.png
"""
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLIT_DIR = "Claude/SFT/new_data_splits1"
MEDS_FILE = "new_medications.txt"
RISKS_FILE = "risk_categories.txt"
PLOT_DIR = "plots"
MED_COL = "Recommended Medication"
SPLITS = ["train", "val", "test"]
TARGET = {"train": 75.0, "val": 10.0, "test": 15.0}
COLORS = {"train": "#2b6cb0", "val": "#c05621", "test": "#6b46c1"}
RED, GREY = "#c53030", "#e2e8f0"

SHORT = {
    "Allergy & Adverse Drug Reaction Risk": "Allergy/ADR",
    "Drug-Drug Interaction Risk": "Drug-Drug",
    "Drug-Food Interaction Risk": "Drug-Food",
    "Dosage & Toxicity Risk": "Dosage/Toxicity",
    "Renal Impairment Risk": "Renal",
    "Hepatic Impairment Risk": "Hepatic",
    "Cardiac Impairment Risk": "Cardiac",
    "Respiratory Impairment Risk": "Respiratory",
    "Bleeding Risk": "Bleeding",
    "Infection Risk": "Infection",
    "Pregnancy & Breastfeeding Risk": "Pregnancy",
    "Alcohol Use Risk": "Alcohol",
    "Tobacco Use Risk": "Tobacco",
    "Substance Use Risk": "Substance",
    "Caffeine Intake Risk": "Caffeine",
    "Weight/BMI Risk": "Weight/BMI",
    "Age Risk": "Age",
}

# ---------------------------------------------------------------- load
read_list = lambda p: [ln.strip() for ln in open(p) if ln.strip()]
DRUGS, RISKS = read_list(MEDS_FILE), read_list(RISKS_FILE)
PAT = {d: re.compile(r"(?<![A-Za-z])" + r"\s+".join(re.escape(p) for p in d.split())
                     + r"(?![A-Za-z])", re.I) for d in DRUGS}

data, Y, unsafe, drugs, oov = {}, {}, {}, {}, {}
for s in SPLITS:
    d = pd.read_csv(os.path.join(SPLIT_DIR, f"{s}.csv"), dtype=str, keep_default_na=False)
    data[s] = d
    Y[s] = np.array([[bool(json.loads(x).get(r, False)) for r in RISKS]
                     for x in d["Risk_Categories"]])
    unsafe[s] = np.array([v.strip().upper() != "TRUE" for v in d["Is_Safe"]])
    assert (unsafe[s] == Y[s].any(1)).all(), f"{s}: Is_Safe is not the negation of any-risk"
    hits = [[k for k, p in PAT.items() if p.search(v)] for v in d[MED_COL]]
    drugs[s] = sorted({k for h in hits for k in h})
    oov[s] = sum(1 for h in hits if not h)

n = {s: len(data[s]) for s in SPLITS}
N = sum(n.values())
POS = {s: Y[s].sum(0) for s in SPLITS}                      # positives per category
TOT = sum(POS[s] for s in SPLITS)
order = sorted(range(len(RISKS)), key=lambda j: -TOT[j])    # most frequent first

# ---------------------------------------------------------------- A. cross-split
out = []
p = out.append
p("=" * 104)
p("A.  CROSS-SPLIT COMPARISON")
p("=" * 104)
p(f"{'split':<7s} {'rows':>6s} {'share':>8s} {'target':>7s} {'delta':>7s} {'drugs':>6s} "
  f"{'no-vocab':>9s} {'safe':>7s} {'unsafe':>7s} {'unsafe%':>8s}")
for s in SPLITS:
    p(f"{s:<7s} {n[s]:6d} {n[s]/N*100:7.2f}% {TARGET[s]:6.1f}% "
      f"{n[s]/N*100-TARGET[s]:+6.2f}% {len(drugs[s]):6d} {oov[s]:9d} "
      f"{int((~unsafe[s]).sum()):7d} {int(unsafe[s].sum()):7d} {unsafe[s].mean()*100:7.2f}%")
tot_unsafe = sum(int(unsafe[s].sum()) for s in SPLITS)
p(f"{'TOTAL':<7s} {N:6d} {'100.00%':>8s} {'100.0%':>7s} {'':>7s} "
  f"{sum(len(drugs[s]) for s in SPLITS):6d} {sum(oov.values()):9d} "
  f"{N-tot_unsafe:7d} {tot_unsafe:7d} {tot_unsafe/N*100:7.2f}%")

overlap = {(a, b): sorted(set(drugs[a]) & set(drugs[b]))
           for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
p("")
for (a, b), v in overlap.items():
    p(f"drug overlap {a:<5s} n {b:<5s}: {len(v)} {v if v else ''}")
assert not any(overlap.values()), "splits are NOT drug-disjoint"
p(f"unsafe-rate spread across splits: "
  f"{max(unsafe[s].mean() for s in SPLITS)*100 - min(unsafe[s].mean() for s in SPLITS)*100:.2f} "
  f"percentage points")

# ---------------------------------------------------------------- B. per split
p("\n" + "=" * 104)
p("B.  SAFE / UNSAFE PER RISK CATEGORY, PER SPLIT")
p("=" * 104)
p(f"{'':<18s}" + "".join(f"{s.upper():^28s}" for s in SPLITS))
p(f"{'risk category':<18s}" + "".join(f"{'unsafe':>8s}{'safe':>9s}{'rate':>11s}"
                                      for _ in SPLITS) + f"{'drift':>8s}{'chi2 p':>9s}")
p("-" * 104)
rows, drift_of, pval_of = [], {}, {}
for j in order:
    r = RISKS[j]
    rate = {s: POS[s][j] / n[s] for s in SPLITS}
    drift = (max(rate.values()) - min(rate.values())) * 100
    # chi-square of independence between split and flag; df=2 -> p = exp(-chi2/2)
    obs = np.array([[POS[s][j], n[s] - POS[s][j]] for s in SPLITS], dtype=float)
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    chi2 = float(((obs - exp) ** 2 / np.maximum(exp, 1e-12)).sum())
    pv = math.exp(-chi2 / 2)
    drift_of[j], pval_of[j] = drift, pv
    line = f"{SHORT[r]:<18s}"
    for s in SPLITS:
        line += f"{int(POS[s][j]):>8d}{int(n[s]-POS[s][j]):>9d}{rate[s]*100:>10.2f}%"
    star = "*" if pv < 0.05 else " "
    p(line + f"{drift:>7.2f}%{pv:>8.3f}{star}")
    for s in SPLITS:
        rows.append({"risk_category": r, "split": s, "split_n": n[s],
                     "unsafe_flagged": int(POS[s][j]),
                     "safe_not_flagged": int(n[s] - POS[s][j]),
                     "positive_rate_pct": round(POS[s][j] / n[s] * 100, 3),
                     "drift_pct": round(drift, 3), "chi2_p": round(pv, 4)})
p("-" * 104)
line = f"{'ANY (Is_Safe)':<18s}"
for s in SPLITS:
    line += f"{int(unsafe[s].sum()):>8d}{int((~unsafe[s]).sum()):>9d}{unsafe[s].mean()*100:>10.2f}%"
p(line + f"{max(unsafe[s].mean() for s in SPLITS)*100 - min(unsafe[s].mean() for s in SPLITS)*100:>7.2f}%")
p("\n'unsafe' = risk flagged for that sample, 'safe' = not flagged; the two sum to the "
  "split size.\n'drift' = max - min positive rate across splits.  * = prevalence differs "
  "across splits at p<0.05.")

# ---------------------------------------------------------------- C. warnings
thin = [j for j in order if min(POS['val'][j], POS['test'][j]) < 10]
p("\n" + "=" * 104)
p("C.  CATEGORIES TOO THIN FOR RELIABLE PER-CLASS EVALUATION")
p("=" * 104)
for j in thin:
    p(f"  {RISKS[j]:<40s} val {int(POS['val'][j]):3d} +   test {int(POS['test'][j]):3d} +   "
      f"total {int(TOT[j]):4d} +")
p(f"\n{len(thin)} of {len(RISKS)} categories have fewer than 10 positives in val or test. "
  "Macro-F1 would\nlet each of them swing the headline number as much as Drug-Drug "
  "Interaction Risk\n(1,244 positives), which is why micro-F1 is the primary metric.")
p(f"\nlargest prevalence drift: "
  + ", ".join(f"{SHORT[RISKS[j]]} {drift_of[j]:.2f}pp"
              for j in sorted(order, key=lambda k: -drift_of[k])[:4]))

report = "\n".join(out)
print(report)

# ---------------------------------------------------------------- save tables
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
tbl = pd.DataFrame(rows)
tbl.to_csv(os.path.join(SPLIT_DIR, "risk_balance_by_split.csv"), index=False)

M = ["# Cross-split comparison and per-category class balance\n", "```", report, "```\n",
     "## Safe / unsafe per risk category and split\n",
     "| Risk category | " + " | ".join(f"{s} unsafe | {s} safe | {s} rate" for s in SPLITS)
     + " | drift | chi2 p |",
     "|---" * (2 + 3 * len(SPLITS)) + "|"]
for j in order:
    cells = []
    for s in SPLITS:
        cells += [f"{int(POS[s][j]):,}", f"{int(n[s]-POS[s][j]):,}",
                  f"{POS[s][j]/n[s]*100:.2f}%"]
    M.append(f"| {RISKS[j]} | " + " | ".join(cells)
             + f" | {drift_of[j]:.2f}pp | {pval_of[j]:.3f} |")
M.append("| **Overall (Is_Safe)** | "
         + " | ".join(f"**{int(unsafe[s].sum()):,}** | **{int((~unsafe[s]).sum()):,}** | "
                      f"**{unsafe[s].mean()*100:.2f}%**" for s in SPLITS) + " | | |\n")
open(os.path.join(SPLIT_DIR, "SPLIT_BALANCE.md"), "w").write("\n".join(M))

# ---------------------------------------------------------------- plot 1
short = [SHORT[RISKS[j]] for j in order]
fig, axes = plt.subplots(1, 3, figsize=(16, 6.6), sharey=True)
yy = np.arange(len(order))
for ax, s in zip(axes, SPLITS):
    pos = np.array([POS[s][j] for j in order])
    ax.barh(yy, n[s] - pos, color=GREY, height=.68, label="safe (not flagged)")
    ax.barh(yy, pos, color=COLORS[s], height=.68, label="unsafe (flagged)")
    for i, v in enumerate(pos):
        ax.text(v + n[s] * .012, i, f"{v}", va="center", fontsize=7, color="#2d3748")
    ax.set_yticks(yy)
    ax.set_yticklabels(short, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("samples")
    ax.set_xlim(0, n[s] * 1.03)
    ax.set_title(f"{s} — {n[s]:,} samples, {len(drugs[s])} drugs", fontweight="bold")
    ax.grid(axis="x", color="#eef2f6", lw=.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
fig.suptitle("Safe vs unsafe samples per risk category, per split",
             fontsize=13, fontweight="bold")
fig.savefig(os.path.join(PLOT_DIR, "split_safe_unsafe.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- plot 2
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6.2),
                              gridspec_kw={"width_ratios": [1.45, 1]})
x = np.arange(len(order))
w = .27
for k, s in enumerate(SPLITS):
    ax.bar(x + (k - 1) * w, [POS[s][j] / n[s] * 100 for j in order], w,
           color=COLORS[s], label=s)
ax.set_xticks(x)
ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("positive rate (% of split)")
ax.set_title("Prevalence of each risk category within each split", fontweight="bold")
ax.legend(fontsize=8, frameon=False)
ax.grid(axis="y", color="#eef2f6", lw=.6)
ax.set_axisbelow(True)
ax.text(.98, .96, "equal-height triplets = category is\nbalanced across the split",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=.35", fc="#f7fafc", ec="#cbd5e0", lw=.6))

C = np.array([[POS[s][j] for s in SPLITS] for j in order], dtype=float)
# colour by prevalence, not raw count: train is 7.5x val, so counts alone would wash out
# the two small splits and hide exactly the cross-split drift this panel is meant to show
R = C / np.array([n[s] for s in SPLITS], dtype=float) * 100
im = ax2.imshow(R, cmap="Blues", aspect="auto")
ax2.set_xticks(range(3))
ax2.set_xticklabels([f"{s}\n({n[s]:,})" for s in SPLITS], fontsize=8)
ax2.set_yticks(range(len(order)))
ax2.set_yticklabels([f"{t} *" if j in thin else t for t, j in zip(short, order)], fontsize=7.5)
ax2.grid(False)
hi = R.max()
for i in range(len(order)):
    for k in range(3):
        ax2.text(k, i, f"{int(C[i,k])}\n{R[i,k]:.1f}%", ha="center", va="center",
                 fontsize=6.5, linespacing=1.25,
                 color="white" if R[i, k] / hi > .62 else "#1a202c")
for lbl, j in zip(ax2.get_yticklabels(), order):
    if j in thin:
        lbl.set_color(RED)
ax2.set_title("Unsafe (flagged) count and rate per category", fontweight="bold")
fig.colorbar(im, ax=ax2, fraction=.03, pad=.02).set_label(
    "positive rate (% of split)", fontsize=8)
fig.suptitle("Cross-split risk-category comparison  "
             "(* = fewer than 10 positives in val or test)",
             fontsize=13, fontweight="bold")
fig.savefig(os.path.join(PLOT_DIR, "split_prevalence_compare.png"), dpi=130,
            bbox_inches="tight")
plt.close(fig)

print(f"\nwrote {SPLIT_DIR}/risk_balance_by_split.csv  ({len(tbl)} rows)")
print(f"wrote {SPLIT_DIR}/SPLIT_BALANCE.md")
print(f"wrote {PLOT_DIR}/split_safe_unsafe.png")
print(f"wrote {PLOT_DIR}/split_prevalence_compare.png")

