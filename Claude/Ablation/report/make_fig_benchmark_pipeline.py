"""Figure: benchmark construction and use, in one panel.

drug monographs -> synthetic patient profiles (20 fields) -> 17 risk categories
-> safe/unsafe verdict -> drug-disjoint train/val/test -> personalized guardrail
training / evaluation.

Counts are read from the repo so they stay in sync with the released splits.
Writes fig_benchmark_pipeline.{pdf,png} next to this script.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SPLIT_DIR = os.path.join(ROOT, "Claude", "SFT", "new_data_splits")

# ------------------------------------------------------------------ data counts
def _count(split):
    import pandas as pd
    df = pd.read_csv(os.path.join(SPLIT_DIR, f"{split}.csv"))
    return len(df), int((~df["Is_Safe"].astype(bool)).sum())

try:
    N = {s: _count(s) for s in ("train", "val", "test")}
    N_TOTAL = sum(n for n, _ in N.values())
    N_UNSAFE = sum(u for _, u in N.values())
    with open(os.path.join(ROOT, "new_medications.txt")) as f:
        N_DRUGS = len([l for l in f if l.strip()])
except Exception:  # fall back to the published numbers
    N = {"train": (3772, 2020), "val": (503, 263), "test": (754, 403)}
    N_TOTAL, N_UNSAFE, N_DRUGS = 5029, 2686, 85

# ------------------------------------------------------------------ styling
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
C_BG = "#F5F7FA"
C_EDGE = "#2F3B4C"
C_MONO = "#8D99AE"
C_PROF = "#4C72B0"
C_RISK = "#DD8452"
C_SAFE = "#55A868"
C_UNSAFE = "#C44E52"
C_SPLIT = ("#4C72B0", "#8172B3", "#CCB974")
C_MODEL = "#2F3B4C"
C_ARROW = "#55606E"
C_TEXT = "#1F2733"
C_SUB = "#55606E"

# 20 profile fields (abbreviated to fit; order follows the dataset columns)
FIELDS = [
    "Age", "Gender", "Weight", "Height", "BMI",
    "Genetic dis.", "Chronic cond.", "Pregnancy/BF",
    "Drug allergies", "Renal impair.", "Hepatic impair.",
    "Cardiac impair.", "Resp. impair.", "Alcohol use",
    "Tobacco use", "Substance use", "Caffeine",
    "Current meds", "Foods (24 h)", "Symptoms",
]
assert len(FIELDS) == 20

# 17 categories grouped so the grid reads as a taxonomy, not a list
RISK_GROUPS = [
    ("Drug", ["Allergy / ADR", "Drug–drug", "Drug–food", "Dose/toxicity"]),
    ("Organ", ["Renal", "Hepatic", "Cardiac", "Respiratory", "Bleeding", "Infection"]),
    ("Lifestyle", ["Alcohol", "Tobacco", "Substance", "Caffeine"]),
    ("Population", ["Pregnancy/BF", "Weight / BMI", "Age"]),
]
assert sum(len(c) for _, c in RISK_GROUPS) == 17
# running example (Patient 1 of the benchmark)
EX_FLAGGED = {"Drug–drug", "Bleeding"}
EX_HIGHLIGHT_FIELDS = {"Age", "Cardiac impair.", "Current meds", "Renal impair."}


def box(ax, x, y, w, h, fc="white", ec=C_EDGE, lw=0.8, r=0.08, z=1, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color=C_ARROW, lw=1.2, style="-|>", rad=0.0, z=3):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9, color=color,
                        lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=z,
                        shrinkA=1.5, shrinkB=1.5)
    ax.add_patch(a)


def stage_label(ax, x, y, n, text):
    ax.text(x, y, f"{n}", ha="center", va="center", fontsize=7, fontweight="bold",
            color="white", zorder=5,
            bbox=dict(boxstyle="circle,pad=0.25", fc=C_EDGE, ec="none"))
    ax.text(x + 0.24, y, text, ha="left", va="center", fontsize=7.5,
            fontweight="bold", color=C_TEXT, zorder=5)


def caption(ax, xc, y_bold, bold, sub):
    ax.text(xc, y_bold, bold, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=C_TEXT)
    ax.text(xc, y_bold - 0.42, sub, ha="center", va="center", fontsize=5.6,
            color=C_SUB, linespacing=1.25)


# ------------------------------------------------------------------ canvas
W, H = 17.8, 8.7
fig = plt.figure(figsize=(7.2, 7.2 * H / W), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# ======================================================= ROW A: construction
yA, hA = 4.25, 3.85       # content band of row A
ax.text(0.25, 8.4, "A   Benchmark construction", fontsize=9, fontweight="bold",
        color=C_TEXT, va="center")
box(ax, 0.2, yA - 0.15, W - 0.4, hA + 0.25, fc=C_BG, ec="#D7DDE5", lw=0.6, r=0.15, z=0)

band_y, band_h = yA + 1.0, 2.0          # y-range of the main graphics in row A
cap_y = yA + 0.6                        # y of the bold caption under graphics

# --- (1) drug monographs ------------------------------------------------------
x1, w1 = 0.45, 2.6
stage_label(ax, x1 + 0.15, yA + hA - 0.2, 1, "Drug monographs")
for i, off in enumerate((0.26, 0.13, 0.0)):
    dx, dy = x1 + 0.42 + off, band_y + 0.15 - off
    box(ax, dx, dy, 1.3, 1.6, fc="white" if i == 2 else "#E9EDF2", ec=C_MONO,
        lw=0.8, r=0.04, z=2 + i)
dx, dy = x1 + 0.42, band_y + 0.15
for k, ww in enumerate((0.9, 0.75, 0.95, 0.6, 0.85, 0.7)):
    yy = dy + 1.38 - 0.2 * k
    ax.add_patch(Rectangle((dx + 0.15, yy), ww, 0.06,
                           fc=C_MONO if k else C_EDGE, ec="none", zorder=6))
caption(ax, x1 + w1 / 2 + 0.1, cap_y, f"{N_DRUGS} drugs",
        "indications · dosing\ncontraindications · interactions")

# --- (2) synthetic patient profile --------------------------------------------
x2, w2 = 3.55, 5.2
stage_label(ax, x2 + 0.15, yA + hA - 0.2, 2, "Synthetic patient profile + scenario")
gx, gy, gw, gh = x2 + 0.1, band_y, 2.8, band_h
box(ax, gx, gy, gw, gh, fc="white", ec=C_PROF, lw=0.9, r=0.06, z=2)
ax.text(gx + gw / 2, gy + gh + 0.09, "20 profile fields", ha="center", va="bottom",
        fontsize=6.8, fontweight="bold", color=C_PROF)
cw, rh = gw / 2, gh / 10
for i, f in enumerate(FIELDS):
    col, row = divmod(i, 10)
    fx = gx + col * cw
    fy = gy + gh - (row + 1) * rh
    hl = f in EX_HIGHLIGHT_FIELDS
    if hl:
        ax.add_patch(Rectangle((fx + 0.03, fy + 0.015), cw - 0.06, rh - 0.03,
                               fc="#DCE6F5", ec="none", zorder=3))
    ax.text(fx + 0.09, fy + rh / 2, f, ha="left", va="center", fontsize=5,
            color=C_TEXT if hl else "#4A5563", fontweight="bold" if hl else "normal",
            zorder=4)
# example patient card
cx, cy, cwid, chei = gx + gw + 0.15, band_y, w2 - gw - 0.35, band_h
box(ax, cx, cy, cwid, chei, fc="#FFFDF7", ec="#C9A227", lw=0.8, r=0.06, z=2, ls="--")
ax.text(cx + cwid / 2, cy + chei + 0.09, "example patient", ha="center", va="bottom",
        fontsize=6.3, color="#8A6D00", style="italic")
ex_lines = [
    ("72 y, male, BMI 27", C_TEXT, False),
    ("AF, rate-controlled", C_TEXT, False),
    ("CrCl 70 mL/min", C_TEXT, False),
    ("on warfarin, aspirin", C_TEXT, False),
    ("325 mg, metoprolol", C_TEXT, False),
    ("Rx: warfarin 5 mg", C_PROF, True),
    ("PO daily, chronic", C_PROF, True),
    ("“Is this safe?”", C_PROF, True),
]
for k, (t, c, b) in enumerate(ex_lines):
    yy = cy + chei - 0.2 - 0.215 * k - (0.15 if b else 0)
    ax.text(cx + 0.1, yy, t, ha="left", va="center", fontsize=5.2, color=c,
            fontweight="bold" if b else "normal")
ax.plot([cx + 0.1, cx + cwid - 0.1], [cy + chei - 0.2 - 0.215 * 5 + 0.07] * 2,
        color="#C9A227", lw=0.5, ls=":")
caption(ax, x2 + w2 / 2, cap_y, f"{N_TOTAL:,} patients",
        "LLM-generated from each monograph\nnear-duplicate profiles dropped")

# --- (3) 17 risk categories ----------------------------------------------------
x3, w3 = 9.15, 5.6
stage_label(ax, x3 + 0.15, yA + hA - 0.2, 3, "17 risk categories (multi-label)")
rx, ry = x3 + 0.09, band_y
chip_h, chip_w, gap = 0.31, 1.28, 0.055
col_x = rx
for gname, cats in RISK_GROUPS:
    ax.text(col_x + chip_w / 2, ry + band_h + 0.09, gname, ha="center", va="bottom",
            fontsize=6, fontweight="bold", color=C_RISK)
    for j, c in enumerate(cats):
        yy = ry + band_h - (j + 1) * (chip_h + gap) + gap
        flagged = c in EX_FLAGGED
        box(ax, col_x, yy, chip_w, chip_h, fc=C_RISK if flagged else "white",
            ec=C_RISK, lw=0.7, r=0.05, z=2)
        ax.text(col_x + chip_w / 2, yy + chip_h / 2, c, ha="center", va="center",
                fontsize=5, color="white" if flagged else C_TEXT,
                fontweight="bold" if flagged else "normal", zorder=4)
    col_x += chip_w + 0.1
caption(ax, x3 + w3 / 2, cap_y, "patient-specific risk analysis",
        "each category judged against profile + monograph\n~32% of patients carry more than one flag")

# --- (4) verdict --------------------------------------------------------------
x4, w4 = 15.05, 2.4
stage_label(ax, x4 + 0.15, yA + hA - 0.2, 4, "Verdict")
vx, vy, vw, vh = x4 + 0.2, band_y, w4 - 0.4, band_h
box(ax, vx, vy + vh / 2 + 0.05, vw, vh / 2 - 0.05, fc="#EAF4EC", ec=C_SAFE, lw=0.9, r=0.06, z=2)
ax.text(vx + vw / 2, vy + vh * 0.75 + 0.1, "SAFE", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=C_SAFE)
ax.text(vx + vw / 2, vy + vh * 0.75 - 0.22, f"{N_TOTAL - N_UNSAFE:,} · no flag",
        ha="center", va="center", fontsize=5.6, color="#3E7A4A")
box(ax, vx, vy, vw, vh / 2 - 0.05, fc="#F8E7E7", ec=C_UNSAFE, lw=1.4, r=0.06, z=2)
ax.text(vx + vw / 2, vy + vh * 0.25 + 0.1, "UNSAFE", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=C_UNSAFE)
ax.text(vx + vw / 2, vy + vh * 0.25 - 0.22, f"{N_UNSAFE:,} · ≥1 flag",
        ha="center", va="center", fontsize=5.6, color="#8E3538")
caption(ax, x4 + w4 / 2, cap_y, "safety verdict",
        "+ rationale\nclinician-reviewed")

# row-A arrows
ym = band_y + band_h / 2
arrow(ax, (x1 + w1 - 0.05, ym), (x2 + 0.05, ym))
arrow(ax, (x2 + w2 - 0.05, ym), (x3 + 0.05, ym))
arrow(ax, (x3 + w3 - 0.05, ym), (x4 + 0.15, ym))

# ======================================================= ROW B: use
yB, hB = 0.35, 3.0
ax.text(0.25, 3.6, "B   Benchmark use", fontsize=9, fontweight="bold",
        color=C_TEXT, va="center")
box(ax, 0.2, yB - 0.15, W - 0.4, hB + 0.2, fc=C_BG, ec="#D7DDE5", lw=0.6, r=0.15, z=0)

# --- (5) drug-disjoint split ---------------------------------------------------
x5, w5 = 0.45, 8.4
stage_label(ax, x5 + 0.15, yB + hB - 0.18, 5, "Drug-disjoint train / val / test split")
total_w = w5 - 0.7
sx = x5 + 0.35
sy, sh = yB + 1.25, 1.1
fracs = [N[s][0] / N_TOTAL for s in ("train", "val", "test")]
names = ("Train", "Val", "Test")
for i, (s, frac, nm) in enumerate(zip(("train", "val", "test"), fracs, names)):
    sw = total_w * frac
    box(ax, sx, sy, sw - 0.06, sh, fc=C_SPLIT[i], ec=C_SPLIT[i], r=0.06, z=2)
    n, u = N[s]
    ax.text(sx + (sw - 0.06) / 2, sy + sh - 0.28, nm, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white")
    ax.text(sx + (sw - 0.06) / 2, sy + sh - 0.58, f"{n:,}", ha="center", va="center",
            fontsize=6.5, color="white")
    ax.text(sx + (sw - 0.06) / 2, sy + 0.2, f"{100 * frac:.0f}%", ha="center",
            va="center", fontsize=5.6, color="white", alpha=0.9)
    # pills under each split: disjoint drug sets
    npills = max(2, int(round(10 * frac)))
    for k in range(npills):
        box(ax, sx + 0.12 + k * 0.30, sy - 0.3, 0.22, 0.12, fc=C_SPLIT[i], ec="white",
            lw=0.5, r=0.06, z=3)
    sx += sw
ax.text(x5 + 0.35, sy - 0.3 + 0.06 - 0.01, "", va="center")
ax.text(x5 + w5 / 2, yB + 0.45,
        "drug sets are disjoint across splits: no drug seen in training appears in test\n"
        "→ test measures generalisation to unseen medications",
        ha="center", va="center", fontsize=5.8, color=C_SUB, linespacing=1.3)
for xx in (x5 + 0.35 + total_w * fracs[0] - 0.03,
           x5 + 0.35 + total_w * (fracs[0] + fracs[1]) - 0.03):
    ax.text(xx, sy + sh / 2, "∅", ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=C_EDGE, zorder=5,
            bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=C_EDGE, lw=0.6))

# --- (6) personalized guardrail train / eval -----------------------------------
x6, w6 = 9.35, 8.0
stage_label(ax, x6 + 0.15, yB + hB - 0.18, 6, "Personalized guardrail: training & evaluation")
ix, iy, iw = x6 + 0.15, yB + 1.2, 2.3
box(ax, ix, iy + 0.55, iw, 0.42, fc="#DCE6F5", ec=C_PROF, lw=0.8, r=0.05, z=2)
ax.text(ix + iw / 2, iy + 0.76, "profile (20 fields)", ha="center", va="center",
        fontsize=5.8, color=C_TEXT)
box(ax, ix, iy, iw, 0.42, fc="white", ec=C_PROF, lw=0.8, r=0.05, z=2)
ax.text(ix + iw / 2, iy + 0.21, "prescription scenario", ha="center", va="center",
        fontsize=5.8, color=C_TEXT)
mx, my, mw, mh = ix + iw + 0.5, iy - 0.05, 2.1, 1.07
box(ax, mx, my, mw, mh, fc=C_MODEL, ec=C_MODEL, r=0.08, z=2)
ax.text(mx + mw / 2, my + mh - 0.28, "guardrail LLM", ha="center", va="center",
        fontsize=7, fontweight="bold", color="white")
ax.text(mx + mw / 2, my + 0.42, "base · SFT · GRPO", ha="center", va="center",
        fontsize=5.8, color="#D7DDE5")
ax.text(mx + mw / 2, my + 0.17, "trained on Train split", ha="center", va="center",
        fontsize=5.2, color="#D7DDE5", style="italic")
arrow(ax, (ix + iw, iy + 0.76), (mx, my + mh * 0.68), lw=1.0)
arrow(ax, (ix + iw, iy + 0.21), (mx, my + mh * 0.32), lw=1.0)
ox, ow = mx + mw + 0.5, 2.0
box(ax, ox, iy + 0.55, ow, 0.42, fc="#F8E7E7", ec=C_UNSAFE, lw=0.8, r=0.05, z=2)
ax.text(ox + ow / 2, iy + 0.76, "safe / unsafe", ha="center", va="center",
        fontsize=6, fontweight="bold", color=C_UNSAFE)
box(ax, ox, iy, ow, 0.42, fc="#FBEBE0", ec=C_RISK, lw=0.8, r=0.05, z=2)
ax.text(ox + ow / 2, iy + 0.21, "risk categories", ha="center", va="center",
        fontsize=6, fontweight="bold", color="#9A5A2E")
arrow(ax, (mx + mw, my + mh * 0.68), (ox, iy + 0.76), lw=1.0)
arrow(ax, (mx + mw, my + mh * 0.32), (ox, iy + 0.21), lw=1.0)
# evaluation strip: profile ablations on the test split
ex0, ey0, exw, exh = x6 + 0.15, yB + 0.1, w6 - 0.3, 0.8
box(ax, ex0, ey0, exw, exh, fc="white", ec="#B8C0CC", lw=0.7, r=0.06, z=2, ls="--")
ax.text(ex0 + 0.12, ey0 + exh - 0.17,
        "Evaluate on Test (unseen drugs) under profile ablations:",
        ha="left", va="center", fontsize=5.8, color=C_TEXT, fontweight="bold")
conds = [("true profile", C_PROF), ("shuffled profile", "#8172B3"),
         ("no profile", "#8D99AE"), ("non-personalized", "#8D99AE")]
cx0 = ex0 + 0.12
for name, col in conds:
    tw = 0.3 + 0.105 * len(name)
    box(ax, cx0, ey0 + 0.1, tw, 0.3, fc=col, ec=col, r=0.05, z=3)
    ax.text(cx0 + tw / 2, ey0 + 0.25, name, ha="center", va="center",
            fontsize=5.4, color="white", fontweight="bold", zorder=4)
    cx0 += tw + 0.1

# split -> train/eval
arrow(ax, (x5 + w5 - 0.05, sy + sh / 2), (x6 + 0.08, sy + sh / 2))

# labelled benchmark (row A verdict) -> split (row B), routed through the gap
xv = x4 + w4 / 2
y_gap = 3.95
x_target = x5 + w5 / 2
ax.plot([xv, xv, x_target], [yA - 0.17, y_gap, y_gap], color=C_ARROW, lw=1.2,
        solid_capstyle="round", zorder=3)
arrow(ax, (x_target, y_gap), (x_target, yB + hB + 0.08), lw=1.2)
ax.text((xv + x_target) / 2, y_gap - 0.1,
        f"labelled benchmark: {N_TOTAL:,} patients × 17 categories × verdict",
        ha="center", va="top", fontsize=6, color=C_SUB)

out = os.path.join(HERE, "fig_benchmark_pipeline")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out + ".png", bbox_inches="tight", pad_inches=0.02, dpi=300)
print("wrote", out + ".{pdf,png}")
