import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]

# The split actually used for training / evaluation (and for the dataset tables
# in the paper). The older "similar patients dropped.{train,val,test}.csv" files
# in this folder are a superseded split and must not be used for paper figures.
SPLIT_DIR = REPO_ROOT / "Claude" / "SFT" / "new_data_splits"
SPLITS = {
    "Train": SPLIT_DIR / "train.csv",
    "Validation": SPLIT_DIR / "val.csv",
    "Test": SPLIT_DIR / "test.csv",
}

# Active medications, as listed in new_medications.txt (includes Prednisolone and Bupropion).
DRUGS = [ln.strip() for ln in open(REPO_ROOT / "new_medications.txt") if ln.strip()]
OTHER = "Other"

# Brand names, abbreviations, and combination products that are the same substance
# as a listed drug. Hemangeol is oral propranolol; Tylenol / APAP / Percocet /
# Excedrin contain acetaminophen; clozapine is Clozaril.
ALIASES = {
    "Acetaminophen": ["tylenol", "apap", "paracetamol", "excedrin", "percocet"],
    "Clozaril": ["clozapine"],
    "Propranolol": ["hemangeol"],
    "Bupropion": ["zyban", "wellbutrin"],
}


def _drug_pattern(name: str) -> re.Pattern:
    """Whole-word match (same rule as split_fair.py), so Acyclovir cannot claim Valacyclovir."""
    return re.compile(
        r"(?<![A-Za-z])" + r"\s+".join(re.escape(p) for p in name.split()) + r"(?![A-Za-z])",
        re.IGNORECASE,
    )


PATTERNS = {d: _drug_pattern(d) for d in DRUGS}
ALIAS_PATTERNS = {
    d: [re.compile(r"(?<![A-Za-z])" + re.escape(a) + r"(?![A-Za-z])", re.IGNORECASE) for a in aliases]
    for d, aliases in ALIASES.items()
}


def match_text(text):
    """Earliest listed drug or alias in text, or None."""
    if pd.isna(text):
        return None
    s = str(text)
    hits = [(m.start(), d) for d, p in PATTERNS.items() if (m := p.search(s))]
    for d, patterns in ALIAS_PATTERNS.items():
        for p in patterns:
            if m := p.search(s):
                hits.append((m.start(), d))
    if not hits:
        return None
    return min(hits)[1]


def match_row(row) -> str:
    """Map a scenario onto a listed drug.

    The recommendation is tried first. Brand names and abbreviations count as the
    listed drug. If the recommendation names no listed drug (for example
    "continue both" or "supportive care"), the dosage, the clinical prompt, and
    the current medications are tried in that order.
    """
    hit = match_text(row["Recommended Medication"])
    if hit:
        return hit
    for col in ("Dosage", "Prompt / Clinical Scenario", "Current Medications"):
        hit = match_text(row[col])
        if hit:
            return hit
    return OTHER


def get_split_drug_pos_neg(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df["Matched_Drug"] = df.apply(match_row, axis=1)
    df["Is_Safe_Bool"] = df["Is_Safe"].map(lambda x: str(x).strip().lower() == "true")

    counts = (
        df.groupby(["Matched_Drug", "Is_Safe_Bool"])
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.rename(columns={True: "Positive (+ / True)", False: "Negative (- / False)"})
    for col in ["Positive (+ / True)", "Negative (- / False)"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["Positive (+ / True)", "Negative (- / False)"]]
    counts["Total"] = counts["Positive (+ / True)"] + counts["Negative (- / False)"]
    # Alphabetical order of the 85 drugs, with the OTHER bucket (if any) last
    ordered = [d for d in sorted(DRUGS) if d in counts.index and counts.loc[d, "Total"] > 0]
    if OTHER in counts.index and counts.loc[OTHER, "Total"] > 0:
        ordered.append(OTHER)
    assert counts.loc[ordered, "Total"].sum() == len(df), "rows lost in per-drug counting"
    return counts.loc[ordered]


# These seven drugs also pick up a few scenarios in a second split. The figure
# keeps each of them only in the split that holds the rest of its samples.
HOME_SPLIT = {
    "Acetaminophen": "Train",
    "Atorvastatin": "Validation",
    "Bupropion": "Train",
    "Dapagliflozin": "Train",
    "Erythromycin": "Train",
    "Propranolol": "Train",
    "Warfarin": "Train",
}


def load_split_frame(split_name, path):
    """Rows counted in the figure: aliases resolved, each of the seven
    cross-split drugs kept only in its home split."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["Matched_Drug"] = df.apply(match_row, axis=1)
    home = df["Matched_Drug"].map(HOME_SPLIT)
    keep = home.isna() | (home == split_name)
    return df.loc[keep].copy()


def main():
    split_tables = {name: get_split_drug_pos_neg(path) for name, path in SPLITS.items()}
    for split_name, tbl in split_tables.items():
        drop = [d for d in tbl.index if HOME_SPLIT.get(d) not in (None, split_name)]
        if drop:
            print(f"{split_name}: omitting {drop}")
            split_tables[split_name] = tbl.drop(index=drop)

    # 1. Combined single-panel plot grouped by split (Train | Validation | Test)
    combined_rows = []
    for split_name, tbl in split_tables.items():
        for i, (drug, row) in enumerate(tbl.iterrows(), start=1):
            combined_rows.append(
                {
                    "Split": split_name,
                    "Split_Index": i,
                    "Drug": drug,
                    "Positive (+ / True)": int(row["Positive (+ / True)"]),
                    "Negative (- / False)": int(row["Negative (- / False)"]),
                    "Total": int(row["Total"]),
                }
            )
    combined_df = pd.DataFrame(combined_rows)
    combined_df["Global_Index"] = np.arange(1, len(combined_df) + 1)
    combined_df.to_csv(BASE_DIR / "drug_pos_neg_by_split.csv", index=False)

    # Create a 3-row figure (Train on top, Validation and Test on bottom row side-by-side)
    fig = plt.figure(figsize=(28, 12), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1, 1], hspace=0.52, wspace=0.16)
    ax_train = fig.add_subplot(gs[0, :])
    ax_val = fig.add_subplot(gs[1, 0])
    ax_test = fig.add_subplot(gs[1, 1])

    split_axes = {
        "Train": ax_train,
        "Validation": ax_val,
        "Test": ax_test,
    }
    split_bg = {
        "Train": "#f4f8fc",
        "Validation": "#fdf6ee",
        "Test": "#f3faf3",
    }

    for split_name, ax in split_axes.items():
        tbl = split_tables[split_name]
        drugs = list(tbl.index)
        n_drugs = sum(d != OTHER for d in drugs)
        x = np.arange(len(drugs))
        width = 0.40

        pos_vals = tbl["Positive (+ / True)"].values
        neg_vals = tbl["Negative (- / False)"].values

        ax.set_facecolor(split_bg[split_name])

        bars_pos = ax.bar(
            x - width / 2,
            pos_vals,
            width,
            label="Positive (+ / Safe, Is_Safe = True)",
            color="#2ca02c",
            edgecolor="white",
            linewidth=0.4,
        )
        bars_neg = ax.bar(
            x + width / 2,
            neg_vals,
            width,
            label="Negative (- / Unsafe, Is_Safe = False)",
            color="#d62728",
            edgecolor="white",
            linewidth=0.4,
        )

        y_max = max(pos_vals.max(), neg_vals.max())
        offset_y = max(1.2, y_max * 0.02)

        for bar, v in zip(bars_pos, pos_vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset_y,
                    f"{v}",
                    ha="center",
                    va="bottom",
                    fontsize=7.0 if split_name == "Train" else 8.0,
                    rotation=90 if split_name == "Train" else 0,
                    color="#1b661b",
                    fontweight="bold",
                )

        for bar, v in zip(bars_neg, neg_vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset_y,
                    f"{v}",
                    ha="center",
                    va="bottom",
                    fontsize=7.0 if split_name == "Train" else 8.0,
                    rotation=90 if split_name == "Train" else 0,
                    color="#8c1919",
                    fontweight="bold",
                )

        for sep in np.arange(0.5, len(drugs) - 0.5, 1.0):
            ax.axvline(sep, color="#bbbbbb", linestyle=":", linewidth=0.75, alpha=0.85, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [("Other" if d == OTHER else f"{i+1}. {d}") for i, d in enumerate(drugs)],
            rotation=90 if split_name == "Train" else 40,
            ha="center" if split_name == "Train" else "right",
            fontsize=8.5 if split_name == "Train" else 9.5,
        )
        ax.set_xlim(-0.65, len(drugs) - 0.35)
        ax.set_ylim(0, y_max * 1.18)
        ax.set_ylabel("Number of Samples", fontsize=11)
        ax.set_title(
            f"{split_name} Split — {n_drugs} Drugs "
            f"(+ / Safe: {int(pos_vals.sum()):,},  - / Unsafe: {int(neg_vals.sum()):,},  Total: {int(tbl['Total'].sum()):,})",
            fontsize=12.5,
            fontweight="bold",
            pad=10,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", frameon=True, fontsize=9.5)

    n_listed = len(DRUGS)
    fig.suptitle(
        "Positive (+ / True) and Negative (- / False) Samples per Drug Grouped by Split\n"
        f"({n_listed} Listed Drugs; Brand Names and Aliases Counted with the Listed Drug)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    out_path = BASE_DIR / "drug_pos_neg_by_split.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")

    for split_name, tbl in split_tables.items():
        print(f"\n=== {split_name} ({len(tbl)} drugs, Total={tbl['Total'].sum()}) ===")
        print(tbl.to_string())


if __name__ == "__main__":
    main()
