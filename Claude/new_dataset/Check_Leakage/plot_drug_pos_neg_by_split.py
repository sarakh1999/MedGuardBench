from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PREFIX = "New_Claude_Personalized_Groundtruth_Data - similar patients dropped"

SPLITS = {
    "Train": BASE_DIR / f"{PREFIX}.train.csv",
    "Validation": BASE_DIR / f"{PREFIX}.val.csv",
    "Test": BASE_DIR / f"{PREFIX}.test.csv",
}

DRUGS_90 = [
    "Acetaminophen", "Acyclovir", "Adalimumab (Humira)", "Albuterol", "Alprazolam",
    "Amiodarone", "Amlodipine", "Amoxicillin", "Atorvastatin", "Bromocriptine",
    "Buspirone", "Cefdinir", "Celecoxib", "Ciprofloxacin", "Citalopram",
    "Clarithromycin", "Clindamycin", "Clonidine", "Clozaril", "Dapagliflozin",
    "Desvenlafaxine", "Diazepam", "Dicyclomine", "Digoxin", "Docusate",
    "Doxepin", "Dulaglutide", "Empagliflozin", "Erythromycin", "Escitalopram",
    "Finasteride", "Fluconazole", "Fluoxetine", "Folic Acid", "Furosemide",
    "Gabapentin", "Hydrochlorothiazide", "Hydrocodone", "Ibuprofen", "Indomethacin",
    "Isosorbide", "Isotretinoin", "Ivermectin", "Ketotifen", "Lactulose",
    "Lepirudin", "Lisinopril", "Liothyronine", "Loperamide", "Losartan",
    "Lovastatin", "Melatonin", "Memantine", "Metformin", "Methadone",
    "Methamphetamine", "Methylphenidate", "Metoprolol", "Mirtazapine", "Morphine",
    "Mounjaro", "Nabumetone", "Naproxen", "Niacin", "Nifedipine",
    "Olanzapine", "Omeprazole", "Oseltamivir", "Phenazopyridine", "Phenelzine",
    "Pimozide", "Pregabalin", "Promethazine", "Propranolol", "Quinidine",
    "Sertraline", "Sildenafil", "Sevelamer", "Spironolactone", "Sumatriptan",
    "Synthroid", "Tamsulosin", "Terazosin", "Terbinafine", "Tetracycline",
    "Tramadol", "Trazodone", "Valacyclovir", "Valsartan", "Warfarin",
]

SEARCH_TERMS = {d: d.split(" (")[0].lower() for d in DRUGS_90}
ORDER_LONGEST = sorted(DRUGS_90, key=lambda d: len(SEARCH_TERMS[d]), reverse=True)


def match_drug(med_string) -> str:
    if pd.isna(med_string):
        return "Other"
    s = str(med_string).lower()
    for d in ORDER_LONGEST:
        if SEARCH_TERMS[d] in s:
            return d
    return "Other"


def get_split_drug_pos_neg(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Matched_Drug"] = df["Recommended Medication"].apply(match_drug)
    df["Is_Safe_Bool"] = df["Is_Safe"].map(
        lambda x: True if x is True or str(x).strip().lower() == "true" else False
    )
    df = df[df["Matched_Drug"] != "Other"]

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
    # Keep drugs in the canonical alphabetical order from DRUGS_90
    ordered = [d for d in DRUGS_90 if d in counts.index and counts.loc[d, "Total"] > 0]
    return counts.loc[ordered]


def main():
    split_tables = {name: get_split_drug_pos_neg(path) for name, path in SPLITS.items()}

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
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[10, 14], hspace=0.52, wspace=0.12)
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
            [f"{i+1}. {d}" for i, d in enumerate(drugs)],
            rotation=90 if split_name == "Train" else 40,
            ha="center" if split_name == "Train" else "right",
            fontsize=8.5 if split_name == "Train" else 9.5,
        )
        ax.set_xlim(-0.65, len(drugs) - 0.35)
        ax.set_ylim(0, y_max * 1.18)
        ax.set_ylabel("Number of Samples", fontsize=11)
        ax.set_title(
            f"{split_name} Split — {len(drugs)} Drugs "
            f"(+ / Safe: {int(pos_vals.sum()):,},  - / Unsafe: {int(neg_vals.sum()):,},  Total Matched: {int(tbl['Total'].sum()):,})",
            fontsize=12.5,
            fontweight="bold",
            pad=10,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", frameon=True, fontsize=9.5)

    fig.suptitle(
        "Positive (+ / True) and Negative (- / False) Samples per Drug Grouped by Split\n"
        "(After Dropping Similar Patient Profiles — 85 Active Drugs Across Train, Validation, and Test)",
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
