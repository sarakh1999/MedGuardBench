from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "New_Claude_Personalized_Groundtruth_Data - similar patients dropped.csv"

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

# Match longest drug name first so e.g. Valacyclovir matches before Acyclovir,
# Escitalopram before Citalopram, and Adalimumab (Humira) matches "adalimumab".
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


def main():
    df = pd.read_csv(CSV_PATH)
    df["Matched_Drug"] = df["Recommended Medication"].apply(match_drug)
    df["Is_Safe_Bool"] = df["Is_Safe"].map(
        lambda x: True if x is True or str(x).strip().lower() == "true" else False
    )

    counts = (
        df.groupby(["Matched_Drug", "Is_Safe_Bool"])
        .size()
        .unstack(fill_value=0)
        .reindex(DRUGS_90, fill_value=0)
    )
    counts = counts.rename(columns={True: "Positive (+)", False: "Negative (-)"})
    for col in ["Positive (+)", "Negative (-)"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["Positive (+)", "Negative (-)"]]
    counts["Total"] = counts["Positive (+)"] + counts["Negative (-)"]
    counts = counts[counts["Total"] > 0]
    active_drugs = list(counts.index)

    x = np.arange(len(active_drugs))
    width = 0.40

    fig, ax = plt.subplots(figsize=(28, 9), dpi=200)

    pos_vals = counts["Positive (+)"].values
    neg_vals = counts["Negative (-)"].values

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

    for bar, v in zip(bars_pos, pos_vals):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{v}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
                color="#1b661b",
            )

    for bar, v in zip(bars_neg, neg_vals):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{v}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
                color="#8c1919",
            )

    for sep in np.arange(0.5, len(active_drugs) - 0.5, 1.0):
        ax.axvline(sep, color="#bbbbbb", linestyle=":", linewidth=0.7, alpha=0.75, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{i+1}. {d}" for i, d in enumerate(active_drugs)],
        rotation=90,
        fontsize=8.5,
    )
    ax.set_xlim(-0.7, len(active_drugs) - 0.3)
    ax.set_ylim(0, max(pos_vals.max(), neg_vals.max()) * 1.15)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_xlabel(f"Medication ({len(active_drugs)} Active Drugs)", fontsize=12)
    ax.set_title(
        "Positive (+) vs. Negative (-) Samples per Drug After Dropping Similar Patient Profiles\n"
        f"(Matched {int(counts['Total'].sum()):,} / {len(df):,} Samples Across {len(active_drugs)} Drugs)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, fontsize=11)

    plt.tight_layout()
    out_path = BASE_DIR / "pos_neg_per_drug.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")

    out_csv = BASE_DIR / "pos_neg_per_drug_counts.csv"
    counts.to_csv(out_csv)
    print(f"Saved counts CSV to: {out_csv}")
    print("\nTop 15 drugs by total samples:")
    print(counts.sort_values("Total", ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()
