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


def count_drugs(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    matched = df["Recommended Medication"].apply(match_drug)
    return matched.value_counts().reindex(DRUGS_90, fill_value=0).astype(int)


def main():
    counts_df = pd.DataFrame({name: count_drugs(path) for name, path in SPLITS.items()})
    counts_df["Total"] = counts_df.sum(axis=1)
    counts_df = counts_df[counts_df["Total"] > 0]
    active_drugs = list(counts_df.index)

    split_names = list(SPLITS.keys())
    colors = {"Train": "#2b5c8f", "Validation": "#e08214", "Test": "#2ca02c"}

    x = np.arange(len(active_drugs))
    width = 0.27
    offsets = {"Train": -width, "Validation": 0.0, "Test": width}

    fig, ax = plt.subplots(figsize=(32, 9.5), dpi=200)

    for split_name in split_names:
        vals = counts_df[split_name].values
        n_drugs_in_split = int((vals > 0).sum())
        n_samples_in_split = int(vals.sum())
        label = f"{split_name} ({n_drugs_in_split} drugs, {n_samples_in_split:,} matched samples)"
        bars = ax.bar(
            x + offsets[split_name],
            vals,
            width=width,
            label=label,
            color=colors[split_name],
            edgecolor="white",
            linewidth=0.35,
        )
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2.5,
                    f"{v}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                    color=colors[split_name],
                    fontweight="bold",
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
    ax.set_ylim(0, counts_df[split_names].values.max() * 1.15)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_xlabel(f"Medication ({len(active_drugs)} Active Drugs)", fontsize=12)
    ax.set_title(
        "Drug Sample Counts Across Train / Validation / Test Splits After Dropping Similar Patient Profiles\n"
        f"(Matched {int(counts_df['Total'].sum()):,} / 5,029 Samples Across {len(active_drugs)} Active Drugs — Drug-Disjoint Split)",
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
    out_path = BASE_DIR / "drug_counts_by_split.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")

    out_csv = BASE_DIR / "drug_counts_by_split.csv"
    counts_df.to_csv(out_csv)
    print(f"Saved counts CSV to: {out_csv}")


if __name__ == "__main__":
    main()
