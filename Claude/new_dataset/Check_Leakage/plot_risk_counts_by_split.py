import ast
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PREFIX = "New_Claude_Personalized_Groundtruth_Data - similar patients dropped"

SPLITS = {
    "Train (n=3,772)": BASE_DIR / f"{PREFIX}.train.csv",
    "Validation (n=503)": BASE_DIR / f"{PREFIX}.val.csv",
    "Test (n=754)": BASE_DIR / f"{PREFIX}.test.csv",
}


def parse_risk(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return {}
    s = str(cell).strip()
    for fn in (json.loads, ast.literal_eval):
        try:
            d = fn(s)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    s2 = s.replace("'", '"').replace("True", "true").replace("False", "false")
    return json.loads(s2)


def count_risks(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    risk_df = pd.DataFrame(list(df["Risk_Categories"].apply(parse_risk)))
    for col in risk_df.columns:
        risk_df[col] = risk_df[col].map(
            lambda x: True if x is True or str(x).lower() == "true" else False
        )
    return risk_df.sum(axis=0).astype(int)


def main():
    counts_df = pd.DataFrame({name: count_risks(path) for name, path in SPLITS.items()})
    counts_df["Total"] = counts_df.sum(axis=1)
    counts_df = counts_df.sort_values("Total", ascending=False)

    categories = [
        f"{i+1}. {c.replace(' Risk', '')}" for i, c in enumerate(counts_df.index)
    ]
    split_names = list(SPLITS.keys())
    colors = ["#2b5c8f", "#e08214", "#2ca02c"]

    x = np.arange(len(categories))
    width = 0.27
    offsets = [-width, 0.0, width]

    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=200)

    for offset, split_name, color in zip(offsets, split_names, colors):
        vals = counts_df[split_name].values
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=split_name,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f"{v}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=color,
            )

    for sep in np.arange(0.5, len(categories) - 0.5, 1.0):
        ax.axvline(sep, color="#bbbbbb", linestyle=":", linewidth=0.8, alpha=0.8, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=40, ha="right", fontsize=9.5)
    ax.set_xlim(-0.6, len(categories) - 0.4)
    ax.set_ylim(0, counts_df[split_names].values.max() * 1.12)
    ax.set_ylabel("Number of Samples (Risk = True)", fontsize=11)
    ax.set_xlabel("Risk Category (17 Categories)", fontsize=11)
    ax.set_title(
        "Number of True Samples per Risk Category Across Train / Validation / Test Splits\n"
        "(After Dropping Similar Patient Profiles, Total N = 5,029)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, fontsize=10.5)

    plt.tight_layout()
    out_path = BASE_DIR / "risk_category_counts_by_split.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")

    out_csv = BASE_DIR / "risk_category_counts_by_split.csv"
    counts_df.to_csv(out_csv)
    print(f"Saved counts CSV to: {out_csv}")
    print("\nCounts table:")
    print(counts_df.to_string())


if __name__ == "__main__":
    main()
