from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PREFIX = "New_Claude_Personalized_Groundtruth_Data - similar patients dropped"

SPLITS = {
    "Total\n(N=5,029)": BASE_DIR / f"{PREFIX}.csv",
    "Train\n(n=3,772)": BASE_DIR / f"{PREFIX}.train.csv",
    "Validation\n(n=503)": BASE_DIR / f"{PREFIX}.val.csv",
    "Test\n(n=754)": BASE_DIR / f"{PREFIX}.test.csv",
}


def count_labels(csv_path: Path):
    df = pd.read_csv(csv_path)
    is_safe = df["Is_Safe"].map(
        lambda x: True if x is True or str(x).strip().lower() == "true" else False
    )
    pos_safe = int((is_safe == True).sum())
    neg_unsafe = int((is_safe == False).sum())
    return pos_safe, neg_unsafe


def main():
    rows = []
    for name, path in SPLITS.items():
        pos_safe, neg_unsafe = count_labels(path)
        total = pos_safe + neg_unsafe
        rows.append(
            {
                "Split": name,
                "Positive (+ / Safe)": pos_safe,
                "Negative (- / Unsafe)": neg_unsafe,
                "Total": total,
            }
        )
    summary_df = pd.DataFrame(rows)

    x = np.arange(len(summary_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=200)

    pos_vals = summary_df["Positive (+ / Safe)"].values
    neg_vals = summary_df["Negative (- / Unsafe)"].values
    totals = summary_df["Total"].values

    bars_pos = ax.bar(
        x - width / 2,
        pos_vals,
        width,
        label="Positive (+ / Safe, Is_Safe = True)",
        color="#2ca02c",
        edgecolor="white",
        linewidth=0.7,
    )
    bars_neg = ax.bar(
        x + width / 2,
        neg_vals,
        width,
        label="Negative (- / Unsafe, Is_Safe = False)",
        color="#d62728",
        edgecolor="white",
        linewidth=0.7,
    )

    for bar, v, tot in zip(bars_pos, pos_vals, totals):
        pct = 100.0 * v / tot
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 35,
            f"{v:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#1b661b",
        )

    for bar, v, tot in zip(bars_neg, neg_vals, totals):
        pct = 100.0 * v / tot
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 35,
            f"{v:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#8c1919",
        )

    for sep in np.arange(0.5, len(summary_df) - 0.5, 1.0):
        ax.axvline(sep, color="#bbbbbb", linestyle=":", linewidth=0.9, alpha=0.8, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Split"], fontsize=11)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.set_title(
        "Positive (+) vs. Negative (-) Samples After Dropping Similar Patient Profiles\n"
        "(Remaining Total N = 5,029)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylim(0, max(neg_vals) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    plt.tight_layout()
    out_path = BASE_DIR / "pos_neg_sample_counts.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")
    print("\nSummary table:")
    print(
        summary_df.assign(
            Split=summary_df["Split"].str.replace("\n", " ", regex=False)
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
