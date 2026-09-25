import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast
import os
import re
import numpy as np


# Set visual style
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------
# 1. LOAD CONFIGURATION FROM PLAIN TEXT FILES
# ----------------------------------------------------------------------
def load_list_from_txt(filename):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return []
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

risk_categories = load_list_from_txt('risk_categories.txt')
drugs_list = load_list_from_txt('new_medications.txt')

# ----------------------------------------------------------------------
# 2. LOAD AND CLEAN DATASET
# ----------------------------------------------------------------------
df = pd.read_csv('Claude/new_dataset/Check_Leakage/New_Claude_Personalized_Groundtruth_Data - similar patients dropped.csv')
# df = pd.read_csv("Claude/new_dataset/Check_Leakage/New_Claude_Personalized_Groundtruth_Data.csv")
def parse_risk_column(risk_str):
    """Parse the Risk_Categories cell into a dict of {category: bool}.
    Handles both JSON style ({"x": true}) and Python style ({'x': True})."""
    if pd.isna(risk_str) or str(risk_str).strip() == "":
        return {risk: False for risk in risk_categories}
    s = str(risk_str).strip()
    # Attempt 1: Python literal (handles single quotes + True/False)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass
    # Attempt 2: JSON (handles double quotes + true/false)
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Attempt 3: convert single->double quotes and True->true, then JSON
    try:
        s2 = s.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
        parsed = json.loads(s2)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    print(f"  [WARN] Could not parse Risk_Categories value: {s[:80]}")
    return {risk: False for risk in risk_categories}

# ----------------------------------------------------------------------
# DRUG MATCHING (containment-based, longest name first)
# ----------------------------------------------------------------------
drugs_sorted = sorted(drugs_list, key=len, reverse=True)

def find_drug_regex(med_string):
    if pd.isna(med_string):
        return "Unknown"
    med_lower = str(med_string).lower()
    for drug in drugs_sorted:
        if drug.lower() in med_lower:   # simple containment, case-insensitive
            return drug
    return "Other"

# Map medications and parse risk categories
df['Matched_Drug'] = df['Recommended Medication'].apply(find_drug_regex)
risk_df = df['Risk_Categories'].apply(parse_risk_column).apply(pd.Series)

# Keep only the expected risk category columns, in the expected order,
# creating any missing ones as 0 — prevents silent column-name mismatches
risk_df = risk_df.reindex(columns=risk_categories, fill_value=False)
risk_df = risk_df.fillna(False).astype(bool).astype(int)

# ----------------------------------------------------------------------
# DIAGNOSTICS — check these before trusting the plots
# ----------------------------------------------------------------------
print("\n=== DIAGNOSTICS ===")
print("Matched_Drug value counts:")
print(df['Matched_Drug'].value_counts().head(20))
n_other = (df['Matched_Drug'] == 'Other').sum()
print(f"\nRows matched to 'Other' (no drug found): {n_other} / {len(df)}")
print(f"\nTotal True risks parsed per category:")
print(risk_df.sum())
if risk_df.sum().sum() == 0:
    print("\n[WARN] All risk values are 0 — Risk_Categories parsing is failing.")
    print("Sample raw values from the column:")
    print(df['Risk_Categories'].dropna().head(3).to_list())
print("===================\n")

# Combine and normalize safety labels
df_processed = pd.concat([df[['Matched_Drug', 'Is_Safe']], risk_df], axis=1)
df_processed['Is_Safe'] = df_processed['Is_Safe'].astype(str).str.upper().str.strip()

# ----------------------------------------------------------------------
# 3. DEFINE CUSTOM COLOR PALETTE
# ----------------------------------------------------------------------
safety_colors = {"TRUE": "#2ecc71", "FALSE": "#e74c3c"}

# ----------------------------------------------------------------------
# PLOT 1: Risk Category Frequency
# ----------------------------------------------------------------------
print("Generating Plot 1: Risk Category Counts...")
risk_counts = df_processed[risk_categories].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(
    x=risk_counts.values,
    y=risk_counts.index,
    hue=risk_counts.index,
    palette="viridis",
    legend=False
)

max_val = int(risk_counts.max()) if not risk_counts.empty else 10
for i in range(0, max_val + 11, 10):
    plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

plt.title("Total Number of 'True' Cases per Risk Category")
plt.xlabel("Count of True Risks")
plt.tight_layout()
plt.savefig("Claude/new_dataset/Check_Leakage/risk_categories_counts.png", dpi=300)

# ----------------------------------------------------------------------
# PLOT 2A: Overall Safety Distribution
# ----------------------------------------------------------------------
print("Generating Plot 2A: Overall Safety...")
plt.figure(figsize=(8, 6))
sns.countplot(
    x="Is_Safe",
    data=df_processed,
    palette=safety_colors,
    hue="Is_Safe",
    order=["TRUE", "FALSE"],
    legend=False
)
plt.title("Overall Safety Distribution (General)")
plt.tight_layout()
plt.savefig("Claude/new_dataset/Check_Leakage/safety_general_distribution.png", dpi=300)

# ----------------------------------------------------------------------
# PLOT 2B: Safety Breakdown by Drug
# ----------------------------------------------------------------------
print("Generating Plot 2B: Safety by Drug...")
plt.figure(figsize=(24, 10))
ax = sns.countplot(
    x="Matched_Drug",
    hue="Is_Safe",
    data=df_processed,
    palette=safety_colors,
    hue_order=["TRUE", "FALSE"],
    order=sorted(drugs_list)
)

for i in range(len(drugs_list) - 1):
    plt.axvline(x=i + 0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)

plt.title("Safety Breakdown by Recommended Medication (True=Green, False=Red)", fontsize=16)
plt.xticks(rotation=90)
plt.xlabel("Matched Medication Name")
plt.tight_layout()
plt.savefig("Claude/new_dataset/Check_Leakage/safety_by_medication.png", dpi=300)

# ----------------------------------------------------------------------
# PLOT 3: Risk Categories per Drug (Heatmap)
# ----------------------------------------------------------------------
print("Generating Plot 3: Risk Heatmap...")
drug_risk_matrix = df_processed.groupby("Matched_Drug")[risk_categories].sum()
drug_risk_matrix = drug_risk_matrix.reindex(drugs_list).fillna(0).astype(float)

heatmap_cmap = plt.cm.get_cmap("YlOrRd").copy()
# heatmap_cmap.set_under("blue")
# heatmap_cmap.set_under("#eef7ff")

plt.figure(figsize=(35, 12))
sns.heatmap(
    drug_risk_matrix.T,
    cmap=heatmap_cmap,
    # vmin=0.1,
    annot=True,
    fmt=".0f",
    cbar_kws={'label': 'Number of True Risks'}
)
plt.title("Number of 'True' Risks per Drug per Category")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("Claude/new_dataset/Check_Leakage/drug_risk_breakdown.png", dpi=300)

print("Process complete. All charts saved as separate PNG files.")

# ----------------------------------------------------------------------
# PLOT 4: Risk Category Fraction per Unsafe Sample (Heatmap)
# ----------------------------------------------------------------------
print("Generating Plot 4: Risk Fraction Heatmap...")

# Count only unsafe samples for each drug
unsafe_df = df_processed[df_processed["Is_Safe"] == "FALSE"]

# Number of unsafe samples per drug
unsafe_counts = (
    unsafe_df.groupby("Matched_Drug")
    .size()
    .reindex(drugs_list)
    .fillna(0)
)

# Number of True occurrences of each risk category among unsafe samples
unsafe_risk_matrix = (
    unsafe_df.groupby("Matched_Drug")[risk_categories]
    .sum()
    .reindex(drugs_list)
    .fillna(0)
)

# Convert counts to fractions
drug_risk_fraction = unsafe_risk_matrix.div(unsafe_counts, axis=0).fillna(0)

plt.figure(figsize=(50, 12))
sns.heatmap(
    drug_risk_fraction.T,
    cmap="YlOrRd",
    vmin=0,
    vmax=1.0,
    annot=True,
    fmt=".2f",
    cbar_kws={'label': 'Fraction of Unsafe Samples'}
)

plt.title("Fraction of Unsafe Samples with Each Risk Category per Drug")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(
    "Claude/new_dataset/Check_Leakage/drug_risk_fraction_breakdown.png",
    dpi=300
)



# --- Continuation from your script after Plot 4 processing ---

# 1. Transpose the matrix so columns represent drugs (matching the heatmap structure)
# Rows: Risk Categories | Columns: Matched Drugs
fraction_matrix_T = drug_risk_fraction.T.copy()

# 2. Compute the sum for each column (each drug) across all risk categories
column_sums = fraction_matrix_T.sum(axis=0)

# 3. Append the 'Sum' as a new row at the bottom of the DataFrame
df_to_export = fraction_matrix_T.copy()
df_to_export.loc["SUM"] = column_sums

# 4. Save to CSV
csv_output_path = (
    "Claude/new_dataset/Check_Leakage/drug_risk_fraction_breakdown_with_sums.csv"
)
df_to_export.to_csv(csv_output_path, float_format="%.4f")

print(f"CSV file successfully saved to: {csv_output_path}")

# Optional: Print a preview of the sums per drug
print("\n=== Column Sums (Total Risk Fraction per Drug) ===")
print(column_sums)