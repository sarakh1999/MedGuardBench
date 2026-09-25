import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast
import os
import re

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

def parse_risk_column(risk_str):
    if pd.isna(risk_str) or risk_str == "":
        return {risk: False for risk in risk_categories}
    try:
        return json.loads(risk_str.replace("'", '"'))
    except:
        try:
            return ast.literal_eval(risk_str)
        except:
            return {risk: False for risk in risk_categories}

def find_drug_regex(med_string):
    if pd.isna(med_string):
        return "Unknown"
    for drug in drugs_list:
        pattern = rf"\b{re.escape(drug)}\b"
        if re.search(pattern, str(med_string), re.IGNORECASE):
            return drug
    return "Other"

# Map medications and parse risk categories
df['Matched_Drug'] = df['Recommended Medication'].apply(find_drug_regex)
risk_df = df['Risk_Categories'].apply(parse_risk_column).apply(pd.Series)
risk_df = risk_df.fillna(False).astype(int) 

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

# Dashed vertical lines every 10 steps
max_val = int(risk_counts.max()) if not risk_counts.empty else 10
for i in range(0, max_val + 11, 10):
    plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

plt.title("Total Number of 'True' Cases per Risk Category")
plt.xlabel("Count of True Risks")
plt.tight_layout()
plt.savefig("Claude/Check_Leakage/risk_categories_counts.png", dpi=300)

# ----------------------------------------------------------------------
# PLOT 2A: Overall Safety Distribution (Separate File)
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
plt.savefig("Claude/Check_Leakage/safety_general_distribution.png", dpi=300)

# ----------------------------------------------------------------------
# PLOT 2B: Safety Breakdown by Drug (Wider + Separator Lines)
# ----------------------------------------------------------------------
print("Generating Plot 2B: Safety by Drug...")
plt.figure(figsize=(24, 10)) # Wider figure to accommodate many drugs
ax = sns.countplot(
    x="Matched_Drug", 
    hue="Is_Safe", 
    data=df_processed, 
    palette=safety_colors,
    hue_order=["TRUE", "FALSE"],
    order=sorted(drugs_list)
)

# Add dashed separator lines between each drug
for i in range(len(drugs_list) - 1):
    plt.axvline(x=i + 0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)

plt.title("Safety Breakdown by Recommended Medication (True=Green, False=Red)", fontsize=16)
plt.xticks(rotation=90)
plt.xlabel("Matched Medication Name")
plt.tight_layout()
plt.savefig("Claude/Check_Leakage/safety_by_medication.png", dpi=300)

# ----------------------------------------------------------------------
# PLOT 3: Risk Categories per Drug (Heatmap)
# ----------------------------------------------------------------------
print("Generating Plot 3: Risk Heatmap...")
drug_risk_matrix = df_processed.groupby("Matched_Drug")[risk_categories].sum()
# Reindex and force to float to prevent TypeError
drug_risk_matrix = drug_risk_matrix.reindex(drugs_list).fillna(0).astype(float)

plt.figure(figsize=(20, 14))
sns.heatmap(
    drug_risk_matrix.T, 
    cmap="YlOrRd", 
    annot=True, 
    fmt=".0f", 
    cbar_kws={'label': 'Number of True Risks'}
)
plt.title("Number of 'True' Risks per Drug per Category")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("Claude/Check_Leakage/drug_risk_breakdown.png", dpi=300)

print("Process complete. All charts saved as separate PNG files.")