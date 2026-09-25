import pandas as pd
import json
import matplotlib.pyplot as plt

# 1. Load the datasets
model_output_path = 'Claude/Qwen3Guard-Gen-8B/Final_Safety_Labels.csv'
gt_path = 'Claude/Claude_Personalized_Groundtruth_Data.csv'

df_model = pd.read_csv(model_output_path)
df_gt = pd.read_csv(gt_path)

# 2. Filter for matching Unsafe labels (Both Model and GT say Is_Safe is False)
# This targets the "True Negatives" (or True Positives for 'Unsafe' class)
matching_unsafe_indices = df_model[(df_model['Is_Safe'] == False) & (df_gt['Is_Safe'] == False)].index

agreement_counts = []

# 3. Comparison Logic for the 17 risk categories
for idx in matching_unsafe_indices:
    try:
        # Convert string representation of dict to actual dict
        model_risk = json.loads(df_model.loc[idx, 'Risk_Categories'].replace("'", '"'))
        gt_risk = json.loads(df_gt.loc[idx, 'Risk_Categories'].replace("'", '"'))
        
        # Count identical boolean values across all 17 keys
        matches = sum(1 for key in model_risk if model_risk[key] == gt_risk.get(key))
        agreement_counts.append(matches)
    except (json.JSONDecodeError, AttributeError):
        continue

# 4. Aggregate data (x-axis: 0 to 17 matches)
plot_data = pd.Series(agreement_counts).value_counts().reindex(range(0, 18), fill_value=0)

# 5. Plotting and Saving
plt.figure(figsize=(12, 7))
colors = plt.cm.viridis([i/17 for i in range(18)]) # Gradient color for visual clarity
plot_data.plot(kind='bar', color=colors, edgecolor='black', width=0.8)

plt.title('Risk Category Agreement Distribution (Only Samples where Both = Unsafe)', fontsize=14, pad=20)
plt.xlabel('Number of Risks with Identical Boolean Values (out of 17)', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle=':', alpha=0.7)

# Adding value labels on top of bars for precision
for i, v in enumerate(plot_data):
    if v > 0:
        plt.text(i, v + (max(plot_data)*0.01), str(v), ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()

# Save the figure to disk - no show() call
plt.savefig('Unsafe_Samples_Risk_Agreement.png', dpi=300)
plt.close()