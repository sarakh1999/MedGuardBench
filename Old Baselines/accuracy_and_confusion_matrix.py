import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from matplotlib.colors import ListedColormap


results_df = pd.read_csv('Claude/Final_Safety_Labels.csv')
groundtruth_df = pd.read_csv('Claude/Claude_Personalized_Groundtruth_Data.csv')

# 2. Merge dataframes on Patient ID
combined_df = pd.merge(
    groundtruth_df[['Patient ID', 'Is_Safe']], 
    results_df[['Patient ID', 'Is_Safe']], 
    on='Patient ID', 
    suffixes=('_true', '_pred')
)

# Ensure no missing values after merge
combined_df = combined_df.dropna(subset=['Is_Safe_true', 'Is_Safe_pred'])

y_true = combined_df['Is_Safe_true']
y_pred = combined_df['Is_Safe_pred']

# 3. Calculate and Print Metrics (Single numbers only)
acc = accuracy_score(y_true, y_pred)
# 'macro' average provides a single recall score weighted across classes
rec = recall_score(y_true, y_pred, average='macro')

print(f"accuracy", round(acc,2))
print(f"recall", round(rec,2))

# 4. Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
labels = sorted(y_true.unique())

# 5. Create the Color-Coded Plot
# Create a matrix where 1 = Diagonal (TP/TN) and 0 = Off-diagonal (FP/FN)
color_mask = np.eye(cm.shape[0])

# Define colormap: Index 0 (Red) for FP/FN, Index 1 (Green) for TP/TN
# Colors used: #ff9999 (Light Red), #99ff99 (Light Green)
custom_cmap = ListedColormap(['#ff9999', '#99ff99'])

plt.figure(figsize=(8, 6))

# Use color_mask for the cell colors and cm for the numeric annotations
sns.heatmap(color_mask, annot=cm, fmt='d', cmap=custom_cmap, cbar=False,
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 14, "weight": "bold", "color": "black"})

plt.title('Confusion Matrix: Safety Labels (Green=Correct, Red=Incorrect)')
plt.ylabel('Actual (Ground Truth)')
plt.xlabel('Predicted (Final Labels)')

# Save the plot
plt.savefig('confusion_matrix_results.png')