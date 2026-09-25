import pandas as pd
import json
import numpy as np
from sklearn.metrics import jaccard_score, hamming_loss

# 1. Define the EXACT 17 categories in the correct order
ALL_CATEGORIES = [
    "Age Risk",
    "Alcohol Use Risk",
    "Allergy & Adverse Drug Reaction Risk",
    "Bleeding Risk",
    "Caffeine Intake Risk",
    "Cardiac Impairment Risk",
    "Dosage & Toxicity Risk",
    "Drug–Drug Interaction Risk",
    "Drug–Food Interaction Risk",
    "Hepatic Impairment Risk",
    "Infection Risk",
    "Pregnancy & Breastfeeding Risk",
    "Renal Impairment Risk",
    "Respiratory Impairment Risk",
    "Substance Use Risk",
    "Tobacco Use Risk",
    "Weight/BMI Risk"
]

def extract_strict_vector(json_str):
    """
    Ensures the output is ALWAYS a 17-element list of booleans.
    """
    vector = []
    try:
        # Handle cases where the data might already be a dict or is a string
        if isinstance(json_str, str):
            data = json.loads(json_str)
        else:
            data = json_str
            
        for cat in ALL_CATEGORIES:
            # Get the value, default to False if key is missing
            val = data.get(cat, False)
            # Convert various truthy values to actual booleans
            vector.append(True if str(val).lower() == 'true' else False)
    except Exception:
        # If parsing fails entirely, return 17 Falses
        return [False] * len(ALL_CATEGORIES)
    
    return vector

# 2. Load the datasets
pred_df = pd.read_csv("Claude/Qwen3Guard-Gen-8B/Final_Safety_Labels.csv")
gt_df = pd.read_csv("Claude/Claude_Personalized_Groundtruth_Data.csv")

# Standardize Patient ID type and sort to align rows
pred_df['Patient ID'] = pred_df['Patient ID'].astype(str)
gt_df['Patient ID'] = gt_df['Patient ID'].astype(str)

pred_df = pred_df.sort_values("Patient ID").reset_index(drop=True)
gt_df = gt_df.sort_values("Patient ID").reset_index(drop=True)

# 3. Extract vectors using the strict method
pred_list = [extract_strict_vector(x) for x in pred_df["Risk_Categories"]]
gt_list = [extract_strict_vector(x) for x in gt_df["Risk_Categories"]]

# Convert to NumPy arrays (should now work perfectly)
pred_vectors = np.array(pred_list)
gt_vectors = np.array(gt_list)

# 4. Calculate Metrics
h_acc = 1 - hamming_loss(gt_vectors, pred_vectors)
j_score = jaccard_score(gt_vectors, pred_vectors, average='samples')
exact_match = np.all(gt_vectors == pred_vectors, axis=1).mean()

print(f"--- Alignment Metrics (N={len(pred_vectors)}) ---")
print(f"Hamming Accuracy: {h_acc:.4f}")
print(f"Jaccard Similarity: {j_score:.4f}")
print(f"Exact Match Ratio: {exact_match:.4f}")

# 5. Identify the "Problematic" Samples
mismatches = np.sum(pred_vectors != gt_vectors, axis=1)
if np.any(mismatches > 0):
    worst_idx = np.argmax(mismatches)
    print(f"\nWorst Match: Patient {pred_df.iloc[worst_idx]['Patient ID']} " 
          f"({mismatches[worst_idx]} bit disagreements)")