import pandas as pd
import numpy as np
from itertools import combinations

similarity_threshold=0.8

def discriminate_similar_patients(file_path, similarity_threshold):
    """
    Identifies patients that are 'extra similar' based on a percentage threshold
    of matching clinical and demographic columns.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Define the 20 clinical profile columns provided in the prompt
    profile_columns = [
        "Age", "Gender", "Weight (kg)", "Height (cm)", "BMI", 
        "Genetic Disorders", "Chronic Conditions", "Pregnancy / Breastfeeding", 
        "Drug Allergies", "Renal Impairment", "Hepatic Impairment", 
        "Cardiac Impairment", "Respiratory Impairment", "Alcohol Use", 
        "Tobacco Use", "Substance Use", "Caffeine Intake", 
        "Current Medications", "Foods (Last 24h)", "Symptoms"
    ]

    # Preprocessing: Convert to string, normalize case, and handle missing values
    # We ignore 'Patient ID' as it is a unique counter.
    df_clean = df[profile_columns].astype(str).apply(lambda x: x.str.strip().str.lower())
    
    data_matrix = df_clean.values
    n_rows = data_matrix.shape[0]
    n_cols = data_matrix.shape[1]
    
    similar_records = []

    # Efficient pairwise comparison
    for i, j in combinations(range(n_rows), 2):
        # Calculate how many columns match exactly
        matches = np.sum(data_matrix[i] == data_matrix[j])
        similarity = matches / n_cols
        
        if similarity >= similarity_threshold:
            similar_records.append({
                "Patient_A_ID": df.iloc[i]["Patient ID"],
                "Patient_B_ID": df.iloc[j]["Patient ID"],
                "Similarity_Score": f"{similarity:.1%}",
                "Matching_Count": f"{matches}/{n_cols}",
                "Discrepancies": [col for k, col in enumerate(profile_columns) 
                                 if data_matrix[i][k] != data_matrix[j][k]]
            })

    results_df = pd.DataFrame(similar_records)
    
    if not results_df.empty:
        # Sort by highest similarity first
        results_df = results_df.sort_values(by="Similarity_Score", ascending=False)
        return results_df
    else:
        return "No extra similar patients found above the threshold."

# Run the discrimination logic
# Adjust threshold as needed (e.g., 0.90 for 90% match)
similar_patients_report = discriminate_similar_patients('Claude/Knowledge_Distillation/Claude_Personalized_Groundtruth_Distill.csv', similarity_threshold)

# Output results to a CSV for manual inspection
if isinstance(similar_patients_report, pd.DataFrame):
    similar_patients_report.to_csv('Claude/Check_Leakage/flat_rate_profile_similarities.csv', index=False)
    print(f"Analysis complete. Found {len(similar_patients_report)} suspicious pairs.")
else:
    print(similar_patients_report)