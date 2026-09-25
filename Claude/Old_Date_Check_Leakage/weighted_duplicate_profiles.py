import pandas as pd
import json
from itertools import combinations

def parse_risk_categories(risk_str):
    """
    Safely extracts a set of only the 'true' risks from the JSON string.
    """
    if pd.isna(risk_str): 
        return set()
    try:
        # Standardize formatting to ensure valid JSON parsing
        clean_str = str(risk_str).replace("'", '"').replace("True", "true").replace("False", "false")
        risk_dict = json.loads(clean_str)
        # Return a set of risks that evaluated to true
        return {k for k, v in risk_dict.items() if v is True}
    except Exception:
        return set()

def detect_clinical_leakage(file_path):
    df = pd.read_csv(file_path)
    
    # Fill NAs to prevent matching errors
    df = df.fillna("None")
    
    # Define feature categories based on clinical weight
    fingerprints = [
        'Age', 'Weight (kg)', 'Height (cm)', 'BMI', 
        'Chronic Conditions', 'Cardiac Impairment', 'Respiratory Impairment'
    ]
    interacting_vars = ['Current Medications', 'Diagnosis']
    specific_risks = ['Renal Impairment', 'Hepatic Impairment', 'Drug Allergies']
    
    # Extract the 'True' risks into a new column for fast comparison
    df['Triggered_Risks'] = df['Risk_Categories'].apply(parse_risk_categories)
    
    leakage_suspects = []

    # Pairwise comparison to find exact leakage pathways
    for i, j in combinations(range(len(df)), 2):
        rowA, rowB = df.iloc[i], df.iloc[j]
        
        # 1. Compare Labels & Reasoning
        same_risks = (rowA['Triggered_Risks'] == rowB['Triggered_Risks']) and len(rowA['Triggered_Risks']) > 0
        same_safety = (rowA['Is_Safe'] == rowB['Is_Safe'])
        same_reasoning = (rowA['Reasoning'].strip().lower() == rowB['Reasoning'].strip().lower())
        
        # 2. Compare High-Weight Variables
        same_interacting = all(rowA[col] == rowB[col] for col in interacting_vars)
        same_specifics = all(rowA[col] == rowB[col] for col in specific_risks)
        
        # 3. Identify differences in Biometrics & Core Health (The Fingerprints)
        diff_fingerprints = [col for col in fingerprints if rowA[col] != rowB[col]]
        
        risk_level = None
        reason = None
        
        # GOLD STANDARD CHECK: 
        # If they trigger the exact same risks AND share the high-weight variables 
        # AND share the same reasoning, the model doesn't have to "read" the differing fingerprints.
        if same_risks and same_safety and same_reasoning:
            if same_interacting:
                risk_level = "HIGH RISK - Interacting Meds Leakage"
                reason = "Model sees the identical drug-drug/diagnosis puzzle and exact same reasoning."
            elif same_specifics:
                risk_level = "HIGH RISK - Contraindication Leakage"
                reason = "Model sees the identical impairment/allergy puzzle and exact same reasoning."
                
        # Valid SFT Data Check:
        # If risks match, but the clinical reasoning or the foundational fingerprints differ, it's safe.
        elif same_risks and (not same_reasoning):
            # We don't flag this as leakage, but we can log it for analysis
            pass 
            
        if risk_level:
            leakage_suspects.append({
                "Patient_A_ID": rowA['Patient ID'],
                "Patient_B_ID": rowB['Patient ID'],
                "Leakage_Category": risk_level,
                "Explanation": reason,
                "Triggered_Risks_Shared": list(rowA['Triggered_Risks']),
                "Differing_Fingerprints": diff_fingerprints
            })

    results_df = pd.DataFrame(leakage_suspects)
    
    if not results_df.empty:
        print(f"Found {len(results_df)} instances of High-Risk Data Leakage.")
        return results_df
    else:
        print("Dataset is clean. No high-risk leakage pathways found based on the Gold Standard Rule.")
        return None

# Execute the check
leakage_report = detect_clinical_leakage('Claude/Knowledge_Distillation/Claude_Personalized_Groundtruth_Distill.csv')
if leakage_report is not None:
    leakage_report.to_csv('Claude/Check_Leakage/weighted_profile_similarities.csv', index=False)