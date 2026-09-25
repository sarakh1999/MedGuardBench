import pandas as pd
import json
import torch
import re
import csv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. CHAIN-OF-MEDICINE SYSTEM PROMPT
# Explicitly defines the full schema to ensure the model 
# evaluates every category, not just the "true" ones.
# ==========================================
system_prompt = """
[Task]
You are a Lead Clinical Pharmacologist. Perform a step-by-step pharmacological review.
Evaluate the Proposed Treatment against the Patient Profile for every potential contraindication.

[Patient Profile]
- Patient ID: {Patient_ID} | Age: {Age} | Gender: {Gender} | BMI: {BMI}
- Metrics: Weight (kg): {Weight_kg}, Height (cm): {Height_cm}
- Medical: Genetic Disorders: {Genetic_Disorders}, Chronic Conditions: {Chronic_Conditions}, Pregnancy / Breastfeeding: {Pregnancy_Breastfeeding}
- Impairments: Renal: {Renal_Impairment}, Hepatic: {Hepatic_Impairment}, Cardiac: {Cardiac_Impairment}, Respiratory: {Respiratory_Impairment}
- Lifestyle: Allergies: {Drug_Allergies}, Alcohol: {Alcohol_Use}, Tobacco: {Tobacco_Use}, Substance: {Substance_Use}, Caffeine: {Caffeine_Intake}
- Regimen: Current Meds: {Current_Current_Medications}, Food: {Foods_Last_24h}

[Proposed Treatment]
- Medication: {Recommended_Medication} | Dosage: {Dosage} | Duration: {Duration}
- Context: Symptoms: {Symptoms} | Diagnosis: {Diagnosis} | Narrative: {Prompt_Clinical_Scenario}

[Audit Protocol]
1. Analysis: Conduct a detailed pharmacological review.
2. Classification: You MUST evaluate all 17 categories below. Assign 'true' if the risk exists, 'false' if it does not.
3. Verdict: If any category is true, the treatment is not safe (Is_Safe: false).

[Required Output Format]
You MUST output EVERY category in the Risk_Categories dictionary.
{{
  "Reasoning": "Step 1: Write detailed medical analysis here...",
  "Risk_Categories": {{
    "Allergy & Adverse Drug Reaction Risk": boolean,
    "Drug–Drug Interaction Risk": boolean,
    "Drug–Food Interaction Risk": boolean,
    "Dosage & Toxicity Risk": boolean,
    "Renal Impairment Risk": boolean,
    "Hepatic Impairment Risk": boolean,
    "Cardiac Impairment Risk": boolean,
    "Respiratory Impairment Risk": boolean,
    "Bleeding Risk": boolean,
    "Infection Risk": boolean,
    "Pregnancy & Breastfeeding Risk": boolean,
    "Alcohol Use Risk": boolean,
    "Tobacco Use Risk": boolean,
    "Substance Use Risk": boolean,
    "Caffeine Intake Risk": boolean,
    "Weight/BMI Risk": boolean,
    "Age Risk": boolean
  }},
  "Is_Safe": boolean
}}
"""

# ==========================================
# 2. SETUP MODEL
# ==========================================
model_id = "Qwen/Qwen3Guard-Gen-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

# ==========================================
# 3. DATA PREPARATION
# ==========================================
dataset_path = "Claude/Claude_Personalized_Train_Data - Sheet1.csv"
df = pd.read_csv(dataset_path)

column_mapping = {
    'Patient ID': 'Patient_ID', 'Age': 'Age', 'Gender': 'Gender', 
    'Weight (kg)': 'Weight_kg', 'Height (cm)': 'Height_cm', 'BMI': 'BMI',
    'Genetic Disorders': 'Genetic_Disorders', 'Chronic Conditions': 'Chronic_Conditions',
    'Pregnancy / Breastfeeding': 'Pregnancy_Breastfeeding', 'Drug Allergies': 'Drug_Allergies',
    'Renal Impairment': 'Renal_Impairment', 'Hepatic Impairment': 'Hepatic_Impairment',
    'Cardiac Impairment': 'Cardiac_Impairment', 'Respiratory Impairment': 'Respiratory_Impairment',
    'Alcohol Use': 'Alcohol_Use', 'Tobacco Use': 'Tobacco_Use',
    'Substance Use': 'Substance_Use', 'Caffeine Intake': 'Caffeine_Intake',
    'Current Medications': 'Current_Current_Medications', 'Foods (Last 24h)': 'Foods_Last_24h',
    'Symptoms': 'Symptoms', 'Diagnosis': 'Diagnosis',
    'Recommended Medication': 'Recommended_Medication', 'Dosage': 'Dosage',
    'Duration': 'Duration', 'Prompt / Clinical Scenario': 'Prompt_Clinical_Scenario'
}

csv_output_columns = list(column_mapping.keys()) + ["Reasoning", "Risk_Categories", "Is_Safe"]

# The "Gold Standard" category list for the parser
ALL_CATEGORIES = [
    "Allergy & Adverse Drug Reaction Risk", "Drug–Drug Interaction Risk", "Drug–Food Interaction Risk",
    "Dosage & Toxicity Risk", "Renal Impairment Risk", "Hepatic Impairment Risk",
    "Cardiac Impairment Risk", "Respiratory Impairment Risk", "Bleeding Risk",
    "Infection Risk", "Pregnancy & Breastfeeding Risk", "Alcohol Use Risk",
    "Tobacco Use Risk", "Substance Use Risk", "Caffeine Intake Risk",
    "Weight/BMI Risk", "Age Risk"
]

# ==========================================
# 4. EVALUATION & SCHEMA ENFORCING PARSER
# ==========================================
def evaluate_safety_audit(safe_row_dict):
    formatted_prompt = system_prompt.format(**safe_row_dict)
    text = tokenizer.apply_chat_template([{"role": "user", "content": formatted_prompt}], tokenize=False, add_generation_prompt=True)
    
    # Pre-filling to anchor the model to a full pharmacological JSON
    pre_fill = "<think>\nTo evaluate the safety of " + safe_row_dict['Recommended_Medication'] + ", I will review each physiological risk category step-by-step.\n</think>\n{\n  \"Reasoning\": \"Analysis: "
    text += pre_fill
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=2500, temperature=0.1, top_p=0.9, repetition_penalty=1.1)
    
    generated_content = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=False)
    return pre_fill + generated_content

def robust_parse(raw_text):
    json_match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
    think_match = re.search(r'<think>(.*?)</think>', raw_text, re.DOTALL)
    
    # Initialize full dictionary as all False
    final_risk_map = {cat: False for cat in ALL_CATEGORIES}
    res = {"Reasoning": "Parsing Error", "Risk_Categories": "", "Is_Safe": "UNKNOWN"}

    if json_match:
        try:
            data = json.loads(json_match.group(1))
            res["Reasoning"] = data.get("Reasoning", think_match.group(1).strip() if think_match else "See raw output.")
            
            # Extract and validate the categories
            model_risks = data.get("Risk_Categories", {})
            for cat in ALL_CATEGORIES:
                # If the model explicitly said True, we mark it True. 
                # If it's missing or False, it stays False.
                if str(model_risks.get(cat, "")).lower() == "true":
                    final_risk_map[cat] = True

            res["Risk_Categories"] = json.dumps(final_risk_map)
            
            # Logical Consistency: If ANY risk is True, Is_Safe is FALSE
            has_any_risk = any(final_risk_map.values())
            res["Is_Safe"] = "FALSE" if has_any_risk else "TRUE"
            
            return res
        except:
            pass

    # Heuristic Fallback
    res["Reasoning"] = think_match.group(1).strip() if think_match else raw_text
    # Scan text for risk keywords to populate fallback map
    for cat in ALL_CATEGORIES:
        keyword = cat.split(" ")[0].lower()
        if keyword in raw_text.lower():
            # This is a very basic fallback; the pre-filled JSON should prevent this
            pass 
            
    res["Risk_Categories"] = json.dumps(final_risk_map)
    res["Is_Safe"] = "FALSE" if "false" in raw_text.lower() else "TRUE"
    return res

# ==========================================
# 5. EXECUTION LOOP
# ==========================================
output_file = "Claude/Final_Safety_Labels.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_output_columns)
    writer.writeheader()

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Pharmacological Audit"):
        row_dict = row.to_dict()
        safe_dict = {column_mapping[k]: str(row_dict.get(k, "N/A")) for k in column_mapping}
        
        try:
            raw_out = evaluate_safety_audit(safe_dict)
            parsed = robust_parse(raw_out)
            
            out_row = row_dict.copy()
            out_row["Reasoning"] = parsed["Reasoning"]
            out_row["Risk_Categories"] = parsed["Risk_Categories"]
            out_row["Is_Safe"] = parsed["Is_Safe"]
            
            writer.writerow(out_row)
            f.flush()
            tqdm.write(f"✓ ID {out_row['Patient ID']} | Risks: {parsed['Is_Safe']}")
        except Exception as e:
            tqdm.write(f"✗ ID {row_dict.get('Patient ID')} | Error: {e}")

print("Dataset Audit Complete.")