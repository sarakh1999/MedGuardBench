import pandas as pd
import toml
import json
import os
from vllm import SamplingParams
from server_llm import ServerLLM, load_url_from_log_file

# ==============================
# SETTINGS
# ==============================
MODEL_NAME = "meta-llama/Meta-Llama-Guard-2-8B"
SERVER_LOG = "server_logs.txt"
DATA_PATH = "Claude/Claude_Personalized_Train_Data - Sheet1.csv"
OUTPUT_PATH = "Claude/Final_Safety_Labels.csv"
PROMPT_PATH = "Claude/personalized_guardrail_general_use_models.toml"

def main():
    # 1. Load Prompt Template
    config = toml.load(PROMPT_PATH)
    prompt_template = config.get("system_prompt", "")

    # 2. Load Data
    df = pd.read_csv(DATA_PATH)

    # 3. Explicitly Map your CSV Columns to Prompt Placeholders
    column_mapping = {
        'Weight (kg)': 'Weight', 'Height (cm)': 'Height',
        'Genetic Disorders': 'Genetic_Disorders', 'Chronic Conditions': 'Chronic_Conditions',
        'Pregnancy / Breastfeeding': 'Pregnancy_Breastfeeding', 'Drug Allergies': 'Drug_Allergies',
        'Renal Impairment': 'Renal_Impairment', 'Hepatic Impairment': 'Hepatic_Impairment',
        'Cardiac Impairment': 'Cardiac_Impairment', 'Respiratory Impairment': 'Respiratory_Impairment',
        'Alcohol Use': 'Alcohol_Use', 'Tobacco Use': 'Tobacco_Use',
        'Substance Use': 'Substance_Use', 'Caffeine Intake': 'Caffeine_Intake',
        'Current Medications': 'Current_Medications', 'Foods (Last 24h)': 'Foods_Last_24h',
        'Prompt / Clinical Scenario': 'Prompt_Clinical_Scenario', 'Recommended Medication': 'Recommended_Medication'
    }
    df = df.rename(columns=column_mapping)

    # 4. Setup Server Client
    base_url = load_url_from_log_file(SERVER_LOG)
    llm = ServerLLM(base_url=base_url, model=MODEL_NAME, num_workers=10)
    params = SamplingParams(temperature=0.0, max_tokens=2048)

    all_labels = []
    all_reasons = []
    all_risks = [] # New list for risk categories

    # 5. Build Prompts
    messages_list = []
    for _, row in df.iterrows():
        clean_row = {k: ("None" if pd.isna(v) else v) for k, v in row.to_dict().items()}
        prompt = prompt_template.format(**clean_row)
        messages_list.append([
            {"role": "system", "content": "You are a clinical pharmacist. Output JSON only."},
            {"role": "user", "content": prompt}
        ])

    # 6. Generate Responses
    print(f"Starting inference for {len(df)} records...")
    responses = llm.generate(messages_list, params)

    for resp in responses:
        try:
            data = json.loads(resp.outputs[0].text)
            all_labels.append(str(data.get("Is_Safe")).upper())
            all_reasons.append(data.get("Reasoning"))
            # Save the Risk_Categories dictionary as a JSON string
            all_risks.append(json.dumps(data.get("Risk_Categories", {})))
        except Exception:
            all_labels.append("PARSE_ERROR")
            all_reasons.append(resp.outputs[0].text)
            all_risks.append("{}")

    # 7. Final Save
    df['Is_Safe'] = all_labels
    df['Risk_Categories'] = all_risks # Adding the new column
    df['Reasoning'] = all_reasons
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Success! Data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()