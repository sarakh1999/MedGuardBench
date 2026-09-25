import pandas as pd
import json

# Load dataset
df = pd.read_csv("Claude/new_dataset/Check_Leakage/New_Claude_Personalized_Groundtruth_Data.csv")

expected_risk_categories = [
    "Allergy & Adverse Drug Reaction Risk",
    "Drug-Drug Interaction Risk",
    "Drug-Food Interaction Risk",
    "Dosage & Toxicity Risk",
    "Renal Impairment Risk",
    "Hepatic Impairment Risk",
    "Cardiac Impairment Risk",
    "Respiratory Impairment Risk",
    "Bleeding Risk",
    "Infection Risk",
    "Pregnancy & Breastfeeding Risk",
    "Alcohol Use Risk",
    "Tobacco Use Risk",
    "Substance Use Risk",
    "Caffeine Intake Risk",
    "Weight/BMI Risk",
    "Age Risk"
]


def check_risk_categories(value):
    reasons = []

    try:
        # Convert JSON string to dictionary
        if isinstance(value, str):
            value = json.loads(value)

        # Check dictionary type
        if not isinstance(value, dict):
            reasons.append("Risk_Categories is not a dictionary")
            return False, "; ".join(reasons)

        # Check number of categories
        if len(value) != 17:
            reasons.append(f"Expected 17 categories but found {len(value)}")

        # Check missing categories
        missing = set(expected_risk_categories) - set(value.keys())
        if missing:
            reasons.append(f"Missing categories: {list(missing)}")

        # Check extra categories
        extra = set(value.keys()) - set(expected_risk_categories)
        if extra:
            reasons.append(f"Extra categories: {list(extra)}")

        # Check boolean values
        non_boolean = [
            k for k, v in value.items()
            if not isinstance(v, bool)
        ]
        if non_boolean:
            reasons.append(f"Non-boolean values in categories: {non_boolean}")

        if reasons:
            return False, "; ".join(reasons)

        return True, ""

    except json.JSONDecodeError:
        return False, "Invalid JSON format"

    except Exception as e:
        return False, f"Error parsing Risk_Categories: {str(e)}"


# Run validation
results = df["Risk_Categories"].apply(check_risk_categories)

df["Risk_Categories_Valid"] = results.apply(lambda x: x[0])
df["Reason"] = results.apply(lambda x: x[1])


# Get only wrong samples
wrong_samples = df[~df["Risk_Categories_Valid"]].copy()

# Print only Patient ID and reason
for _, row in wrong_samples.iterrows():
    print(f"Patient ID: {row['Patient ID']} | Reason: {row['Reason']}")


# Save only required 3 columns
wrong_samples_output = wrong_samples[
    ["Patient ID", "Risk_Categories", "Reason"]
]

wrong_samples_output.to_csv(
    "wrong_risk_categories.csv",
    index=False
)

print(f"\nSaved {len(wrong_samples_output)} wrong samples to wrong_risk_categories.csv")