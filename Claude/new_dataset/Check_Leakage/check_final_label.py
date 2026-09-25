import pandas as pd
import json

# Load dataset
df = pd.read_csv("Claude/new_dataset/Check_Leakage/New_Claude_Personalized_Groundtruth_Data.csv")


def check_safety_consistency(row):
    try:
        risk_categories = row["Risk_Categories"]
        is_safe = row["Is_Safe"]

        # Convert JSON string to dictionary
        if isinstance(risk_categories, str):
            risk_categories = json.loads(risk_categories)

        # Count if any risk is True
        has_risk = any(risk_categories.values())

        # Rule:
        # - All risks False -> Is_Safe should be True
        # - At least one risk True -> Is_Safe should be False
        if not has_risk and is_safe != True:
            return False, "All Risk_Categories are false but Is_Safe is not True"

        if has_risk and is_safe != False:
            return False, "At least one Risk_Category is true but Is_Safe is not False"

        return True, ""

    except Exception as e:
        return False, f"Error checking row: {str(e)}"


# Apply validation
results = df.apply(check_safety_consistency, axis=1)

df["Safety_Check_Valid"] = results.apply(lambda x: x[0])
df["Reason"] = results.apply(lambda x: x[1])


# Extract incorrect samples
wrong_samples = df[~df["Safety_Check_Valid"]].copy()


# Print only Patient ID and reason
for _, row in wrong_samples.iterrows():
    print(f"Patient ID: {row['Patient ID']} | Reason: {row['Reason']}")


# Save only incorrect samples with 3 columns
wrong_samples_output = wrong_samples[
    ["Patient ID", "Risk_Categories", "Reason"]
]

wrong_samples_output.to_csv(
    "wrong_Is_Safe_consistency.csv",
    index=False
)

print(f"\nFound {len(wrong_samples_output)} inconsistent samples.")
print("Saved to wrong_Is_Safe_consistency.csv")