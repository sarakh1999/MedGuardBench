import pandas as pd

# Load your dataset
df = pd.read_csv("Claude/new_dataset/Check_Leakage/New_Claude_Personalized_Groundtruth_Data.csv")

# Calculate BMI from weight and height
df["Calculated_BMI"] = df["Weight (kg)"] / ((df["Height (cm)"] / 100) ** 2)

# Compare with provided BMI (allow small rounding tolerance)
tolerance = 0.1  # BMI values are usually rounded to 1 decimal place

df["BMI_Correct"] = (
    abs(df["BMI"] - df["Calculated_BMI"]) <= tolerance
)

# Display rows with incorrect BMI
incorrect_bmi = df[df["BMI_Correct"] == False]

print(f"Total samples: {len(df)}")
print(f"Incorrect BMI samples: {len(incorrect_bmi)}")

if len(incorrect_bmi) > 0:
    print("\nRows with incorrect BMI:")
    print(
        incorrect_bmi[
            [
                "Age (year)",
                "Gender",
                "Weight (kg)",
                "Height (cm)",
                "BMI",
                "Calculated_BMI"
            ]
        ]
    )
else:
    print("All BMI values are correct.")

# Optional: save the checked dataset
df.to_csv("dataset_bmi_checked.csv", index=False)