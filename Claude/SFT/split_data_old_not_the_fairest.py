import pandas as pd
import numpy as np
import os

# 1. Define file paths
input_csv = 'Claude/Knowledge_Distillation/Claude_Personalized_Groundtruth_New_Data_Distill.csv'
meds_file = 'new_medications.txt' 
output_dir = 'Claude/SFT/new_data_splits1'

# 2. Load and clean the medication list from the text file
with open(meds_file, 'r') as f:
    # Read lines, strip whitespace/newlines, and ignore any empty lines
    base_medications = [line.strip() for line in f if line.strip()]

# 3. Shuffle the base medications list
np.random.seed(42)
np.random.shuffle(base_medications)

# 4. Calculate split indices for the 35 medications
total_meds = len(base_medications)
train_end = int(total_meds * 0.75)
val_end = train_end + int(total_meds * 0.10)

# Assign the base medications to their splits
train_meds = base_medications[:train_end]
val_meds = base_medications[train_end:val_end]
test_meds = base_medications[val_end:]

# 5. Helper function for substring matching
def get_split(csv_med_name, train_list, val_list, test_list):
    if not isinstance(csv_med_name, str):
        return 'unmatched'
    
    csv_med_name_lower = csv_med_name.lower()
    
    # Check if the base medication name is contained within the CSV row's medication string
    for med in train_list:
        if med.lower() in csv_med_name_lower: return 'train'
    for med in val_list:
        if med.lower() in csv_med_name_lower: return 'val'
    for med in test_list:
        if med.lower() in csv_med_name_lower: return 'test'
        
    return 'unmatched'

# 6. Load and initially shuffle the dataset
df = pd.read_csv(input_csv)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 7. Map each row to a split based on the substring match
df['split_group'] = df['Recommended Medication'].apply(
    lambda x: get_split(x, train_meds, val_meds, test_meds)
)

# Optional safety check to catch any rows that didn't match the 35 provided medications
unmatched_count = (df['split_group'] == 'unmatched').sum()
if unmatched_count > 0:
    print(f"Warning: {unmatched_count} rows did not contain any medication name from {meds_file}.")
    # Uncomment the next line if you want to drop unmatched rows entirely
    # df = df[df['split_group'] != 'unmatched']

# 8. Separate the dataframe into the three final splits and drop the helper column
train_df = df[df['split_group'] == 'train'].drop(columns=['split_group'])
val_df = df[df['split_group'] == 'val'].drop(columns=['split_group'])
test_df = df[df['split_group'] == 'test'].drop(columns=['split_group'])

# 9. Create the output directory and save the files
os.makedirs(output_dir, exist_ok=True)
train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

# 10. Print summaries and assigned medications
print(f"Splitting complete based on {total_meds} root medications.\n")

print(f"--- TRAIN SET ({len(train_meds)} unique base meds, {len(train_df)} rows) ---")
print(", ".join(sorted(train_meds)))
print()

print(f"--- VAL SET ({len(val_meds)} unique base meds, {len(val_df)} rows) ---")
print(", ".join(sorted(val_meds)))
print()

print(f"--- TEST SET ({len(test_meds)} unique base meds, {len(test_df)} rows) ---")
print(", ".join(sorted(test_meds)))
print()