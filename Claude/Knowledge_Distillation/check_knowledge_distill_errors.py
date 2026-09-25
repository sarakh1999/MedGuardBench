import csv
import os

# Path to the file you want to inspect
FILE_TO_CHECK = "Claude/Knowledge_Distillation/Claude_Personalized_Groundtruth_Distill.csv"


def check_and_list_error_patient_ids():
    if not os.path.exists(FILE_TO_CHECK):
        print(f"Error: The file '{FILE_TO_CHECK}' does not exist.")
        return

    total_records = 0
    true_count = 0
    false_count = 0
    error_rows = []

    with open(FILE_TO_CHECK, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Check if necessary columns exist
        headers = reader.fieldnames
        if "Trace_Valid" not in headers or "Patient ID" not in headers:
            print(
                "Error: 'Trace_Valid' or 'Patient ID' column missing from headers."
            )
            return

        for row in reader:
            total_records += 1
            status = str(row.get("Trace_Valid", "")).strip().upper()
            patient_id = row.get("Patient ID", "UNKNOWN_ID")
            note = row.get("Validation_Note", "No note provided")

            if status == "TRUE":
                true_count += 1
            else:
                false_count += 1
                error_rows.append((patient_id, note))

    # Print Summary Table
    print("=" * 50)
    print(f"SUMMARY FOR: {FILE_TO_CHECK}")
    print("=" * 50)
    print(f"Total Rows Scored : {total_records}")
    print(f"Total TRUE Rows   : {true_count}")
    print(f"Total FALSE Rows  : {false_count}")
    print("=" * 50)

    # Print Patient IDs with Errors
    if error_rows:
        print(f"\nPATIENT IDS WITH VALIDATION ERRORS ({len(error_rows)} total):")
        print("-" * 50)
        print(f"{'Patient ID':<20} | {'Validation Failure Reason'}")
        print("-" * 50)
        for pid, reason in error_rows:
            print(f"{pid:<20} | {reason}")
        print("-" * 50)
    else:
        print("\n No validation errors found! All rows are TRUE.")


if __name__ == "__main__":
    check_and_list_error_patient_ids()