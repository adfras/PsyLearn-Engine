# create_golden_set.py
# This script should be run only ONCE to create the permanent benchmark dataset.

import pandas as pd
import os

# --- Configuration ---
PROCESSED_DATA_FILE = "data/processed/ALL_PSYCHOLOGY_DATA_normalized.parquet"
TESTS_DIR = "tests"
GOLDEN_SET_PATH = os.path.join(TESTS_DIR, "golden_test_set.parquet")
SAMPLE_SIZE = 1000

# --- Main Logic ---
if __name__ == "__main__":
    print("--- Creating Golden Test Set ---")

    # 1. Ensure the tests directory exists
    os.makedirs(TESTS_DIR, exist_ok=True)
    
    # 2. Check if the source data file exists
    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"FATAL: Source data file not found at '{PROCESSED_DATA_FILE}'.")
        print("Please run normalize_psych_data.py first.")
    
    # 3. Check if the golden set already exists
    elif os.path.exists(GOLDEN_SET_PATH):
        print(f"INFO: Golden test set already exists at '{GOLDEN_SET_PATH}'. No action taken.")
        print("If you need to recreate it, please delete the old file first.")
    
    # 4. Create the file if it's missing
    else:
        try:
            print(f"Loading source data from '{PROCESSED_DATA_FILE}'...")
            df = pd.read_parquet(PROCESSED_DATA_FILE)
            
            print(f"Taking a fixed, random sample of {SAMPLE_SIZE} rows...")
            # Using random_state=42 ensures the sample is the same every time
            golden_set = df.sample(n=SAMPLE_SIZE, random_state=42)
            
            print(f"Saving golden test set to '{GOLDEN_SET_PATH}'...")
            golden_set.to_parquet(GOLDEN_SET_PATH, index=False)
            
            print("\nSUCCESS: Golden test set created successfully.")
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")