import pandas as pd
import os
import argparse

# --- Configuration ---
TRAINING_DATA_FILE = "data/training_sets/distractor_generation_training_data.parquet"

def verify_data(file_path: str, num_samples: int):
    """
    Loads the generated training data and prints a random sample
    for human verification.
    """
    print("--- Starting Training Data Verification ---")
    
    # 1. Validate file exists
    if not os.path.exists(file_path):
        print(f"\n❌ FATAL: Training data file not found at '{file_path}'.")
        print("Please run generate_distractor_training_set.py first.")
        return

    print(f"Loading data from '{file_path}'...")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"\n❌ FATAL: Could not read parquet file. Error: {e}")
        return

    # 2. Take a random sample for review
    if num_samples > len(df):
        print(f"Warning: Requested {num_samples} samples, but dataset only has {len(df)}. Showing all.")
        num_samples = len(df)
        
    sample_df = df.sample(n=num_samples, random_state=42)

    print(f"\nDisplaying {num_samples} random examples for your review:")
    print("-" * 80)

    # 3. Print samples in a readable format
    for i, row in sample_df.iterrows():
        print(f"\n--- Example {i+1}/{num_samples} ---")
        print(f"\n[QUESTION]:")
        print(f"  {row['question']}")
        print(f"\n  [CORRECT ANSWER]:")
        print(f"    {row['correct_answer']}")
        print(f"\n  [GENERATED DISTRACTOR (is this a good distractor?)]:")
        print(f"    {row['distractor']}")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spot-check the quality of the auto-generated distractor training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--file",
        default=TRAINING_DATA_FILE,
        help="Path to the training data .parquet file."
    )
    parser.add_argument(
        "-n", "--num_samples",
        type=int,
        default=5,
        help="The number of random samples to display for verification."
    )

    args = parser.parse_args()
    
    verify_data(file_path=args.file, num_samples=args.num_samples)