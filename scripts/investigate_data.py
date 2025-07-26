# investigate_data.py
# A robust script to find and analyze rows in the dataset based on column length.

"""Utility to explore dataset rows based on column length."""

import importlib
import sys
import argparse
import os

REQUIRED_PACKAGES = ["pandas", "pyarrow"]

for pkg in REQUIRED_PACKAGES:
    if importlib.util.find_spec(pkg) is None:
        sys.exit(
            f"Missing required package '{pkg}'. Install dependencies via 'pip install -r requirements.txt'."
        )

import pandas as pd

# --- Configuration ---
DEFAULT_DATA_FILE = "data/processed/ALL_PSYCHOLOGY_DATA_normalized.parquet"
DEFAULT_COLUMN = "question"
DEFAULT_THRESHOLD = 10

def investigate_column_length(file_path: str, column_name: str, threshold: int, comparison: str):
    """
    Loads a parquet file and prints rows where the length of a specified column
    is less than or greater than a given threshold.
    
    Args:
        file_path (str): The path to the parquet data file.
        column_name (str): The name of the column to investigate.
        threshold (int): The length threshold to check against.
        comparison (str): Either 'less' or 'greater'.
    """
    # --- 1. Input Validation ---
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'")
        return

    if comparison not in ['less', 'greater']:
        print(f"Error: Invalid comparison type '{comparison}'. Must be 'less' or 'greater'.")
        return

    print(f"--- Starting Investigation ---")
    print(f"File:        {file_path}")
    print(f"Column:      {column_name}")
    print(f"Threshold:   {threshold}")
    print(f"Condition:   Length is {comparison} than {threshold}")
    print("----------------------------\n")

    # --- 2. Data Loading and Analysis ---
    try:
        df = pd.read_parquet(file_path)

        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in the dataset.")
            print(f"Available columns are: {list(df.columns)}")
            return
            
        # Ensure the column is of string type for .str accessor
        df[column_name] = df[column_name].astype(str)
        
        # Apply the filter based on the comparison type
        if comparison == 'less':
            filtered_df = df[df[column_name].str.len() < threshold]
        else: # comparison == 'greater'
            filtered_df = df[df[column_name].str.len() > threshold]

        # --- 3. Reporting Results ---
        num_found = len(filtered_df)
        if num_found > 0:
            print(f"SUCCESS: Found {num_found} rows meeting the condition.\n")
            # Display relevant columns for context
            display_columns = [column_name, 'source', 'answer']
            # Ensure display columns exist before trying to show them
            valid_display_columns = [col for col in display_columns if col in df.columns]
            
            with pd.option_context('display.max_rows', None, 'display.max_colwidth', 100):
                print(filtered_df[valid_display_columns])
        else:
            print("SUCCESS: Found 0 rows meeting the specified condition.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # --- 4. Command-Line Interface (CLI) Setup ---
    parser = argparse.ArgumentParser(
        description="Investigate a dataset by finding rows with specific column lengths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )
    
    parser.add_argument(
        "--file",
        default=DEFAULT_DATA_FILE,
        help="Path to the .parquet data file to investigate."
    )
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN,
        help="The column whose length you want to check."
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help="The character length threshold."
    )
    parser.add_argument(
        "--comparison",
        choices=['less', 'greater'],
        default='less',
        help="Set to 'less' to find rows shorter than the threshold, or 'greater' for longer."
    )

    args = parser.parse_args()
    
    investigate_column_length(
        file_path=args.file,
        column_name=args.column,
        threshold=args.threshold,
        comparison=args.comparison
    )
