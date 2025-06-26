# src/reasoning_engine/prepare_data.py

import pandas as pd
import numpy as np
from pathlib import Path
import json
import ast # For safely parsing stringified Python objects
from langdetect import detect, LangDetectException
from tqdm import tqdm
import zipfile
import io

tqdm.pandas()

# Define paths based on your folder structure
RAW_DATA_ROOT = Path("data/1_raw_source_data")
PROCESSED_DATA_DIR = Path("data/2_processed_data")
OUTPUT_FILE = PROCESSED_DATA_DIR / "normalized_questions.parquet"

# --- Helper function for robust text file reading ---
def read_file_content(file_handle) -> str:
    """Reads file content from a handle, trying utf-8 then falling back to latin-1."""
    try:
        return file_handle.read().decode('utf-8')
    except UnicodeDecodeError:
        file_handle.seek(0) # Reset pointer before re-reading
        return file_handle.read().decode('latin-1', errors='ignore')

# --- DATA PROCESSING FUNCTIONS ---

def process_pubmedqa(path: Path) -> pd.DataFrame:
    print(f"Processing PubMedQA from: {path.resolve()}")
    dfs = []
    for f in path.glob("*.csv"):
        with open(f, 'rb') as file_handle:
            content = read_file_content(file_handle)
            df = pd.read_csv(io.StringIO(content))
            
            if 'question' in df.columns and 'long_answer' in df.columns:
                # Safely parse the 'context' column if it exists
                if 'context' in df.columns:
                    def safe_literal_eval(val):
                        try: return ast.literal_eval(val)
                        except (ValueError, SyntaxError, TypeError): return val
                    df['context_parsed'] = df['context'].apply(safe_literal_eval)

                df = df.rename(columns={'long_answer': 'answer'})
                df['source'] = 'pubmedqa'
                dfs.append(df[['question', 'answer', 'source']])
                
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def process_mentat_mcq(file_path: Path) -> pd.DataFrame:
    print(f"Processing MENTAT from: {file_path.resolve()}")
    if not file_path.exists(): return pd.DataFrame()
    
    with open(file_path, 'rb') as file_handle:
        content = read_file_content(file_handle)
        df = pd.read_csv(io.StringIO(content))
        
    answer_cols = ['answer_a', 'answer_b', 'answer_c', 'answer_d', 'answer_e']
    
    def find_correct_answer(row):
        try:
            # Safely parse the stringified list
            truth_list = ast.literal_eval(row['creator_truth'])
            correct_index = truth_list.index(1.0)
            return row[answer_cols[correct_index]]
        except: return None
            
    df['answer'] = df.apply(find_correct_answer, axis=1)
    df = df.rename(columns={'text_male': 'question'})
    df['source'] = 'mentat'
    return df[['question', 'answer', 'source']].dropna()

# Note: Other processing functions (GSM8K, etc.) are assumed to be simple and are omitted for brevity,
# but would be included here in a full script, using the `read_file_content` helper for robustness.
# We will focus on the ones that required fixes.

# --- Main Data Preparation Orchestrator ---

def prepare_all_data():
    """
    Ingests and normalizes ALL data sources, handling identified issues.
    This function generates the primary question bank for the Reasoning Engine.
    """
    print("--- Starting Data Ingestion for Question Bank ---")
    
    # We only process data sources that fit the static Question/Answer format.
    # Student log data (ASSISTments, KDD) is handled by a separate pipeline for the Proactive Tutor.
    question_bank_sources = [
        # Example of a simple source
        # process_gsm8k(RAW_DATA_ROOT / "1_chain_of_thought/GSM8K"), 
        
        # Sources with confirmed issues that are now handled
        process_pubmedqa(RAW_DATA_ROOT / "2_scholarly_qa/PubMedQA"),
        process_mentat_mcq(RAW_DATA_ROOT / "2_scholarly_qa/MENTAT/final_dataset_raw_questions.csv"),
        
        # Add other Q&A sources here (boltmonkey, gragroo, etc.)
    ]
    
    # Exclude student log data from this pipeline.
    print("\nNOTE: Student interaction logs (ASSISTments, KDD Cup) are NOT included in this question bank.")
    print("They contain valuable data for student modeling, but not in a static Q&A format.")
    print("They should be processed by the feature engineering pipeline for the Proactive Tutor.")

    # Filter out any sources that returned an empty DataFrame
    all_dfs = [df for df in question_bank_sources if not df.empty]
    print(f"\nSuccessfully processed {len(all_dfs)} data sources for the question bank.")
    
    if not all_dfs:
        print("Error: No Q&A data was successfully processed. Halting.")
        return

    print("Combining all sources into a single dataset...")
    df = pd.concat(all_dfs, ignore_index=True).dropna(subset=['question', 'answer'])
    print(f"Total records combined: {len(df)}")
    
    print("\nPerforming final cleaning and standardization...")
    df['question'] = df['question'].astype(str)
    df['answer'] = df['answer'].astype(str)
    df = df[(df['question'].str.len() > 10) & (df['answer'].str.len() > 0)]
    
    # Final language check (optional but good practice)
    # def is_english(text): ...
    # df = df[df['question'].progress_apply(is_english)]

    df.reset_index(drop=True, inplace=True)
    
    print(f"\nSaving normalized question bank to {OUTPUT_FILE}...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE)
    
    print("\n--- Data Preparation Summary ---")
    print(f"Successfully saved {len(df)} records to '{OUTPUT_FILE}'.")
    print("Source distribution:")
    print(df['source'].value_counts())
    print("\nQuestion Bank generation complete.")

if __name__ == "__main__":
    # For a complete run, we would add calls to all the other processing functions
    # inside prepare_all_data(). This example focuses on demonstrating the fixes.
    prepare_all_data()