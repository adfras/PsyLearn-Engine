# src/reasoning_engine/prepare_data.py

import pandas as pd
import numpy as np
from pathlib import Path
import json
import ast
from tqdm import tqdm
import zipfile
import io
import re

tqdm.pandas()

# Define paths
RAW_DATA_ROOT = Path("data/1_raw_source_data")
PROCESSED_DATA_DIR = Path("data/2_processed_data")
OUTPUT_FILE = PROCESSED_DATA_DIR / "normalized_questions.parquet"

# --- DATA PROCESSING FUNCTIONS FOR EACH SOURCE ---

def process_gsm8k(path: Path) -> pd.DataFrame:
    """Processes GSM8K parquet files, extracting the final numerical answer."""
    dfs = []
    for f in path.glob("*.parquet"):
        df = pd.read_parquet(f)
        df['answer'] = df['answer'].str.split('#### ').str[-1].str.strip()
        df['source'] = 'gsm8k'
        dfs.append(df[['question', 'answer', 'source']])
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def process_proofwriter(path: Path) -> pd.DataFrame:
    """Parses ProofWriter's complex JSONL format using regex."""
    records = []
    for f in path.glob("*.jsonl"):
        with open(f, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    # We need to handle both 'en' and 'ro' for question/answer
                    en_string = data.get('translation', {}).get('en', '')
                    ro_string = data.get('translation', {}).get('ro', '')
                    
                    q_match = re.search(r'\$question\$\s*=\s*(.*?)\s*;', en_string)
                    c_match = re.search(r'\$context\$\s*=\s*(.*)', en_string)
                    # The answer is in the 'ro' string for some reason
                    a_match = re.search(r'\$answer\$\s*=\s*(\w+)', ro_string)

                    if q_match and c_match and a_match:
                        full_question = f"[CONTEXT]\n{c_match.group(1).strip()}\n\n[QUESTION]\nIs the following statement true or false?\n'{q_match.group(1).strip()}'"
                        records.append({'question': full_question, 'answer': a_match.group(1).strip(), 'source': 'proofwriter'})
                except (json.JSONDecodeError, AttributeError): 
                    continue
    return pd.DataFrame(records)

def process_thoughtsource(zip_path: Path) -> pd.DataFrame:
    """Unpacks the deeply nested JSON from the ThoughtSource zip archive."""
    if not zip_path.exists(): return pd.DataFrame()
    records = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Assumes the first file in the zip is the one we want
        json_filename = next((name for name in z.namelist() if name.endswith('.json')), None)
        if not json_filename: return pd.DataFrame()

        with z.open(json_filename) as f:
            data = json.load(f)
            # The structure is {train/test: {dataset_name: [items]}}
            for split_data in data.values():
                if not isinstance(split_data, dict): continue
                for dataset_name, dataset_list in split_data.items():
                    if isinstance(dataset_list, list):
                        for item in dataset_list:
                            # Check for 'answer' existence, as some items are just questions
                            if isinstance(item, dict) and 'question' in item and 'answer' in item:
                                records.append({'question': item['question'], 'answer': item['answer'], 'source': dataset_name})
    return pd.DataFrame(records)

def process_bioasq(zip_path: Path) -> pd.DataFrame:
    """Processes all JSON files within the BioASQ zip archive."""
    if not zip_path.exists(): return pd.DataFrame()
    records = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_info in z.infolist():
            if file_info.filename.endswith('.json') and not file_info.is_dir():
                with z.open(file_info) as f:
                    data = json.load(f)
                    for q_data in data.get('questions', []):
                        question, snippets = q_data.get('body'), q_data.get('snippets', [])
                        if question and snippets:
                            # The answer is formed by concatenating context snippets
                            answer = " ".join([s.get('text', '') for s in snippets])
                            if answer: records.append({'question': question, 'answer': answer, 'source': 'bioasq'})
    return pd.DataFrame(records)

def process_pubmedqa(path: Path) -> pd.DataFrame:
    """Processes PubMedQA CSVs, filtering for 'yes'/'no' answers."""
    dfs = []
    for f in path.glob("*labeled.csv"): # Only process labeled data
        df = pd.read_csv(f)
        if 'question' in df.columns and 'long_answer' in df.columns and 'final_decision' in df.columns:
            # We only want questions with definitive answers for training
            df = df[df['final_decision'] == 'yes']
            df = df.rename(columns={'long_answer': 'answer'})
            df['source'] = 'pubmedqa'
            dfs.append(df[['question', 'answer', 'source']])
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def process_mentat_csv(file_path: Path) -> pd.DataFrame:
    """Correctly processes MENTAT CSV, finding the correct answer from multiple choices."""
    if not file_path.exists(): return pd.DataFrame()
    df = pd.read_csv(file_path)
    answer_cols = ['answer_a', 'answer_b', 'answer_c', 'answer_d', 'answer_e']
    
    def find_correct_answer(row):
        try:
            # creator_truth is a stringified list like '[1.0, 0.0, 0.0, 0.0, 0.0]'
            truth_list = ast.literal_eval(row['creator_truth'])
            correct_index = truth_list.index(1.0)
            return row[answer_cols[correct_index]]
        except (ValueError, TypeError, SyntaxError): 
            return None
            
    df['answer'] = df.apply(find_correct_answer, axis=1)
    # Use the 'text_male' as the base question template
    df = df.rename(columns={'text_male': 'question'})
    df['source'] = 'mentat'
    return df[['question', 'answer', 'source']].dropna()

def process_psychology_qa(path: Path) -> pd.DataFrame:
    """Processes custom psychology JSON and Parquet files."""
    dfs = []
    boltmonkey_file = path / "boltmonkey.json"
    if boltmonkey_file.exists():
        df_bolt = pd.read_json(boltmonkey_file)
        df_bolt['source'] = 'boltmonkey'
        dfs.append(df_bolt)
        
    gragroo_file = path / "gragroo_train.parquet"
    if gragroo_file.exists():
        df_grag = pd.read_parquet(gragroo_file)
        # Unpack the conversational format
        def extract_qa(c): 
            return (
                next((m['value'] for m in c if m['from'] == 'human'), None),
                next((m['value'] for m in c if m['from'] == 'gpt'), None)
            )
        pairs = df_grag['conversations'].apply(extract_qa)
        df_proc = pd.DataFrame(pairs.tolist(), columns=['question', 'answer']).dropna()
        if not df_proc.empty: 
            df_proc['source'] = 'gragroo'
            dfs.append(df_proc)
            
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# --- Main Data Preparation Orchestrator ---

def prepare_all_data():
    print("--- Starting Data Ingestion for Question Bank ---")
    sources_to_process = {
        "GSM8K": (process_gsm8k, RAW_DATA_ROOT / "1_chain_of_thought/GSM8K"),
        "ProofWriter": (process_proofwriter, RAW_DATA_ROOT / "1_chain_of_thought/ProofWriter"),
        "ThoughtSource": (process_thoughtsource, RAW_DATA_ROOT / "1_chain_of_thought/ThoughtSource/ThoughtSource-open-data-snapshot.zip"),
        "BioASQ": (process_bioasq, RAW_DATA_ROOT / "2_scholarly_qa/BioASQ/BIOASQ_JSON.zip"),
        "PubMedQA": (process_pubmedqa, RAW_DATA_ROOT / "2_scholarly_qa/PubMedQA"),
        "MENTAT_CSV": (process_mentat_csv, RAW_DATA_ROOT / "2_scholarly_qa/MENTAT/final_dataset_raw_questions.csv"),
        "PsychologyQA": (process_psychology_qa, RAW_DATA_ROOT / "3_psychology_qa"),
    }
    
    all_dfs = []
    print("\n--- Processing Individual Data Sources ---")
    for name, (func, path) in sources_to_process.items():
        print(f"-> Processing: {name}...")
        try:
            df = func(path)
            if not df.empty and {'question', 'answer', 'source'}.issubset(df.columns):
                all_dfs.append(df)
                print(f"  [SUCCESS] Found {len(df)} records.")
            else:
                print(f"  [WARNING] No valid records were extracted.")
        except Exception as e:
            print(f"  [FATAL ERROR] An exception occurred: {e.__class__.__name__}: {e}")

    if not all_dfs:
        print("\nError: No data was successfully processed. Halting.")
        return

    print("\n--- Combining All Processed Sources ---")
    final_df = pd.concat(all_dfs, ignore_index=True).dropna(subset=['question', 'answer'])
    print(f"Total records combined: {len(final_df)}")
    
    print("\nPerforming final cleaning...")
    final_df['question'] = final_df['question'].astype(str)
    final_df['answer'] = final_df['answer'].astype(str)
    final_df = final_df[final_df['question'].str.len() > 20] # Filter out very short/trivial questions
    final_df = final_df[final_df['answer'].str.len() > 1]
    final_df.drop_duplicates(subset=['question'], inplace=True, keep='first')
    final_df = final_df.reset_index(drop=True)

    print(f"\nSaving normalized question bank to {OUTPUT_FILE}...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUTPUT_FILE, index=False)
    
    print("\n--- Data Preparation Summary ---")
    print(f"Successfully saved {len(final_df)} unique records to '{OUTPUT_FILE}'.")
    print("Final source distribution:")
    print(final_df['source'].value_counts())
    print("\nQuestion Bank generation complete.")

if __name__ == "__main__":
    prepare_all_data()