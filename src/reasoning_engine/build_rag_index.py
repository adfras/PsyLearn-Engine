# src/reasoning_engine/build_rag_index.py

import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch

# --- Configuration ---
# Correctly pointing to the output of our data quality step
DATA_FILE = Path("data/2_processed_data/normalized_questions.parquet")
OUTPUT_FILE = Path("data/2_processed_data/questions_with_embeddings.parquet")
MODEL_NAME = 'all-MiniLM-L6-v2' 

def compute_embeddings():
    print("--- Starting Embedding Computation ---")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cpu':
        print("WARNING: GPU not found. See options below for accelerating this process.")

    if not DATA_FILE.exists():
        print(f"FATAL: Data file not found at {DATA_FILE}. Please run prepare_data.py first.")
        return

    df = pd.read_parquet(DATA_FILE)

    print(f"Loading sentence-transformer model: '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print(f"Computing embeddings for {len(df)} questions... (This may take a while)")
    embeddings = model.encode(df['question'].tolist(), show_progress_bar=True)

    embedding_df = pd.DataFrame(embeddings, index=df.index)
    embedding_df = embedding_df.add_prefix('embed_')
    df_with_embeddings = pd.concat([df, embedding_df], axis=1)

    print(f"Saving new dataframe with embeddings to '{OUTPUT_FILE}'...")
    df_with_embeddings.to_parquet(OUTPUT_FILE)

    print("\nSUCCESS: Embeddings computed and saved.")
    print(f"New dataframe shape: {df_with_embeddings.shape}")
    print(f"Output file: {OUTPUT_FILE}")

if __name__ == "__main__":
    compute_embeddings()