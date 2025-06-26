# compute_embeddings.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import torch # Import the torch library

# --- Configuration ---
DATA_FILE = "data/processed/ALL_PSYCHOLOGY_DATA_normalized.parquet"
OUTPUT_FILE = "data/processed/psychology_data_with_embeddings.parquet"
# This model is small, fast, and effective for sentence-level tasks.
MODEL_NAME = 'all-MiniLM-L6-v2' 

if __name__ == "__main__":
    print("--- Starting Embedding Computation ---")

    # --- THIS IS THE FIX: Explicitly set the device to GPU ---
    # Check if a CUDA-enabled GPU is available, otherwise fall back to CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cpu':
        print("WARNING: GPU not found. Computation will be very slow.")
    # --- END OF FIX ---

    if not os.path.exists(DATA_FILE):
        print(f"FATAL: Data file not found at {DATA_FILE}. Please run normalize_psych_data.py first.")
    else:
        df = pd.read_parquet(DATA_FILE)

        print(f"Loading sentence-transformer model: '{MODEL_NAME}'...")
        # Pass the device to the model when you load it
        model = SentenceTransformer(MODEL_NAME, device=device)

        print(f"Computing embeddings for {len(df)} questions... (This may take a while)")
        
        # The .encode() method will now automatically run on the specified device (GPU)
        embeddings = model.encode(df['question'].tolist(), show_progress_bar=True)

        # The embedding is a 384-dimensional vector for this model.
        # We'll store it as separate columns in the dataframe.
        embedding_df = pd.DataFrame(embeddings, index=df.index)
        embedding_df = embedding_df.add_prefix('embed_')

        # Combine the original dataframe with the new embedding columns
        df_with_embeddings = pd.concat([df, embedding_df], axis=1)

        print(f"Saving new dataframe with embeddings to '{OUTPUT_FILE}'...")
        df_with_embeddings.to_parquet(OUTPUT_FILE)

        print("\nSUCCESS: Embeddings computed and saved.")
        print(f"New dataframe shape: {df_with_embeddings.shape}")