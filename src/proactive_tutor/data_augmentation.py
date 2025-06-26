import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm

# --- Configuration ---
DATA_FILE = "data/processed/psychology_data_with_embeddings.parquet"
OUTPUT_DIR = "data/training_sets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "distractor_generation_training_data.parquet")

NUM_SAMPLES_TO_GENERATE = 50000 
SIMILARITY_MIN = 0.3
SIMILARITY_MAX = 0.7
# Process in batches to balance speed and memory usage.
# This size is safe for most standard computers.
BATCH_SIZE = 1000 

if __name__ == "__main__":
    print("--- Starting FAST & SAFE Automated Training Set Generation ---")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print(f"FATAL: Data file with embeddings not found at '{DATA_FILE}'. Please run compute_embeddings.py first.")
    else:
        print("Loading data and embeddings...")
        df = pd.read_parquet(DATA_FILE)
        
        if NUM_SAMPLES_TO_GENERATE > len(df):
            NUM_SAMPLES_TO_GENERATE = len(df)

        df_sample = df.sample(n=NUM_SAMPLES_TO_GENERATE, random_state=42)

        embedding_cols = [col for col in df.columns if col.startswith('embed_')]
        all_embeddings = df[embedding_cols].values
        
        training_records = []
        num_batches = int(np.ceil(len(df_sample) / BATCH_SIZE))

        print(f"Processing {len(df_sample)} samples in {num_batches} batches of size {BATCH_SIZE}...")

        for i in tqdm(range(num_batches)):
            # Get the current batch of questions
            batch_start = i * BATCH_SIZE
            batch_end = (i + 1) * BATCH_SIZE
            batch_df = df_sample.iloc[batch_start:batch_end]
            batch_embeddings = batch_df[embedding_cols].values
            
            # --- FAST MATRIX OPERATION ---
            # Calculate similarity for the entire batch at once.
            # This creates a small, temporary matrix (e.g., 1000 x 400k)
            sim_matrix_batch = cosine_similarity(batch_embeddings, all_embeddings)
            
            # Now, iterate through the results for this small batch
            for j in range(len(batch_df)):
                scores = sim_matrix_batch[j]
                
                candidate_indices = np.where((scores > SIMILARITY_MIN) & (scores < SIMILARITY_MAX))[0]
                
                # Get the original index of the current question
                current_question_original_index = batch_df.index[j]
                candidate_indices = candidate_indices[candidate_indices != current_question_original_index]
                
                if len(candidate_indices) > 0:
                    distractor_idx = np.random.choice(candidate_indices)
                else:
                    sorted_indices = np.argsort(scores)
                    fallback_choice = sorted_indices[int(len(sorted_indices) * 0.3)]
                    distractor_idx = fallback_choice if fallback_choice != current_question_original_index else sorted_indices[int(len(sorted_indices) * 0.3) + 1]

                training_records.append({
                    'question': batch_df.iloc[j]['question'],
                    'correct_answer': batch_df.iloc[j]['answer'],
                    'distractor': df.iloc[distractor_idx]['answer']
                })

        print("\nConstructing final training set from processed records...")
        training_data = pd.DataFrame(training_records)
        
        training_data.to_parquet(OUTPUT_FILE, index=False)
        
        print("\n--- SUCCESS ---")
        print(f"Automatically generated a training set with {len(training_data)} examples.")
        print(f"File saved to: '{OUTPUT_FILE}'")
        print("\nFirst 5 examples:")
        print(training_data.head())