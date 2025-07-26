# scripts/generate_review_sheet.py
# A simple, assumption-free script to generate distractors for human review.
# This script does NOT look for a 'choices' column.

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
from tqdm import tqdm

# --- Configuration (using files we KNOW exist) ---
MODEL_PATH = Path("models/distractor_generator_t5_production")
QUESTIONS_FILE = Path("data/2_processed_data/normalized_questions.parquet")
OUTPUT_FILE = Path("t5_qualitative_review_sheet.csv")
SAMPLE_SIZE = 100

def generate_distractors(model, tokenizer, question, answer, device, num_distractors=3):
    """Generates distractors using the fine-tuned T5 model."""
    input_text = f"generate distractors: <ANS> {answer} </ANS> <CTX> {question} </CTX>"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    outputs = model.generate(
        **inputs,
        max_length=64,
        num_return_sequences=num_distractors,
        do_sample=True,
        top_p=0.95,
        temperature=0.9,
        no_repeat_ngram_size=2
    )
    
    distractors = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [d for d in distractors if d.lower().strip() != answer.lower().strip()]

if __name__ == "__main__":
    print("--- Generating Distractor Review Sheet (The Simple Way) ---")

    # 1. Check for required files
    if not MODEL_PATH.exists() or not QUESTIONS_FILE.exists():
        print(f"❌ FATAL: Missing required files.")
        print(f"   Check for model at: '{MODEL_PATH}'")
        print(f"   Check for data at: '{QUESTIONS_FILE}'")
        exit()

    # 2. Load model and tokenizer
    print(f"✅ Loading model from '{MODEL_PATH}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    # 3. Load questions and take a random sample
    print(f"✅ Loading questions from '{QUESTIONS_FILE}'...")
    df = pd.read_parquet(QUESTIONS_FILE)
    sample_df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"✅ Took a random sample of {SAMPLE_SIZE} questions.")

    # 4. Generate distractors and collect results
    results = []
    print("\n⚙️  Generating distractors...")
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        generated = generate_distractors(model, tokenizer, row['question'], row['answer'], device)
        # Pad with empty strings if fewer than 3 distractors were generated
        while len(generated) < 3:
            generated.append("")
            
        results.append({
            "question": row['question'],
            "correct_answer": row['answer'],
            "generated_distractor_1": generated[0],
            "generated_distractor_2": generated[1],
            "generated_distractor_3": generated[2],
            "Plausibility_Rating (1-5)": "",
            "Notes": ""
        })

    # 5. Save the results to a CSV file
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)

    print("\n--- ✅ SUCCESS ---")
    print(f"Review sheet with {SAMPLE_SIZE} generated examples has been saved to:")
    print(f"--> {OUTPUT_FILE}")