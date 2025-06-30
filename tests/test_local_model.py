# test_local_model.py
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random

# Define the local path to your model directory
LOCAL_MODEL_PATH = "models/distractor_generator_t5_production"

# Check if CUDA is available locally (it might just be CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading production model from local path: {LOCAL_MODEL_PATH}")
print(f"Using device: {device}")

# Load the model and tokenizer from the local directory
try:
    model = T5ForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH).to(device)
    tokenizer = T5Tokenizer.from_pretrained(LOCAL_MODEL_PATH)
    print("✅ Model loaded successfully!")
except OSError:
    print(f"❌ ERROR: Cannot find model at '{LOCAL_MODEL_PATH}'. Did you unzip the file correctly?")
    exit()

# The same generation function from Colab
def generate_distractor(question, correct_answer):
    if correct_answer.isdigit():
        return str(int(correct_answer) + random.choice([-2, -1, 1, 2]))

    input_text = f"generate distractor: question: {question} answer: <ANS> {correct_answer} </ANS>"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    bad_words_ids = [tokenizer(correct_answer, add_special_tokens=False).input_ids]

    outputs = model.generate(
        input_ids,
        max_length=60,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        no_repeat_ngram_size=2,
        bad_words_ids=bad_words_ids
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# Test it
q1 = "How does behaviorism differ from psychoanalytic theory?"
a1 = "behaviorism suggests that behavior is learned through environmental stimuli, while psychoanalytic theory posits that behavior is driven by unconscious desires and conflicts"
distractor = generate_distractor(q1, a1)
print(f"\nGenerated Distractor: {distractor}")