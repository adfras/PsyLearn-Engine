import os
import random
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import T5ForConditionalGeneration, T5Tokenizer

LOCAL_MODEL_PATH = "models/distractor_generator_t5_production"

def generate_distractor(question: str, correct_answer: str, model, tokenizer, device="cpu") -> str:
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
        bad_words_ids=bad_words_ids,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

@pytest.mark.skipif(not os.path.exists(LOCAL_MODEL_PATH), reason="Local model not available")
def test_local_model_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = T5ForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH).to(device)
    tokenizer = T5Tokenizer.from_pretrained(LOCAL_MODEL_PATH)
    distractor = generate_distractor(
        "How does behaviorism differ from psychoanalytic theory?",
        "behaviorism suggests that behavior is learned through environmental stimuli, while psychoanalytic theory posits that behavior is driven by unconscious desires and conflicts",
        model,
        tokenizer,
        device,
    )
    assert isinstance(distractor, str) and len(distractor) > 0
