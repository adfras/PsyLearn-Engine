import os
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

# --- Configuration ---------------------------------------------------------

TRAINING_DATA_REPO = "adfras/psychology-distractor-data"
BASE_MODEL          = "t5-small"
OUTPUT_MODEL_DIR    = "/data/distractor_generator_t5_small"   # survives restarts

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# --- Main ------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Fine-tuning {BASE_MODEL} on {TRAINING_DATA_REPO}")

    # 1  Load and subsample the dataset directly on the Hub object
    hf_dataset = (
        load_dataset(TRAINING_DATA_REPO, split="train")
        .shuffle(seed=42)
        .select(range(min(10_000, len(load_dataset(TRAINING_DATA_REPO, split='train')))))
    )
    print(f"Loaded {len(hf_dataset)} examples")

    # 2  Tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)
    model     = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)

    # 3  Pre-processing
    def preprocess(examples):
        prefix   = "generate distractor: "
        inputs   = [
            f"{prefix}question: {q} answer: {a}"
            for q, a in zip(examples["question"], examples["correct_answer"])
        ]
        model_in = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=examples["distractor"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        model_in["labels"] = labels["input_ids"]
        return model_in

    tokenized_dataset = hf_dataset.map(
        preprocess,
        batched=True,
        remove_columns=hf_dataset.column_names,
    )

    # 4  Data collator (avoids double padding)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

    # 5  Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=2,
    )

    # 6  Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting fine-tuning…")
    trainer.train()

    # 7  Save final artefacts
    trainer.save_model(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"Model saved to {OUTPUT_MODEL_DIR}")

