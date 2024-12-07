import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_from_disk

# Load preprocessed dataset
data_path = "./data/processed_dataset"
dataset = load_from_disk(data_path)

# Load model and tokenizer
model_name = "t5-small"  # You can use "t5-base" or other variations
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenization function
def tokenize_data(example):
    inputs = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(example["target_text"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_data, batched=True, remove_columns=["input_text", "target_text"])

# Set up Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    report_to=["none"],  # Disable default reporting
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save final model
model.save_pretrained("./t5-custom")
tokenizer.save_pretrained("./t5-custom")
