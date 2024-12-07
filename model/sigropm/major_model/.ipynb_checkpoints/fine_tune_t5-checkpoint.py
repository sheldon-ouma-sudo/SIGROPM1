import os
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_from_disk, load_metric

def tokenize_data(example, tokenizer, max_seq_length):
    """
    Tokenization function to preprocess data for T5.
    """
    inputs = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=max_seq_length)
    targets = tokenizer(example["target_text"], truncation=True, padding="max_length", max_length=max_seq_length)
    inputs["labels"] = targets["input_ids"]
    return inputs

def fine_tune_t5(args):
    """
    Fine-tunes the T5 model with the given arguments.
    """
    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk(args.data_path)
    
    # Load tokenizer and model
    print(f"Loading model {args.model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        lambda x: tokenize_data(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=["input_text", "target_text"]
    )
    
    # Define evaluation metric
    rouge = load_metric("rouge")
    
    def compute_metrics(pred):
        """
        Compute ROUGE scores for evaluation.
        """
        preds = pred.predictions
        labels = pred.label_ids
        
        # Decode predictions and references
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {key: value.mid.fmeasure for key, value in result.items()}
    
    # Define training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=500,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        save_total_limit=2,
        predict_with_generate=True,
        report_to=["none"],  # Disable default reporting
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Fine-tune model
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Model fine-tuning completed!")

if __name__ == "__main__":
    # Argument parser for hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the preprocessed dataset.")
    parser.add_argument("--output_dir", type=str, default="./t5-fine-tuned", help="Path to save the fine-tuned model.")
    parser.add_argument("--model_name", type=str, default="t5-small", help="Pretrained T5 model to fine-tune.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length for tokenization.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")
    args = parser.parse_args()
    
    fine_tune_t5(args)
