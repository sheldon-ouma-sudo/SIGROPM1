import os
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, DatasetDict, Dataset
import evaluate
import nltk

# Download required NLTK data
nltk.download("punkt")

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Set environment variables for SageMaker paths
    train_data_path = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/model")
    logs_dir = os.environ.get("SM_LOG_DIR", "/opt/ml/logs")
    
    logger.info("Environment paths set successfully.")
    logger.info(f"Train data path: {train_data_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Logs directory: {logs_dir}")

    # Load the dataset
    try:
        dataset = load_dataset("json", data_files={"train": os.path.join(train_data_path, "training_data.jsonl")})
        logger.info("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Data cleaning
    def clean_data(example):
        example["prompt"] = example.get("prompt", "").strip()
        example["squad"] = example.get("squad", "").strip()
        return example

    dataset = dataset.map(clean_data)
    logger.info("Data cleaned successfully.")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    
    def preprocess_function(example):
        model_inputs = tokenizer(
            example["prompt"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        labels = tokenizer(
            example["squad"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    logger.info(f"Tokenized dataset size: {len(tokenized_dataset['train'])}")

    # Split the dataset
    def split_dataset(dataset, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
        """
        Splits the dataset into train, validation, and test sets.
        """
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        validation_size = int(total_size * validation_ratio)

        train_data = dataset.select(range(0, train_size))
        validation_data = dataset.select(range(train_size, train_size + validation_size))
        test_data = dataset.select(range(train_size + validation_size, total_size))

        return train_data, validation_data, test_data

    try:
        train_data, val_data, test_data = split_dataset(tokenized_dataset["train"])
        dataset_dict = DatasetDict({
            "train": train_data,
            "validation": val_data,
            "test": test_data,
        })
        logger.info(f"Data split into training ({len(dataset_dict['train'])}), "
                    f"validation ({len(dataset_dict['validation'])}), "
                    f"and test ({len(dataset_dict['test'])}) sets.")
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise

    # Save processed data
    processed_data_path = os.path.join(output_dir, "processed_data")
    try:
        os.makedirs(processed_data_path, exist_ok=True)
        dataset_dict.save_to_disk(processed_data_path)
        logger.info(f"Processed data saved to {processed_data_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise

    # Initialize model
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    logger.info("Model initialized successfully.")

    # Define training arguments
    try:
        import tensorboard
        report_to = ["tensorboard"]
    except ImportError:
        logger.warning("TensorBoard is not installed. Skipping TensorBoard logging.")
        report_to = []

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=500,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        logging_dir=logs_dir,
        report_to=report_to,  # Use dynamic reporting
    )
    logger.info("Training arguments defined successfully.")

    # Define trainer with metrics
    rouge_metric = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {key: value.mid.fmeasure * 100 for key, value in result.items()}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

    # Evaluate the model
    try:
        eval_results = trainer.evaluate(eval_dataset=dataset_dict["test"])
        logger.info(f"Evaluation results: {eval_results}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
