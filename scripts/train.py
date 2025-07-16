import os
import numpy as np
import pandas as pd # New import for data handling
import kagglehub # New import for downloading from Kaggle
from datasets import load_dataset, DatasetDict, Dataset # Ensure Dataset and DatasetDict are imported
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import mlflow
import mlflow.transformers

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "google/flan-t5-base"
DATASET_NAME = "tmdb/tmdb-movie-metadata" # Updated to the Kaggle dataset ID
OUTPUT_DIR = "RESULTS"
MODEL_SAVE_PATH = "MODELS"

MFLOW_SERVER_IP = "34.251.243.175"
if MFLOW_SERVER_IP is None:
    raise ValueError("MFLOW_SERVER_IP environment variable is not set. Please set it to your MLflow server's public IP or ensure it's in your .env file.")

MLFLOW_TRACKING_URI = f"http://{MFLOW_SERVER_IP}/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Global tokenizer for tokenize_function to access
tokenizer = None

# Load Dataset - Now uses kagglehub to download and process the TMDB dataset
def _load_dataset(dataset_name_param: str) -> DatasetDict:
    print(f"Downloading and loading dataset from Kaggle: {dataset_name_param}")
    try:
        # Download the dataset using kagglehub
        # This will download the dataset to a local cache managed by kagglehub
        path = kagglehub.dataset_download(dataset_name_param)
        print(f"Dataset downloaded to: {path}")

        # Construct the full path to the movies CSV file within the downloaded dataset
        movies_csv_file = os.path.join(path, "tmdb_5000_movies.csv")
        # credits_csv_file = os.path.join(path, "tmdb_5000_credits.csv") # Not strictly needed for this task

        if not os.path.exists(movies_csv_file):
            raise FileNotFoundError(f"Expected file not found: {movies_csv_file}")

        # Load the movies CSV into a pandas DataFrame
        df_movies = pd.read_csv(movies_csv_file)
        print(f"Loaded {len(df_movies)} movie entries from {movies_csv_file}.")
        print("Columns available:", df_movies.columns.tolist())

        # Select relevant columns ('overview' for description, 'title' for movie title)
        # and handle missing values
        if 'overview' not in df_movies.columns or 'title' not in df_movies.columns:
            raise ValueError("Required columns 'overview' and 'title' not found in the dataset. Please check the CSV structure.")

        # Drop rows where either 'overview' or 'title' is missing, as they are essential
        df_processed = df_movies[['overview', 'title']].dropna().reset_index(drop=True)
        # Rename 'overview' to 'description' to match the tokenize_function's expectation
        df_processed = df_processed.rename(columns={'overview': 'description'})
        df_processed = df_processed.head(5)

        print(f"Processed DataFrame has {len(df_processed)} entries after cleaning.")
        print(f"Sample processed data (first 2 entries):\n{df_processed.head(2)}")

        # Convert the pandas DataFrame to a Hugging Face Dataset
        hf_dataset = Dataset.from_pandas(df_processed)

        # Create train and validation splits (e.g., 80% train, 20% validation)
        # Using Dataset.train_test_split for a simple split
        train_test_split = hf_dataset.train_test_split(test_size=0.2, seed=42)
        raw_datasets = DatasetDict({
            'train': train_test_split['train'],
            'validation': train_test_split['test'] # Using 'test' split as 'validation'
        })

        print("Dataset loaded, processed, and split successfully.")
        print(raw_datasets)
        return raw_datasets

    except Exception as e:
        print(f"Error loading dataset from Kaggle: {e}")
        raise


def load_tokenizer(model_name: str):
    print(f"Loading tokenizer for Model: {model_name}")
    global tokenizer # Access the global tokenizer variable
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully.")
    return tokenizer

def load_model(model_name: str):
    print(f"Loading model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Model loaded successfully.")
    return model

def tokenize_function(examples):
    # This function expects 'description' for input and 'title' for target.
    # It adds a task prefix "generate title: " which is common for T5 models.

    if "description" not in examples or "title" not in examples:
        raise ValueError("Dataset must contain 'description' and 'title' columns for tokenization.")

    # Prepend a task prefix for T5 models
    input_texts = [f"generate title: {desc}" for desc in examples["description"]]
    target_texts = examples["title"]

    # Tokenize input sequences (description)
    model_inputs = tokenizer(input_texts, max_length=512, truncation=True)

    # Tokenize target sequences (title) for the labels (decoder input IDs)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_texts, max_length=128, truncation=True) # Max length for titles

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# For a text generation task, standard classification metrics (accuracy, f1) are not suitable.
# You would typically use metrics like ROUGE or BLEU. This is a placeholder.
def compute_metrics(eval_pred):
    # Implement more sophisticated generation metrics (e.g., ROUGE, BLEU) here
    # Requires decoding predictions and labels, and then using a metric library like 'evaluate'.
    print("Compute metrics for text generation is not fully implemented. Using default Trainer evaluation.")
    return {} # Return empty dictionary for now


if __name__ == "__main__":
    # Load the dataset using the updated function
    raw_datasets = _load_dataset(DATASET_NAME)

    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME)

    # Apply tokenization to the raw_datasets (which is a DatasetDict: 'train', 'validation')
    # `remove_columns` removes the original 'description' and 'title' columns,
    # leaving 'input_ids', 'attention_mask', and 'labels'.
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names # Assumes 'train' split exists
    )

    # Ensure the datasets are in PyTorch tensor format for Trainer
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"] # Use the 'validation' split from the split dataset

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    # Configure Training Arguments
    print("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8, # Reduced batch size for larger T5 models
        per_device_eval_batch_size=8,  # Reduced batch size for larger T5 models
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", # Changed from f1 for generation task
        report_to="none",
        # predict_with_generate=True, # Important for generation models during evaluation
        # generation_max_length=128, # Max length for generated titles
        fp16=False, # Set to True if you have a compatible GPU and want to use mixed precision
    )

    # Data Collator for Seq2Seq models
    # This pads inputs and labels to the longest sequence in a batch
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer, # Pass tokenizer to trainer for internal use (e.g., generation)
        data_collator=data_collator, # Add data collator for dynamic padding
        compute_metrics=None, # Set to None for now, will need custom generation metric
    )

    # Train the Model
    print("Starting model training (this might take a while)...")
    trainer.train()
    print("Model training finished.")

    # Evaluate the model and log metrics
    print("Evaluating the model and logging metrics...")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        mlflow.log_metric(key, value)
    print("Metrics logged successfully.")

    # Save the Model locally before logging to MLflow
    print(f"Saving fine-tuned model and tokenizer to {MODEL_SAVE_PATH}...")
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("Model and tokenizer saved locally.")

    # Log the Hugging Face model to MLflow
    print("Logging model to MLflow...")
    mlflow.transformers.log_model(
        transformers_model=MODEL_SAVE_PATH,
        name="huggingface-model",
        tokenizer=tokenizer,
        task="text2text-generation",
        registered_model_name="MovieTitleGeneratorFlanT5"
    )
    print("Model logged to MLflow successfully.")

print("Training script completed and MLflow run finished.")