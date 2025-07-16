import os
import pandas as pd
import kagglehub
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import mlflow
import mlflow.transformers
import evaluate # New import for evaluation metrics
import numpy as np
import nltk # For sentence tokenization if needed by ROUGE
import torch

# Ensure nltk data is available for ROUGE (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # Corrected: Catch LookupError when resource is not found
    print("NLTK 'punkt' tokenizer data not found. Downloading...")
    try:
        nltk.download('punkt')
        print("NLTK 'punkt' tokenizer data downloaded successfully.")
    except Exception as e: # Catch a more general exception for download errors
        print(f"Failed to download NLTK 'punkt' tokenizer data: {e}")
        print("Please try running 'python -c \"import nltk; nltk.download(\'punkt\')\"' manually.")
        raise # Re-raise the exception as it's critical for evaluation


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
MFLOW_SERVER_IP = os.getenv('MFLOW_SERVER_IP')
if MFLOW_SERVER_IP is None:
    raise ValueError("MFLOW_SERVER_IP environment variable is not set. Please set it to your MLflow server's public IP or ensure it's in your .env file.")
MLFLOW_TRACKING_URI = f"http://{MFLOW_SERVER_IP}/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

REGISTERED_MODEL_NAME = "MovieTitleGeneratorFlanT5"
DATASET_NAME = "tmdb/tmdb-movie-metadata" # Added DATASET_NAME here

# --- Global Variables ---
tokenizer = None # Will be loaded with the model

# --- Data Loading Function (similar to train.py, but for test set) ---
def _load_test_dataset(dataset_name_param: str) -> Dataset:
    print(f"Downloading and loading test dataset from Kaggle: {dataset_name_param}")
    try:
        path = kagglehub.dataset_download(dataset_name_param)
        movies_csv_file = os.path.join(path, "tmdb_5000_movies.csv")
        credits_csv_file = os.path.join(path, "tmdb_5000_credits.csv")

        if not os.path.exists(movies_csv_file) or not os.path.exists(credits_csv_file):
            raise FileNotFoundError("TMDB movie or credits CSV files not found after download.")

        df_movies = pd.read_csv(movies_csv_file)
        df_credits = pd.read_csv(credits_csv_file)

        df_credits.rename(columns={'title': 'credit_title', 'movie_id': 'id'}, inplace=True)
        movies_df = df_movies.merge(df_credits, on='id')

        movies_df.dropna(subset=['overview', 'title'], inplace=True)
        movies_df.reset_index(drop=True, inplace=True)

        # Rename 'overview' to 'description' to match the tokenize_function's expectation
        df_processed = movies_df[['overview', 'title']].rename(columns={'overview': 'description'})
        
        # Take a subset for quick testing, e.g., the last 100 entries for evaluation
        # For full evaluation, you'd use a dedicated test split or the full processed data.
        df_processed = df_processed.tail(100) # Using last 100 for evaluation
        
        hf_dataset = Dataset.from_pandas(df_processed)
        print(f"Loaded {len(hf_dataset)} test examples.")
        print(f"Sample test data (first 2 entries):\n{hf_dataset.to_pandas().head(2)}")
        return hf_dataset

    except Exception as e:
        print(f"Error loading test dataset from Kaggle: {e}")
        raise

# --- Tokenization Function (same as in train.py) ---
def tokenize_function(examples):
    if "description" not in examples or "title" not in examples:
        raise ValueError("Dataset must contain 'description' and 'title' columns for tokenization.")

    input_texts = [f"generate title: {desc}" for desc in examples["description"]]
    target_texts = examples["title"]

    model_inputs = tokenizer(input_texts, max_length=512, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_texts, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- Main Evaluation Logic ---
if __name__ == "__main__":
    # Start an MLflow run for evaluation
    with mlflow.start_run(run_name="Model Evaluation"):
        # Log parameters related to evaluation setup
        mlflow.log_param("model_evaluated", REGISTERED_MODEL_NAME)
        mlflow.log_param("evaluation_dataset_source", DATASET_NAME)
        mlflow.log_param("evaluation_subset_size", 100) # Documenting the subset size

        print("Loading tokenizer and model for evaluation...")
        try:
            # Load the latest model from MLflow Model Registry
            model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
            loaded_model_components = mlflow.transformers.load_model(model_uri)
            model = loaded_model_components.model # Access the raw model from the pipeline
            tokenizer = loaded_model_components.tokenizer # Access the raw tokenizer from the pipeline
            
            # Ensure model is in evaluation mode
            model.eval()

            # Move model to GPU if available
            if torch.cuda.is_available():
                model.to("cuda")
                print("Model moved to GPU for evaluation.")
            else:
                print("No GPU found, model will run on CPU for evaluation.")

            print("Model and tokenizer loaded for evaluation.")
        except Exception as e:
            print(f"Failed to load model for evaluation: {e}")
            mlflow.log_param("evaluation_status", "Failed - Model Load Error")
            raise RuntimeError(f"Could not load model for evaluation: {e}")

        # Load and tokenize the test dataset
        test_dataset = _load_test_dataset(DATASET_NAME)
        tokenized_test_dataset = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=test_dataset.column_names
        )
        tokenized_test_dataset.set_format("torch")
        print(f"Tokenized test dataset size: {len(tokenized_test_dataset)}")

        # Initialize Data Collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # Initialize Hugging Face Trainer for evaluation only
        # We only need the evaluation loop, not training.
        eval_args = TrainingArguments(
            output_dir="./eval_results", # Temporary directory for evaluation outputs
            do_train=False,
            do_eval=True,
            per_device_eval_batch_size=8,
            report_to="none",
            fp16=False,
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=tokenized_test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=None
        )

        print("Starting prediction on test dataset...")
        predictions = trainer.predict(tokenized_test_dataset)
        print("Prediction complete.")

        # Extract predictions and labels
        # When predict_with_generate=True, predictions.predictions directly contains the generated token IDs
        preds = predictions.predictions
        labels = predictions.label_ids

        # Decode generated predictions and true labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in labels as it's used for padding and should be ignored by decode
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute ROUGE scores
        print("Computing ROUGE metrics...")
        rouge = evaluate.load("rouge")
        # ROUGE expects lists of strings
        rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        # Log ROUGE metrics to MLflow
        print("Logging ROUGE metrics to MLflow:")
        for key, value in rouge_results.items():
            # ROUGE scores are often between 0 and 1, multiply by 100 for percentage
            mlflow.log_metric(f"rouge_{key}", value * 100)
            print(f"  rouge_{key}: {value * 100:.2f}")

        # You can also log the raw predictions and labels as artifacts for debugging
        mlflow.log_text("\n".join(decoded_preds), "predictions.txt")
        mlflow.log_text("\n".join(decoded_labels), "references.txt")

        mlflow.log_param("evaluation_status", "Success")
        print("Evaluation script completed and metrics logged to MLflow.")
