import os
import pandas as pd
import kagglehub
import mlflow
import mlflow.transformers
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import nltk
import json
import boto3
from botocore.exceptions import ClientError
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# Ensure nltk 'punkt' is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- AWS Secrets Manager Configuration ---
# You must set these environment variables before running the script.
AWS_REGION = os.getenv('AWS_REGION') or "eu-west-1"
DATASET_NAME = "tmdb/tmdb-movie-metadata"
# Define the names for the different secrets
SECRETS_MANAGER_LTS_NAME = os.getenv('SECRETS_MANAGER_LTS_NAME') or "lts-model-versions"
SECRETS_MANAGER_LATEST_NAME = os.getenv('SECRETS_MANAGER_LATEST_NAME') or "latest-model-versions"

# Define the keys expected in the secrets
LTS_VERSION_KEY = "lts_model_version"
LATEST_VERSION_KEY = "latest_model_version"

MFLOW_SERVER_URL = os.getenv('MFLOW_SERVER_URL')
REGISTERED_MODEL_NAME = os.getenv('REGISTERED_MODEL_NAME') or "MovieTitleGeneratorFlanT5"

if MFLOW_SERVER_URL is None:
    raise ValueError("MFLOW_SERVER_URL environment variable is not set. Please set it to your MLflow server's public IP or ensure it's in your .env file.")

if REGISTERED_MODEL_NAME is None:
    raise ValueError("REGISTERED_MODEL_NAME environment variable is not set in your .env file.")

# Set up MLflow tracking
mlflow.set_tracking_uri(MFLOW_SERVER_URL)
mlflow_client = MlflowClient()

# Initialize AWS clients
try:
    secrets_manager_client = boto3.client('secretsmanager', region_name=AWS_REGION)
except ClientError as e:
    print(f"Error initializing AWS Secrets Manager client: {e}")
    secrets_manager_client = None

def get_secret_value(secret_name):
    """Retrieves a secret's value from AWS Secrets Manager."""
    if not secrets_manager_client:
        return None
    try:
        get_secret_value_response = secrets_manager_client.get_secret_value(SecretId=secret_name)
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            return json.loads(secret)
    except ClientError as e:
        print(f"Error retrieving secret '{secret_name}': {e}")
        return None

def update_secret_value(secret_name, new_value):
    """Updates a secret's value in AWS Secrets Manager."""
    if not secrets_manager_client:
        return
    try:
        response = secrets_manager_client.update_secret(
            SecretId=secret_name,
            SecretString=json.dumps(new_value)
        )
        print(f"Successfully updated secret '{secret_name}'.")
        return response
    except ClientError as e:
        print(f"Error updating secret '{secret_name}': {e}")
        return None

def get_mlflow_run_id(model_name: str, model_version: str):
    """
    Finds the MLflow run ID associated with a specific model version.
    """
    try:
        versions = mlflow_client.search_model_versions(f"name='{model_name}'")
        
        target_version = None
        for v in versions:
            if v.version == model_version:
                target_version = v
                break
        
        if not target_version or not target_version.run_id:
            raise ValueError(f"Model version '{model_version}' not found or has no run_id in MLflow registry.")

        return target_version.run_id
    except MlflowException as e:
        print(f"MLflow API error: {e}")
        raise
    except Exception as e:
        print(f"Error retrieving run_id from MLflow: {e}")
        raise

def get_mlflow_metric(run_id: str, metric_name: str):
    """
    Retrieves a specific metric from the MLflow run.
    Returns None if the metric is not found.
    """
    try:
        run = mlflow_client.get_run(run_id)
        metric_value = run.data.metrics.get(metric_name)
        if metric_value is None:
            print(f"Metric '{metric_name}' not found for MLflow run '{run_id}'.")
        return metric_value
    except MlflowException as e:
        print(f"MLflow API error while getting run '{run_id}': {e}")
    except Exception as e:
        print(f"Error retrieving metric from MLflow for run '{run_id}': {e}")
    return None

# --- Main Evaluation Logic ---
print("Fetching LTS and latest model versions from Secrets Manager...")
# Fetch the LTS model version from Secrets Manager
lts_secret = get_secret_value(SECRETS_MANAGER_LTS_NAME)
if not lts_secret or LTS_VERSION_KEY not in lts_secret:
    raise ValueError(f"Could not retrieve a valid '{LTS_VERSION_KEY}' from secret '{SECRETS_MANAGER_LTS_NAME}'.")
LTS_MODEL_VERSION = lts_secret[LTS_VERSION_KEY]

# Fetch the latest model version from Secrets Manager
latest_secret = get_secret_value(SECRETS_MANAGER_LATEST_NAME)
if not latest_secret or LATEST_VERSION_KEY not in latest_secret:
    raise ValueError(f"Could not retrieve a valid '{LATEST_VERSION_KEY}' from secret '{SECRETS_MANAGER_LATEST_NAME}'.")
LATEST_MODEL_VERSION = latest_secret[LATEST_VERSION_KEY]

print(f"LTS Model Version from Secrets Manager: {LTS_MODEL_VERSION}")
print(f"Latest Model Version from Secrets Manager: {LATEST_MODEL_VERSION}")

# --- Fetch ROUGE-L scores from MLflow for both models ---
print(f"Fetching ROUGE-L score for LTS version '{LTS_MODEL_VERSION}' from MLflow...")
lts_run_id = get_mlflow_run_id(REGISTERED_MODEL_NAME, LTS_MODEL_VERSION)
LTS_ROUGE_L_SCORE = None
if lts_run_id:
    LTS_ROUGE_L_SCORE = get_mlflow_metric(lts_run_id, "rouge_rougeL")

# Load the LATEST model & tokenizer from MLflow for evaluation
model_uri = f"models:/{REGISTERED_MODEL_NAME}/{LATEST_MODEL_VERSION}"
print(f"Loading latest model from URI: {model_uri}")
try:
    loaded = mlflow.transformers.load_model(model_uri)
    model = loaded.model
    tokenizer = loaded.tokenizer
except MlflowException as e:
    raise ValueError(f"Failed to load latest model '{LATEST_MODEL_VERSION}' from MLflow: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load test data from Kaggle
def load_test_data():
    path = kagglehub.dataset_download(DATASET_NAME)
    movies = pd.read_csv(os.path.join(path, "tmdb_5000_movies.csv"))
    credits = pd.read_csv(os.path.join(path, "tmdb_5000_credits.csv"))
    credits.rename(columns={'title': 'credit_title', 'movie_id': 'id'}, inplace=True)
    df = movies.merge(credits, on="id")
    df = df[['title', 'overview']].dropna().tail(100)
    return Dataset.from_pandas(df.rename(columns={'overview': 'description'}))

# We will always re-evaluate the latest model to get a fresh score
test_data = load_test_data()
preds, refs = [], []
current_rouge_L_score = None

with mlflow.start_run(run_name=f"Model Evaluation Comparison for {LATEST_MODEL_VERSION}"):
    mlflow.log_param("model_name", REGISTERED_MODEL_NAME)
    mlflow.log_param("model_version", LATEST_MODEL_VERSION) 
    mlflow.log_param("dataset", DATASET_NAME)
    mlflow.log_param("lts_model_version", LTS_MODEL_VERSION)
    mlflow.log_param("latest_model_version", LATEST_MODEL_VERSION)
    if LTS_ROUGE_L_SCORE is not None:
        mlflow.log_metric("lts_rougeL", LTS_ROUGE_L_SCORE * 100)
    
    rouge = evaluate.load("rouge")

    for example in test_data:
        input_text = f"generate title: {example['description']}"
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids, max_length=128, num_beams=4, early_stopping=True
            )
            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        preds.append(generated)
        refs.append(example["title"])

    # Compute ROUGE
    rouge_scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    current_rouge_L_score = rouge_scores["rougeL"]

    for k, v in rouge_scores.items():
        mlflow.log_metric(f"rouge_{k}", v * 100)
    mlflow.log_metric("latest_rougeL", current_rouge_L_score * 100)

    # Save predictions and references
    mlflow.log_text("\n".join(preds), "outputs/predictions.txt")
    mlflow.log_text("\n".join(refs), "outputs/references.txt")

    print("\nLogged ROUGE Scores:")
    for k, v in rouge_scores.items():
        print(f"{k}: {v:.4f}")

    # --- Compare and Update LTS Version ---
    if LTS_ROUGE_L_SCORE is None:
        print("\nLTS ROUGE-L score could not be retrieved. Promoting latest model to LTS by default.")
        promote = True
    else:
        print(f"\nComparing latest model ROUGE-L score ({current_rouge_L_score:.4f}) with LTS score ({LTS_ROUGE_L_SCORE:.4f})")
        promote = current_rouge_L_score > LTS_ROUGE_L_SCORE
    
    if promote:
        print(f"Latest model version {LATEST_MODEL_VERSION} is better than LTS version {LTS_MODEL_VERSION} or LTS score was unavailable.")
        
        # Update the LTS secret with the new version and its score from MLflow
        new_lts_secret_value = {
            LTS_VERSION_KEY: LATEST_MODEL_VERSION,
            "rougeL_score": current_rouge_L_score
        }
        update_secret_value(SECRETS_MANAGER_LTS_NAME, new_lts_secret_value)

        # Update the latest secret to reflect the promotion
        new_latest_secret_value = {
            LATEST_VERSION_KEY: LATEST_MODEL_VERSION
        }
        update_secret_value(SECRETS_MANAGER_LATEST_NAME, new_latest_secret_value)
    else:
        print("Latest model is not an improvement. No updates made to Secrets Manager.")