import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print("Starting model evaluation script...")


# Path to the fine-tuned model saved by train.py
MODEL_PATH = "MODELS"
DATASET_NAME = "imdb"

# Ensure the model path exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}. Please ensure training completed successfully.")
    exit()


print(f"Loading fine-tuned model and tokenizer from: {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
print("Model and tokenizer loaded successfully.")


print(f"Loading test split of dataset: {DATASET_NAME}...")
# We specifically load the 'test' split for independent evaluation.
raw_test_dataset = load_dataset(DATASET_NAME, split="test")
print("Test dataset loaded successfully.")
print(raw_test_dataset)


print("Tokenizing and preprocessing test data...")
def tokenize_function(examples):
    # Use the loaded tokenizer to preprocess the test data
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized_test_dataset = raw_test_dataset.map(tokenize_function, batched=True)
print("Test data tokenization complete.")

toeknized_test_dataset = tokenized_test_dataset.select(range(100))
# Rename the 'label' column to 'labels' as required by the Trainer API
# and set the format to PyTorch tensors.
tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels")
tokenized_test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

print(tokenized_test_dataset)

# This function is the same as in train.py, used for consistent metric computation.
def compute_metrics(p):
    """
    Computes accuracy, precision, recall, and f1-score for evaluation.
    Args:
        p (EvalPrediction): A tuple containing predictions and labels.
    Returns:
        dict: A dictionary of metrics.
    """
    preds = np.argmax(p.predictions, axis=1) # Get the predicted class (0 or 1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }



# We use the Trainer's evaluate method. TrainingArguments are still needed,
# but many parameters won't apply since we're only doing evaluation.
# We set `per_device_eval_batch_size` and `report_to="none"`.
print("Initializing Trainer for evaluation...")
# You need a dummy output_dir for TrainingArguments even if just evaluating
# as it might store temporary evaluation results.
EVAL_OUTPUT_DIR = "../EVAL_RESULTS"
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

eval_args = TrainingArguments(
    output_dir=EVAL_OUTPUT_DIR,
    per_device_eval_batch_size=32, # Batch size for evaluation
    report_to="none",               # Do not report to any external services
    do_train=False,                 # Explicitly state we are not training
    do_eval=True,                   # Explicitly state we are evaluating
)

evaluator = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


print("Performing evaluation on the test set...")
evaluation_results = evaluator.evaluate()
print("Evaluation finished.")


print("\n--- Final Evaluation Metrics ---")
for metric_name, value in evaluation_results.items():
    if isinstance(value, float):
        print(f"{metric_name}: {value:.4f}")
    else:
        print(f"{metric_name}: {value}")

print("\nEvaluation script completed.")
