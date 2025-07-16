import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# --- Configuration ---
# MLflow Tracking Server URI (from .env or direct IP)
# MFLOW_SERVER_IP = os.getenv('MFLOW_SERVER_IP')
MFLOW_SERVER_IP = "34.251.243.175"
if MFLOW_SERVER_IP is None:
    raise ValueError("MFLOW_SERVER_IP environment variable is not set. Please set it to your MLflow server's public IP or ensure it's in your .env file.")
MLFLOW_TRACKING_URI = f"http://{MFLOW_SERVER_IP}/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Registered Model Name (must match what you used in train.py)
REGISTERED_MODEL_NAME = "MovieTitleGeneratorFlanT5"

# Model Version to load. You can specify a number (e.g., "1", "2")
# or a stage (e.g., "Production", "Staging").
# For initial deployment, "latest" will load the most recently registered version.
MODEL_VERSION_OR_STAGE = "latest"

# --- Global Model and Tokenizer Variables ---
model = None
tokenizer = None

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Movie Title Generation API",
    description="API for suggesting movie titles based on descriptions using a fine-tuned Flan-T5 model.",
    version="0.1.0",
)

# --- Request Body Model ---
class InferenceRequest(BaseModel):
    """
    Defines the structure of the request body for movie title generation.
    """
    description: str

# --- API Endpoints ---

@app.on_event("startup")
async def load_model_on_startup():
    """
    Loads the MLflow model and tokenizer when the FastAPI application starts.
    This ensures the model is loaded only once and is ready for inference requests.
    """
    global model, tokenizer
    print(f"Loading model '{REGISTERED_MODEL_NAME}' version/stage '{MODEL_VERSION_OR_STAGE}' from MLflow Registry...")
    try:
        # Construct the MLflow model URI
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION_OR_STAGE}"

        # Load the model using mlflow.transformers.load_model
        # This will download the model artifacts from S3 to a local cache
        # and load them into memory.
        loaded_model_components = mlflow.transformers.load_model(model_uri)
        
        # mlflow.transformers.load_model returns a dictionary with 'model' and 'tokenizer'
        model = loaded_model_components["model"]
        tokenizer = loaded_model_components["tokenizer"]

        # Set model to evaluation mode
        model.eval()

        # Move model to GPU if available
        if torch.cuda.is_available():
            model.to("cuda")
            print("Model moved to GPU.")
        else:
            print("No GPU found, model will run on CPU.")

        print("Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise RuntimeError(f"Could not load model on startup: {e}")

@app.post("/predict")
async def predict_movie_title(request: InferenceRequest):
    """
    Generates a movie title based on the provided description.

    Args:
        request (InferenceRequest): A Pydantic model containing the movie description.

    Returns:
        dict: A dictionary containing the generated movie title.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again in a moment.")

    description = request.description
    print(f"Received request for description: '{description}'")

    # Prepend the task prefix as used during training
    input_text = f"generate title: {description}"

    try:
        # Tokenize the input description
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        # Move inputs to GPU if model is on GPU
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate the title
        # You can adjust generation parameters like num_beams, do_sample, top_k, top_p, etc.
        outputs = model.generate(
            **inputs,
            max_new_tokens=128, # Max length for the generated title
            num_beams=5,       # Use beam search for better quality
            early_stopping=True
        )

        # Decode the generated tokens back to text
        generated_title = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated title: '{generated_title}'")

        return {"suggested_title": generated_title}

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# --- Main entry point for Uvicorn (for local testing/development) ---
if __name__ == "__main__":
    # To run this API locally:
    # 1. Ensure your MLflow server is running and accessible.
    # 2. Set the MFLOW_SERVER_IP environment variable (or in your .env file).
    # 3. Run: python api/inference_api.py
    #    Or: uvicorn api.inference_api:app --host 0.0.0.0 --port 8000 --reload
    # Access the API at http://localhost:8000/docs for Swagger UI
    uvicorn.run(app, host="0.0.0.0", port=8000)

