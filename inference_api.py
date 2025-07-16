import os
import uvicorn
import pandas as pd
import kagglehub
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.transformers
from transformers import Pipeline
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast # For literal_eval to parse JSON strings

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
MODEL_VERSION_OR_STAGE = "latest"

# --- Global Variables for Models and Data ---
text_generation_pipeline: Pipeline = None
movies_df = None # DataFrame to store movie metadata for recommendations
cosine_sim = None # Cosine similarity matrix for recommendations
indices = None # Series mapping movie titles to their indices

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Movie AI Services",
    description="API for suggesting movie titles and recommending movies.",
    version="0.1.0",
)

# --- Request Body Models ---
class GenerateTitleRequest(BaseModel):
    """
    Defines the structure of the request body for movie title generation.
    """
    description: str

class RecommendMovieRequest(BaseModel):
    """
    Defines the structure of the request body for movie recommendations.
    """
    movie_title: str
    num_recommendations: int = 10 # Default to 10 recommendations

# --- Helper Functions for Recommendation System ---

# Function to convert stringified list of dicts to a list of names
def convert_json_to_list(obj):
    if isinstance(obj, str):
        L = []
        try:
            for i in ast.literal_eval(obj):
                L.append(i['name'])
        except (ValueError, SyntaxError):
            return [] # Handle malformed JSON strings
        return [] # Return empty list if parsing fails or obj is not a string
    return []

# Function to get director's name from crew json
def get_director(obj):
    if isinstance(obj, str):
        try:
            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    return i['name']
        except (ValueError, SyntaxError):
            return '' # Handle malformed JSON strings
    return ''

# Function to combine features into a single string
def combine_features(row):
    features = []
    # Ensure features are lists before extending
    if isinstance(row['keywords'], list):
        features.extend([i.replace(" ", "") for i in row['keywords']])
    if isinstance(row['genres'], list):
        features.extend([i.replace(" ", "") for i in row['genres']])
    if isinstance(row['cast'], list):
        features.extend([i.replace(" ", "") for i in row['cast'][:3]]) # Top 3 actors
    if isinstance(row['director'], str):
        features.append(row['director'].replace(" ", ""))
    if isinstance(row['overview'], str):
        features.append(row['overview'])
    return " ".join(features)

# --- Data Loading and Preprocessing for Recommendation System ---
@app.on_event("startup")
async def load_models_and_data_on_startup():
    """
    Loads the MLflow model (for generation) and prepares data for the recommendation system
    when the FastAPI application starts.
    """
    global text_generation_pipeline, movies_df, cosine_sim, indices

    # 1. Load Flan-T5 Model for Title Generation
    print(f"Loading model '{REGISTERED_MODEL_NAME}' version/stage '{MODEL_VERSION_OR_STAGE}' from MLflow Registry...")
    try:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION_OR_STAGE}"
        text_generation_pipeline = mlflow.transformers.load_model(model_uri)
        
        if text_generation_pipeline.device.type == "cuda":
            print("Title Generation Model pipeline loaded and moved to GPU.")
        else:
            print("No GPU found, Title Generation Model pipeline will run on CPU.")
        print("Title Generation Model pipeline loaded successfully!")
    except Exception as e:
        print(f"Failed to load title generation model: {e}")
        raise RuntimeError(f"Could not load title generation model on startup: {e}")

    # 2. Prepare Data for Movie Recommendation System
    print("Preparing data for Movie Recommendation System...")
    try:
        path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
        movies_csv_file = os.path.join(path, "tmdb_5000_movies.csv")
        credits_csv_file = os.path.join(path, "tmdb_5000_credits.csv")

        if not os.path.exists(movies_csv_file) or not os.path.exists(credits_csv_file):
            raise FileNotFoundError("TMDB movie or credits CSV files not found after download.")

        df_movies = pd.read_csv(movies_csv_file)
        df_credits = pd.read_csv(credits_csv_file)

        # Rename 'title' in df_credits to avoid conflict with df_movies' 'title' during merge
        # We will use the 'title' from df_movies as the primary movie title.
        df_credits.rename(columns={'title': 'credit_title', 'movie_id': 'id'}, inplace=True)
        
        # Merge the two dataframes on 'id'
        movies_df = df_movies.merge(df_credits, on='id')

        # Drop rows with missing 'overview' or 'title' (from df_movies)
        movies_df.dropna(subset=['overview', 'title'], inplace=True)
        movies_df.reset_index(drop=True, inplace=True)

        # Process JSON columns
        features_to_process = ['genres', 'keywords', 'cast', 'crew']
        for feature in features_to_process:
            movies_df[feature] = movies_df[feature].apply(convert_json_to_list)

        # Extract director
        movies_df['director'] = movies_df['crew'].apply(get_director)

        # Combine relevant features into a single 'soup' string
        movies_df['soup'] = movies_df.apply(combine_features, axis=1)
        
        # Handle potential NaN values in 'soup' after combining features
        movies_df.dropna(subset=['soup'], inplace=True)
        movies_df = movies_df[movies_df['soup'].str.strip() != ''].reset_index(drop=True)


        # Initialize TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_df['soup'])

        # Compute cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Create a Series that maps movie titles to their indices
        indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

        print("Movie Recommendation System data prepared successfully!")
        print(f"Recommendation data loaded for {len(movies_df)} movies.")

    except Exception as e:
        print(f"Failed to prepare recommendation data: {e}")
        raise RuntimeError(f"Could not prepare recommendation data on startup: {e}")

# --- Recommendation Logic Function ---
def get_recommendations(title: str, num_recommendations: int):
    """
    Generates content-based movie recommendations.
    """
    if title not in indices:
        return {"error": f"Movie title '{title}' not found in the database. Please try another title."}

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the N most similar movies (excluding itself)
    # Ensure we don't go out of bounds if num_recommendations is too large
    sim_scores = sim_scores[1:num_recommendations+1] 

    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()


# --- API Endpoints ---

@app.post("/generate_title")
async def generate_movie_title(request: GenerateTitleRequest):
    """
    Generates a movie title based on the provided description using Flan-T5.
    """
    if text_generation_pipeline is None:
        raise HTTPException(status_code=503, detail="Title generation model not loaded yet. Please try again in a moment.")

    description = request.description
    print(f"Received title generation request for description: '{description}'")

    try:
        input_text_for_pipeline = f"generate title: {description}"
        
        results = text_generation_pipeline(
            input_text_for_pipeline,
            max_new_tokens=128,
            num_beams=5,
            early_stopping=True
        )

        generated_title = results[0]["generated_text"]
        print(f"Generated title: '{generated_title}'")

        return {"suggested_title": generated_title}

    except Exception as e:
        print(f"Error during title generation: {e}")
        raise HTTPException(status_code=500, detail=f"Title generation failed: {e}")

@app.post("/recommend_movie")
async def recommend_movie(request: RecommendMovieRequest):
    """
    Recommends movies based on a given movie title using content-based filtering.
    """
    if movies_df is None or cosine_sim is None or indices is None:
        raise HTTPException(status_code=503, detail="Recommendation data not loaded yet. Please try again in a moment.")

    movie_title = request.movie_title
    num_recommendations = request.num_recommendations
    print(f"Received recommendation request for movie: '{movie_title}' with {num_recommendations} recommendations.")

    try:
        recommendations = get_recommendations(movie_title, num_recommendations)
        if "error" in recommendations:
            raise HTTPException(status_code=404, detail=recommendations["error"])
        return {"recommended_movies": recommendations}
    except Exception as e:
        print(f"Error during movie recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Movie recommendation failed: {e}")

# --- Main entry point for Uvicorn ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

