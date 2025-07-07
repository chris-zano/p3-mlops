from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer, util # New library for embeddings
import torch
import os

# --- Model Loading Configuration ---
# We'll use a pre-trained Sentence Transformer model for generating embeddings.
# This model is different from your sentiment classification model.
SENTENCE_TRANSFORMER_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Global Variables for Models and Data ---
embedding_model = None
device = None

# Dummy movie database for demonstration
# In a real application, this would come from a database (e.g., Firestore, SQL, etc.)
# and would likely be much larger.
MOVIE_DATABASE = [
    {"title": "Interstellar", "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.", "genre": "Sci-Fi"},
    {"title": "Inception", "description": "A thief who steals information by entering people's dreams is given the inverse task of planting an idea into the mind of a C.E.O.", "genre": "Sci-Fi, Action"},
    {"title": "The Shawshank Redemption", "description": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.", "genre": "Drama"},
    {"title": "Pulp Fiction", "description": "The lives of two mob hitmen, a boxer, a gangster's wife, and a pair of diner bandits intertwine in four tales of violence and redemption.", "genre": "Crime, Drama"},
    {"title": "Forrest Gump", "description": "The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75, whose only desire is to be reunited with his childhood sweetheart.", "genre": "Drama, Romance"},
    {"title": "The Matrix", "description": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.", "genre": "Sci-Fi, Action"},
    {"title": "Spirited Away", "description": "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits, and where humans are changed into beasts.", "genre": "Animation, Fantasy"},
    {"title": "Eternal Sunshine of the Spotless Mind", "description": "When their relationship turns sour, a couple undergoes a medical procedure to have each other erased from their memories.", "genre": "Drama, Romance, Sci-Fi"},
    {"title": "Blade Runner 2049", "description": "A young blade runner's discovery of a long-buried secret leads him to track down former blade runner Rick Deckard, who's been missing for 30 years.", "genre": "Sci-Fi, Thriller"},
    {"title": "Arrival", "description": "A linguist is recruited by the military to assist in translating alien communications.", "genre": "Sci-Fi, Drama"}
]

# Store pre-computed embeddings for the movie database
movie_embeddings = None

# --- FastAPI App Setup ---
app = FastAPI(
    title="Movie Suggestion API",
    description="A FastAPI application to suggest movies based on textual descriptions using semantic search.",
    version="0.0.1",
)

# --- Pydantic Model for Input ---
class MovieDescriptionInput(BaseModel):
    """
    Defines the expected input structure for the movie suggestion endpoint.
    The model expects a single string describing the desired movie.
    """
    description: str
    top_n: int = 5 # Number of top suggestions to return, default to 5

# --- Application Startup Event ---
@app.on_event("startup")
async def load_models_and_compute_embeddings():
    """
    Loads the Sentence Transformer model and computes embeddings for the movie database
    when the FastAPI application starts.
    """
    global embedding_model, device, movie_embeddings

    print(f"Loading Sentence Transformer model: {SENTENCE_TRANSFORMER_MODEL_NAME}...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME, device=device)
        embedding_model.eval() # Set the model to evaluation mode
        print("Sentence Transformer model loaded successfully.")

        # Compute embeddings for all movie descriptions in the database
        print("Computing embeddings for the movie database...")
        movie_descriptions = [movie["description"] for movie in MOVIE_DATABASE]
        # Encode in batches for efficiency, if MOVIE_DATABASE were large
        movie_embeddings = embedding_model.encode(movie_descriptions, convert_to_tensor=True, show_progress_bar=False)
        print(f"Computed {len(movie_embeddings)} movie embeddings.")

    except Exception as e:
        print(f"Failed to load Sentence Transformer model or compute embeddings: {e}")
        embedding_model = None
        movie_embeddings = None
        # In a production setting, you might want to log this error more robustly
        # and potentially prevent the app from starting if this is critical.


# --- API Endpoints ---
@app.get("/")
async def read_root():
    """
    Returns a simple welcome message for the API.
    """
    return {"message": "Welcome to the Movie Suggestion API! Use /suggest_movies for recommendations."}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify if the embedding model and movie embeddings are loaded.
    """
    if embedding_model is not None and movie_embeddings is not None:
        return {"status": "ok", "embedding_model_loaded": True, "device": device, "movies_indexed": len(MOVIE_DATABASE)}
    else:
        raise HTTPException(status_code=503, detail="Service Unavailable: Embedding model or movie database not ready.")

@app.post("/suggest_movies")
async def suggest_movies(input_data: MovieDescriptionInput):
    """
    Suggests movies based on a textual description using semantic similarity.

    Args:
        input_data (MovieDescriptionInput): A Pydantic model containing the 'description'
                                            string and 'top_n' for the number of suggestions.

    Returns:
        dict: A dictionary containing the suggested movies, each with title, description,
              genre, and similarity score.
    """
    if embedding_model is None or movie_embeddings is None:
        raise HTTPException(status_code=503, detail="Movie suggestion service not ready. Please check API health.")

    try:
        # 1. Encode the user's input description
        # Move the query embedding to the same device as the movie embeddings
        query_embedding = embedding_model.encode(input_data.description, convert_to_tensor=True).to(device)

        # 2. Compute cosine similarity between the query and all movie embeddings
        # util.cos_sim returns a tensor of similarity scores
        cosine_scores = util.cos_sim(query_embedding, movie_embeddings)[0] # Get the scores for the single query

        # 3. Get the top N most similar movies
        # torch.topk returns values (scores) and indices
        top_results = torch.topk(cosine_scores, k=min(input_data.top_n, len(MOVIE_DATABASE)))

        suggested_movies = []
        for score, idx in zip(top_results.values, top_results.indices):
            movie = MOVIE_DATABASE[idx.item()]
            suggested_movies.append({
                "title": movie["title"],
                "description": movie["description"],
                "genre": movie["genre"],
                "similarity_score": score.item()
            })

        return {
            "query_description": input_data.description,
            "suggestions": suggested_movies
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Movie suggestion failed: {e}. "
                                                     "Ensure input is valid text.")

# --- Uvicorn Run ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
