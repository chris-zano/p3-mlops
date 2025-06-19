from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="Hello World FastAPI App",
    description="A simple FastAPI application to demonstrate basic setup.",
    version="0.0.1",
)


@app.get("/")
async def read_root():
    """
    Returns a simple 'Hello World!' message.
    """
    return {"message": "Hello World!"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)