from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import ask

app = FastAPI(title="Ashen Era Detective API")

# Crucial: Allows the Streamlit UI to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, this would be restricted to specific URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connects the routes from ask.py to the main app
app.include_router(ask.router, prefix="/api")

@app.get("/health")
def health_check():
    return {"status": "Live", "pipeline": "Connected to Qdrant & OpenRouter"}