from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import ask

app = FastAPI(title="Ashen Era API")

# Allow Streamlit to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask.router, prefix="/api")

@app.get("/health")
def health_check():
    return {"status": "Database Mocked. Ready for UI integration."}