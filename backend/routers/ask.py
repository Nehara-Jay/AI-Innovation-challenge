from fastapi import APIRouter, HTTPException
from backend.models import AskRequest, AskResponse
# Import the completed logic from Person 1 & 2
from backend.services.orchestrator import orchestrator_service

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    try:
        # Pass the entire request object to the actual AI engine
        # This function will take a few seconds as it hits Qdrant and DeepSeek
        result = orchestrator_service.ask(request)
        return result
    except Exception as e:
        # If OpenRouter crashes or Qdrant fails, send a clean error to the frontend
        raise HTTPException(status_code=500, detail=str(e))