from fastapi import APIRouter
from backend.models import AskRequest, AskResponse, ReasoningStep, SourceCitation
import time

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    # Simulate the 3-second delay of an AI reasoning loop
    time.sleep(3) 
    
    # Return a perfect dummy response so you can build the UI right now
    return AskResponse(
        answer=f"Based on the archives, the answer to '{request.query}' involves Brannoc Ironmere.",
        reasoning_trace=[
            ReasoningStep(step_number=1, thought="Searching Qdrant for initial entities...", action="SEARCH_HOP_1"),
            ReasoningStep(step_number=2, thought="Insufficient context on the successor's laws. Generating sub-query.", action="EVALUATE"),
            ReasoningStep(step_number=3, thought="Found contradictory laws in the tavern ballad vs codex. Prioritizing codex.", action="SYNTHESIZE")
        ],
        citations=[
            SourceCitation(document_name="codex_vaeloria_i.pdf", page_number=42, snippet="Ironmere decreed the ban."),
            SourceCitation(document_name="weeping_lurker.md", page_number=1, snippet="Rumors in the tavern suggest otherwise.")
        ]
    )