"""
FastAPI router for Ask (RAG QA) and Search endpoints.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from backend.models import (
    AskRequest,
    AskResponse,
    SearchRequest,
    SearchResponse,
    SearchHit,
)
from backend.services.orchestrator import orchestrator_service, OrchestratorService
from backend.services.retrieval import retrieval_service, RetrievalService
from backend.config import settings

router = APIRouter(tags=["Question Answering & Search"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question with multi-hop RAG reasoning",
    description="Decomposes question, retrieves relevant context from Qdrant, and synthesizes evidence-backed answers with citations using DeepSeek reasoning.",
)
async def ask_question(
    request: AskRequest,
) -> AskResponse:
    try:
        response = orchestrator_service.ask(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process question: {str(e)}",
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic vector search across archive chunks",
    description="Embeds query with Voyage AI and performs cosine similarity search against Qdrant collection.",
)
async def semantic_search(
    request: SearchRequest,
) -> SearchResponse:
    try:
        hits = retrieval_service.search(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            doc_type=request.doc_type,
        )
        return SearchResponse(
            query=request.query,
            total_hits=len(hits),
            results=hits,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}",
        )
