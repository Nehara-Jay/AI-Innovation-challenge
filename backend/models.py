"""
Pydantic data schemas and contract models for API requests, responses,
multi-hop reasoning traces, search hits, and source citations.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# 1. Search Models
# ---------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., description="Semantic search query string")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to retrieve")
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum cosine similarity score")
    doc_type: Optional[str] = Field(default=None, description="Filter by document type (e.g., 'document' or 'ephemera')")


class SearchHit(BaseModel):
    id: int = Field(..., description="Unique Qdrant point ID")
    score: float = Field(..., description="Cosine similarity score")
    chunk_id: str = Field(..., description="Unique chunk identifier (e.g. chunk_00001)")
    source: str = Field(..., description="Origin document filename")
    page: int = Field(default=1, description="Page number of the original document")
    type: str = Field(default="document", description="Document type tag ('document' or 'ephemera')")
    content: str = Field(..., description="Text content snippet of the chunk")


class SearchResponse(BaseModel):
    query: str = Field(..., description="Original search query")
    total_hits: int = Field(..., description="Total matching chunks returned")
    results: List[SearchHit] = Field(default_factory=list, description="Ranked list of matching chunks")


# ---------------------------------------------------------
# 2. Citations & Reasoning Trace Models
# ---------------------------------------------------------

class SourceCitation(BaseModel):
    citation_id: int = Field(..., description="Sequential citation index [1], [2], etc.")
    source: str = Field(..., description="Origin document filename (e.g., codex_vaeloria_i.pdf)")
    page: int = Field(default=1, description="Page number in original document")
    type: str = Field(default="document", description="Type: 'document' (official) or 'ephemera' (unreliable narrator)")
    excerpt: str = Field(..., description="Relevant text excerpt used as evidence")
    relevance_score: Optional[float] = Field(default=None, description="Similarity score from retrieval")


class ReasoningStep(BaseModel):
    step_number: int = Field(..., description="Hop index (1, 2, 3...)")
    sub_query: str = Field(..., description="Decomposed sub-question for this hop")
    thought: str = Field(..., description="Chain-of-thought deduction or intermediate finding")
    evidence_found: List[str] = Field(default_factory=list, description="Key facts discovered in this step")


# ---------------------------------------------------------
# 3. Ask (RAG QA) Request & Response Models
# ---------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User's natural language question")
    max_hops: int = Field(default=2, ge=1, le=4, description="Maximum multi-hop retrieval iterations")
    top_k_per_hop: int = Field(default=5, ge=1, le=20, description="Number of passages fetched per hop")
    doc_type_filter: Optional[str] = Field(default=None, description="Optional filter: 'document' or 'ephemera'")
    include_reasoning: bool = Field(default=True, description="Whether to include full chain-of-thought trace in response")


class AskResponse(BaseModel):
    question: str = Field(..., description="Original question asked")
    answer: str = Field(..., description="Synthesized, evidence-backed answer")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Estimated confidence score (0.0 to 1.0)")
    reasoning_trace: List[ReasoningStep] = Field(default_factory=list, description="Step-by-step multi-hop reasoning trace")
    citations: List[SourceCitation] = Field(default_factory=list, description="Verified source document citations")
    latency_ms: float = Field(default=0.0, description="Total processing time in milliseconds")
    model_used: str = Field(default="deepseek/deepseek-r1:free", description="LLM model used for synthesis")


# ---------------------------------------------------------
# 4. System Health Model
# ---------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="API server status")
    qdrant_status: str = Field(..., description="Qdrant connection status ('connected' or 'disconnected')")
    total_vectors: int = Field(default=0, description="Total vector points in collection")
    collection_name: str = Field(..., description="Active Qdrant collection name")
    voyage_model: str = Field(..., description="Configured Voyage AI embedding model")
    llm_model: str = Field(..., description="Configured LLM reasoning model")
