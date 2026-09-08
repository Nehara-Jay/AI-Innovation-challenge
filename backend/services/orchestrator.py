"""
Orchestrator Service
Glues Retrieval (Person 2) and Reasoning (Person 1) together into an
end-to-end multi-hop Question-Answering pipeline.
"""

import time
from typing import List, Optional

from backend.config import settings
from backend.models import (
    AskRequest,
    AskResponse,
    ReasoningStep,
    SearchHit,
    SourceCitation,
)
from backend.services.retrieval import retrieval_service, RetrievalService
from backend.services.reasoning import reasoning_service, ReasoningService


class OrchestratorService:
    def __init__(
        self,
        retrieval: RetrievalService = retrieval_service,
        reasoning: ReasoningService = reasoning_service,
    ):
        self.retrieval = retrieval
        self.reasoning = reasoning

    def ask(self, request: AskRequest) -> AskResponse:
        """
        Main end-to-end RAG and Multi-Hop QA pipeline:
        1. Decomposes the user query into sub-queries if multi-hop.
        2. Retrieves relevant chunks for Hop 1 and Hop 2.
        3. Formulates citations and reasoning steps.
        4. Synthesizes an evidence-backed answer using LLM reasoning.
        5. Returns structured AskResponse with citations and latency.
        """
        start_time = time.time()
        question = request.question.strip()
        reasoning_steps: List[ReasoningStep] = []

        # Step 1: Query Decomposition
        if request.max_hops > 1:
            sub_queries = self.reasoning.decompose_query(question)
        else:
            sub_queries = [question]

        primary_query = sub_queries[0]
        secondary_query = sub_queries[1] if len(sub_queries) > 1 else None

        # Step 2: Multi-Hop Retrieval
        reasoning_steps.append(
            ReasoningStep(
                step_number=1,
                sub_query=primary_query,
                thought=f"Executing primary search for: '{primary_query}'",
                evidence_found=[],
            )
        )

        all_hits: List[SearchHit] = []

        if secondary_query:
            reasoning_steps.append(
                ReasoningStep(
                    step_number=2,
                    sub_query=secondary_query,
                    thought=f"Executing secondary search for: '{secondary_query}'",
                    evidence_found=[],
                )
            )
            all_hits = self.retrieval.multi_hop_retrieve(
                primary_query=primary_query,
                secondary_query=secondary_query,
                top_k_per_hop=request.top_k_per_hop,
                doc_type=request.doc_type_filter,
            )
        else:
            all_hits = self.retrieval.search(
                query=primary_query,
                top_k=request.top_k_per_hop * 2,
                doc_type=request.doc_type_filter,
            )

        # Update evidence found in reasoning trace
        if all_hits and len(reasoning_steps) > 0:
            evidence_summaries = [f"{h.source} (Page {h.page}): {h.content[:100]}..." for h in all_hits[:4]]
            reasoning_steps[0].evidence_found = evidence_summaries

        # Step 3: Format Citations
        citations: List[SourceCitation] = self.retrieval.to_citations(all_hits, max_citations=6)

        # Step 4: Synthesize Final Answer with LLM Reasoning
        answer_text, confidence = self.reasoning.synthesize_answer(
            question=question,
            context_chunks=all_hits[:6],
            reasoning_steps=reasoning_steps,
        )

        elapsed_ms = (time.time() - start_time) * 1000.0

        return AskResponse(
            question=question,
            answer=answer_text,
            confidence=confidence,
            reasoning_trace=reasoning_steps if request.include_reasoning else [],
            citations=citations,
            latency_ms=round(elapsed_ms, 2),
            model_used=settings.llm_model,
        )


# Singleton instance
orchestrator_service = OrchestratorService()
