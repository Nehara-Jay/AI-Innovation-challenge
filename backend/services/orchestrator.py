"""
Orchestrator Service
Glues Retrieval (Person 2) and Reasoning (Person 1) together into an
end-to-end multi-hop Question-Answering pipeline with Multimodal Image Support.
"""

import re
import time
from pathlib import Path
from typing import List, Optional, Set

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
        self.raw_archive_dir = settings.data_dir / "raw" / "Ashen_Era_Archive"

    def _get_all_archive_images(self) -> List[Path]:
        """Collects all image files across the raw archive."""
        if not self.raw_archive_dir.exists():
            return []
        imgs = list(self.raw_archive_dir.rglob("*.png")) + list(self.raw_archive_dir.rglob("*.jpg"))
        return imgs

    def find_matching_images(self, question: str, hits: List[SearchHit]) -> List[str]:
        """
        Discovers relevant image plates matching the question and retrieved context,
        ranked by semantic and entity relevance.
        """
        all_plates = self._get_all_archive_images()
        if not all_plates:
            return []

        plate_scores: Dict[str, float] = {}
        q_lower = question.lower()

        # Stop words to ignore
        stop_words = {
            "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
            "does", "show", "image", "images", "plate", "plates", "portrait", "portraits",
            "banner", "banners", "central", "emblem", "emblems", "holding", "look", "like",
            "their", "they", "this", "that", "with", "from", "the", "and", "for", "object",
            "there", "tell", "describe", "depict", "depicted"
        }
        raw_words = re.findall(r"[a-zA-Z0-9_]{3,}", q_lower)
        keywords = [w for w in raw_words if w not in stop_words and len(w) >= 3]

        # 1. Score based on query entity / keyword matches
        for plate in all_plates:
            plate_stem = plate.stem.lower()
            plate_path_str = str(plate.resolve())
            score = 0.0

            matching_kws = [kw for kw in keywords if kw in plate_stem]
            if matching_kws:
                # Base score for keyword matches
                score += sum(3.0 if len(kw) >= 5 else 1.5 for kw in matching_kws)
                # Extra bonus for multiple keyword matches on the same entity
                if len(matching_kws) >= 2:
                    score += 5.0

            if score > 0:
                plate_scores[plate_path_str] = plate_scores.get(plate_path_str, 0.0) + score

        # 2. Score based on explicit markdown image links in retrieved chunks
        for h in hits:
            found_links = re.findall(r"!\[.*?\]\((.*?)\)", h.content)
            for link_name in found_links:
                link_clean = Path(link_name).name.lower()
                for plate in all_plates:
                    if plate.name.lower() == link_clean or plate.stem.lower() == Path(link_clean).stem:
                        plate_path_str = str(plate.resolve())
                        plate_scores[plate_path_str] = plate_scores.get(plate_path_str, 0.0) + 2.0

        # Sort by score descending
        ranked_plates = sorted(plate_scores.keys(), key=lambda p: plate_scores[p], reverse=True)
        return ranked_plates[:3]

    def ask(self, request: AskRequest) -> AskResponse:
        """
        Main end-to-end RAG and Multi-Hop QA pipeline with Multimodal Image Support.
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

        # Step 3: Multimodal Image Discovery & Vision Analysis
        matched_image_paths = self.find_matching_images(question, all_hits)
        is_visual_query = any(w in question.lower() for w in ["look like", "holding", "color", "symbol", "emblem", "chart", "plate", "image", "visual", "portrait", "banner", "depict", "wear"])

        if matched_image_paths and is_visual_query:
            for img_path in matched_image_paths[:2]:
                vision_analysis = self.reasoning.analyze_image(img_path, question)
                if vision_analysis:
                    # Inject vision analysis as high-relevance synthetic context chunk
                    img_name = Path(img_path).name
                    all_hits.insert(
                        0,
                        SearchHit(
                            id=999999,
                            score=0.99,
                            chunk_id=f"vision_{img_name}",
                            source=f"Visual Plate: {img_name}",
                            page=1,
                            type="document",
                            content=f"[Visual Inspection of {img_name}]: {vision_analysis}",
                        )
                    )
                    reasoning_steps.append(
                        ReasoningStep(
                            step_number=len(reasoning_steps) + 1,
                            sub_query=f"Visual Analysis of {img_name}",
                            thought=f"Analyzed visual elements from {img_name}: {vision_analysis[:200]}...",
                            evidence_found=[vision_analysis[:150]],
                        )
                    )

        # Update evidence found in reasoning trace
        if all_hits and len(reasoning_steps) > 0:
            evidence_summaries = [f"{h.source} (Page {h.page}): {h.content[:100]}..." for h in all_hits[:4]]
            reasoning_steps[0].evidence_found = evidence_summaries

        # Step 4: Format Citations
        citations: List[SourceCitation] = self.retrieval.to_citations(all_hits, max_citations=6)

        # Step 5: Synthesize Final Answer with LLM Reasoning
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
            images=matched_image_paths,
            latency_ms=round(elapsed_ms, 2),
            model_used=settings.llm_model,
        )


# Singleton instance
orchestrator_service = OrchestratorService()
