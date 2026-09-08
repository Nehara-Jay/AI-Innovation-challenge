"""
Retrieval Service (Person 2)
Handles Voyage AI query embeddings, Qdrant vector search,
metadata filtering, keyword boosting, and multi-hop deduplication.
"""

import re
from typing import List, Dict, Any, Optional, Set
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.config import settings
from backend.models import SearchHit, SourceCitation


class RetrievalService:
    def __init__(self):
        self.voyage_client: Optional[voyageai.Client] = None
        self.qdrant_client: Optional[QdrantClient] = None

    def _get_voyage(self) -> voyageai.Client:
        """Lazily initializes and returns Voyage AI client."""
        if self.voyage_client is None:
            if not settings.voyage_api_key:
                raise ValueError("VOYAGE_API_KEY is not configured in .env")
            self.voyage_client = voyageai.Client(api_key=settings.voyage_api_key)
        return self.voyage_client

    def _get_qdrant(self) -> QdrantClient:
        """Lazily connects to Qdrant cluster or local storage."""
        if self.qdrant_client is None:
            url = settings.qdrant_url
            if url == "local" or not url.startswith("http"):
                self.qdrant_client = QdrantClient(path=settings.qdrant_local_path)
            else:
                self.qdrant_client = QdrantClient(
                    url=url,
                    api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
                )
        return self.qdrant_client

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single search query using Voyage AI with input_type='query'."""
        client = self._get_voyage()
        result = client.embed(
            texts=[query],
            model=settings.voyage_model,
            input_type="query",
        )
        return result.embeddings[0]

    def _apply_keyword_boost(self, hits: List[SearchHit], query: str, boost: float = 0.06) -> List[SearchHit]:
        """
        Boosts the score of chunks that contain exact capitalized proper nouns,
        character names, or historical dates (e.g. '301 AS', 'Brannoc', 'Mournthrone').
        """
        keywords = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b\d{2,4}\s*AS\b", query)
        if not keywords:
            return hits

        for hit in hits:
            content_lower = hit.content.lower()
            matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            if matches > 0:
                hit.score += min(matches * boost, 0.15)

        hits.sort(key=lambda x: x.score, reverse=True)
        return hits

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        doc_type: Optional[str] = None,
        apply_boost: bool = True,
    ) -> List[SearchHit]:
        """
        Executes single-hop semantic search against Qdrant with optional filtering.
        """
        query_vector = self.embed_query(query)
        qdrant = self._get_qdrant()

        query_filter = None
        if doc_type:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="type",
                        match=qmodels.MatchValue(value=doc_type),
                    )
                ]
            )

        search_results = qdrant.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )

        hits: List[SearchHit] = []
        for point in search_results.points:
            hits.append(
                SearchHit(
                    id=point.id,
                    score=float(point.score),
                    chunk_id=point.payload.get("chunk_id", ""),
                    source=point.payload.get("source", "unknown"),
                    page=point.payload.get("page", 1),
                    type=point.payload.get("type", "document"),
                    content=point.payload.get("content", ""),
                )
            )

        if apply_boost:
            hits = self._apply_keyword_boost(hits, query)

        return hits

    def multi_hop_retrieve(
        self,
        primary_query: str,
        secondary_query: Optional[str] = None,
        top_k_per_hop: int = 4,
        doc_type: Optional[str] = None,
    ) -> List[SearchHit]:
        """
        Executes a 2-hop iterative retrieval:
        1. Searches for primary_query.
        2. Searches for secondary_query (if provided).
        3. Merges and deduplicates chunks by chunk_id, preserving the highest score.
        """
        seen_chunk_ids: Set[str] = set()
        merged_hits: List[SearchHit] = []

        # Hop 1
        hop1_hits = self.search(query=primary_query, top_k=top_k_per_hop, doc_type=doc_type)
        for h in hop1_hits:
            if h.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(h.chunk_id)
                merged_hits.append(h)

        # Hop 2
        if secondary_query and secondary_query.strip():
            hop2_hits = self.search(query=secondary_query, top_k=top_k_per_hop, doc_type=doc_type)
            for h in hop2_hits:
                if h.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(h.chunk_id)
                    merged_hits.append(h)

        merged_hits.sort(key=lambda x: x.score, reverse=True)
        return merged_hits

    def to_citations(self, hits: List[SearchHit], max_citations: int = 5) -> List[SourceCitation]:
        """Converts SearchHit results into validated SourceCitation models."""
        citations: List[SourceCitation] = []
        for i, hit in enumerate(hits[:max_citations]):
            excerpt = hit.content.strip()
            if len(excerpt) > 300:
                excerpt = excerpt[:300] + "..."
            citations.append(
                SourceCitation(
                    citation_id=i + 1,
                    source=hit.source,
                    page=hit.page,
                    type=hit.type,
                    excerpt=excerpt,
                    relevance_score=round(hit.score, 4),
                )
            )
        return citations


# Singleton instance
retrieval_service = RetrievalService()
