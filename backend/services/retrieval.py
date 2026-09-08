"""
Retrieval service using Voyage AI query embeddings and Qdrant similarity search.
"""

from typing import List, Dict, Any, Optional
import voyageai
from qdrant_client import QdrantClient
from backend.config import settings
from data_pipeline.embed_archive import get_qdrant_client


class RetrievalService:
    def __init__(self):
        self.voyage_client = voyageai.Client(api_key=settings.voyage_api_key) if settings.voyage_api_key else None
        self.qdrant_client: Optional[QdrantClient] = None

    def _get_qdrant(self) -> QdrantClient:
        if self.qdrant_client is None:
            self.qdrant_client = get_qdrant_client()
        return self.qdrant_client

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single search query using Voyage AI (with input_type='query')."""
        if not self.voyage_client:
            raise ValueError("VOYAGE_API_KEY is not configured.")

        result = self.voyage_client.embed(
            texts=[query],
            model=settings.voyage_model,
            input_type="query",
        )
        return result.embeddings[0]

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search against Qdrant collection.
        """
        query_vector = self.embed_query(query)
        qdrant = self._get_qdrant()

        # Build filter if doc_type specified
        query_filter = None
        if doc_type:
            from qdrant_client.http import models as qmodels
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

        formatted_results = []
        for point in search_results.points:
            formatted_results.append({
                "id": point.id,
                "score": point.score,
                "chunk_id": point.payload.get("chunk_id"),
                "source": point.payload.get("source"),
                "page": point.payload.get("page"),
                "type": point.payload.get("type"),
                "content": point.payload.get("content"),
            })

        return formatted_results


# Singleton instance
retrieval_service = RetrievalService()
