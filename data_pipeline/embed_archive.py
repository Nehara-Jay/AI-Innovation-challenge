"""
Voyage AI Embedding & Qdrant Vector DB Ingestion Pipeline.
Features:
- Smart chunking normalization
- Checkpoint & Resume capability (continues where it left off)
- Adaptive rate-limiting handling (works seamlessly on 3 RPM / 10K TPM free tier or full rate limits)
- Docker & local Qdrant support
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import voyageai
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm

from backend.config import settings
from data_pipeline.chunker import normalize_and_chunk_dataset


def get_qdrant_client(custom_url: Optional[str] = None) -> QdrantClient:
    """Initializes and returns a Qdrant client based on configuration."""
    url = custom_url or settings.qdrant_url
    if url == "local" or not url.startswith("http"):
        local_path = Path(settings.qdrant_local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        print(f"Connecting to local embedded Qdrant storage at: {local_path.resolve()}")
        return QdrantClient(path=str(local_path))
    else:
        print(f"Connecting to Qdrant at: {url}")
        return QdrantClient(
            url=url,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
        )


def embed_batch_with_adaptive_retry(
    client: voyageai.Client,
    texts: List[str],
    model: str,
    input_type: str = "document",
    max_retries: int = 8,
) -> List[List[float]]:
    """Embeds a batch of texts using Voyage AI with adaptive backoff on rate limits."""
    delay = 20.0
    for attempt in range(max_retries):
        try:
            result = client.embed(
                texts=texts,
                model=model,
                input_type=input_type,
            )
            return result.embeddings
        except Exception as e:
            err_msg = str(e)
            if "rate limit" in err_msg.lower() or "429" in err_msg or "rpm" in err_msg.lower() or "tpm" in err_msg.lower():
                print(f"\n[Rate Limit] Voyage AI 3 RPM / 10k TPM limit hit. Pausing for {delay:.0f}s before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                delay = min(delay * 1.5, 60.0)
            else:
                if attempt == max_retries - 1:
                    raise
                print(f"\n[API Error] {e}. Retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 1.5, 60.0)

    raise RuntimeError("Max retries exceeded for Voyage AI embedding.")


def run_embedding_pipeline(
    input_file: Optional[Path] = None,
    model: Optional[str] = None,
    batch_size: int = 16,
    limit: Optional[int] = None,
    recreate_collection: bool = False,
    qdrant_url: Optional[str] = None,
    delay_between_batches: float = 21.0,
) -> int:
    """
    Executes the full embedding pipeline with resume capability.
    """
    input_path = input_file or settings.raw_extracted_json
    model_name = model or settings.voyage_model
    api_key = settings.voyage_api_key

    if not api_key:
        raise ValueError("VOYAGE_API_KEY is not set in .env or environment.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    # 1. Load or produce chunked dataset
    if settings.chunked_json.exists():
        print(f"Loading existing chunked dataset from: {settings.chunked_json}")
        with open(settings.chunked_json, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        print(f"Loading raw extracted data from: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        chunks = normalize_and_chunk_dataset(
            raw_data,
            max_chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        settings.chunked_json.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.chunked_json, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    if limit is not None and limit > 0:
        chunks = chunks[:limit]

    total_chunks = len(chunks)
    print(f"Total dataset chunks to index: {total_chunks}")

    # 2. Setup Clients
    voyage_client = voyageai.Client(api_key=api_key)
    qdrant = get_qdrant_client(custom_url=qdrant_url)
    collection_name = settings.qdrant_collection

    # 3. Check existing collection and points
    existing_collections = [c.name for c in qdrant.get_collections().collections]
    
    if collection_name in existing_collections and recreate_collection:
        print(f"Recreating Qdrant collection '{collection_name}'...")
        qdrant.delete_collection(collection_name=collection_name)
        existing_collections.remove(collection_name)

    start_idx = 0
    if collection_name in existing_collections:
        col_info = qdrant.get_collection(collection_name=collection_name)
        existing_points = col_info.points_count or 0
        if existing_points > 0 and not recreate_collection:
            start_idx = existing_points
            print(f"Found {existing_points} already indexed points in '{collection_name}'. Resuming from chunk index {start_idx}...")
            if start_idx >= total_chunks:
                print(f"Collection '{collection_name}' is already fully indexed ({existing_points}/{total_chunks} chunks)!")
                return existing_points

    vector_dimension = None

    pbar = tqdm(total=total_chunks, initial=start_idx, desc="Embedding & Ingesting", unit="chunk")

    for i in range(start_idx, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        batch_texts = [c["content"] for c in batch]

        embeddings = embed_batch_with_adaptive_retry(
            client=voyage_client,
            texts=batch_texts,
            model=model_name,
            input_type="document",
        )

        if vector_dimension is None and embeddings:
            vector_dimension = len(embeddings[0])
            # Ensure collection exists
            current_collections = [c.name for c in qdrant.get_collections().collections]
            if collection_name not in current_collections:
                print(f"\nCreating collection '{collection_name}' with vector dimension {vector_dimension}...")
                qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_dimension,
                        distance=qmodels.Distance.COSINE,
                    ),
                )

        points = []
        for offset, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            point_id = i + offset + 1
            payload = {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "type": chunk["type"],
                "sub_index": chunk["sub_index"],
                "char_count": chunk["char_count"],
                "content": chunk["content"],
            }
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        qdrant.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

        pbar.update(len(batch))

        # Respect 3 RPM / 10K TPM throttle if delay specified
        if delay_between_batches > 0 and (i + batch_size < total_chunks):
            time.sleep(delay_between_batches)

    pbar.close()

    col_info = qdrant.get_collection(collection_name=collection_name)
    total_indexed = col_info.points_count
    print(f"\nSuccessfully indexed {total_indexed} points in Qdrant collection '{collection_name}'.")
    return total_indexed


if __name__ == "__main__":
    run_embedding_pipeline()
