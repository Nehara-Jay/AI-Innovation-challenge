"""
Restore Vectors Script
Populates Qdrant vector database with pre-computed Voyage-3 embeddings (3507 points)
without making any API calls. Takes ~2-3 seconds.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import settings


def restore_qdrant_vectors(
    qdrant_url: str = settings.qdrant_url,
    collection_name: str = settings.qdrant_collection,
    npz_path: Path = settings.data_dir / "embeddings.npz",
    chunks_path: Path = settings.data_dir / "chunked_archive.json",
    batch_size: int = 500,
) -> int:
    """
    Restores pre-computed embeddings into Qdrant.
    Returns total points inserted.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Embeddings file not found at {npz_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunked archive not found at {chunks_path}")

    print(f"Connecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url)

    # 1. Load Embeddings & Chunks
    print(f"Loading vectors from {npz_path.name}...")
    data = np.load(npz_path)
    vectors = data["vectors"]

    print(f"Loading chunks from {chunks_path.name}...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if len(vectors) != len(chunks):
        raise ValueError(f"Mismatch: {len(vectors)} vectors vs {len(chunks)} chunks")

    total_count = len(chunks)
    vector_dim = vectors.shape[1]
    print(f"Loaded {total_count} points (vector dimension: {vector_dim})")

    # 2. Recreate Collection if needed
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        print(f"Creating collection '{collection_name}' (dim={vector_dim}, distance=Cosine)...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
    else:
        existing_count = client.get_collection(collection_name).points_count
        if existing_count == total_count:
            print(f"Collection '{collection_name}' is already populated with {existing_count} points.")
            return existing_count

    # 3. Upload Points in Batches
    print(f"Uploading {total_count} points in batches of {batch_size}...")
    start_time = time.time()

    for i in range(0, total_count, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_vectors = vectors[i : i + batch_size].tolist()

        points = [
            qmodels.PointStruct(
                id=i + j + 1,
                vector=batch_vectors[j],
                payload=batch_chunks[j],
            )
            for j in range(len(batch_chunks))
        ]

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        print(f" - Uploaded {min(i + batch_size, total_count)}/{total_count} points...")

    elapsed = time.time() - start_time
    final_count = client.get_collection(collection_name).points_count
    print(f"Successfully restored {final_count} points into '{collection_name}' in {elapsed:.2f}s!")
    return final_count


if __name__ == "__main__":
    restore_qdrant_vectors()
