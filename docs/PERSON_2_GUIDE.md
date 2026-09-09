# 📘 Person 2: Implementation & Environment Setup Guide
## Role: Retrieval Engine & Search Optimization Specialist

This document contains everything **Person 2** needs to set up their local environment, understand their role, and implement high-precision multi-hop retrieval for the **SLIIT Codefest 2026 AI Competition** (*"The Ashen Era Archive"*).

---

## 📑 Table of Contents
1. [Role Overview & Objectives](#-role-overview--objectives)
2. [Complete Local Environment Setup Guide](#-complete-local-environment-setup-guide)
3. [Connecting to the Shared Qdrant Vector DB](#-connecting-to-the-shared-qdrant-vector-db)
4. [Step-by-Step Implementation Tasks](#-step-by-step-implementation-tasks)
5. [Code Implementation Blueprint (`backend/services/retrieval.py`)](#-code-implementation-blueprint)
6. [Testing & Local Validation](#-testing--local-validation)
7. [Interface Contract with Person 1 (Reasoning)](#-interface-contract-with-person-1)
8. [Definition of Done Checklist](#-definition-of-done-checklist)

---

## 🎯 Role Overview & Objectives

In **Sub-track 1B ("Connecting Facts Across Thousands of Pages")**, questions are deliberately structured so that facts are split across multiple files.

* **Your Mission**: Build and optimize the retrieval service in [`backend/services/retrieval.py`](../backend/services/retrieval.py) that embeds queries via **Voyage AI (`voyage-3`)**, searches the **3,464 indexed Qdrant chunks**, executes multi-hop iterative lookups, boosts exact proper nouns/dates, and returns deduplicated citations.
* **Why it matters**: If your retrieval layer misses the second-hop passage, even the most capable LLM reasoning model cannot deduce the answer.

---

## 💻 Complete Local Environment Setup Guide

Follow these steps to set up your machine from scratch.

### 1. Prerequisites
* **Python 3.10+** (Recommended: Python 3.11 or 3.12 or 3.13)
* **Git** installed on your computer.
* **uv** (Ultra-fast Python package and environment manager):
  ```powershell
  # Windows (PowerShell):
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

  # macOS / Linux (Terminal):
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

---

### 2. Git Clone & Branch Checkout
Clone the repository and switch to your designated branch:

```bash
# Clone the repository
git clone https://github.com/Nehara-Jay/AI-Innovation-challenge.git
cd AI-Innovation-challenge

# Switch to your branch (or create it if new)
git checkout <your-branch-name>

# Pull the latest shared models, config, and dependencies from main
git pull origin main
```

---

### 3. Initialize Python Virtual Environment with `uv`
```powershell
# 1. Create the virtual environment (.venv)
uv venv

# 2. Install all required dependencies
uv pip install -r requirements.txt
```

---

### 4. Configure Your Environment Variables (`.env`)
Create a file named `.env` in the root folder of the project (copy from `.env.example`):

```env
# Voyage AI Configuration
VOYAGE_API_KEY=pa-your-voyage-api-key-here
VOYAGE_MODEL=voyage-3

# Qdrant Vector Database Configuration
# If running Docker locally:
QDRANT_URL=http://localhost:6333
# If connecting to the host machine running Docker on LAN:
# QDRANT_URL=http://<HOST_IP_ADDRESS>:6333

QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=ashen_era_corpus

# OpenRouter / LLM Settings
OPENROUTER_API_KEY=your-openrouter-key-here
LLM_MODEL=deepseek/deepseek-r1:free
```

> **API Key Setup**:
> * **Voyage AI**: Sign up at [dash.voyageai.com](https://dash.voyageai.com/) to get your key. Free accounts receive 200M free tokens.

---

## 🐳 Connecting to the Shared Qdrant Vector DB

You have two choices for accessing the 3,464 indexed vector points:

### Option A: Run Qdrant Locally in Docker (Recommended)
If you have Docker Desktop installed:
```powershell
# Start Qdrant container with persistent storage mount
docker compose up -d
```
Open your browser and verify: 👉 **`http://localhost:6333/dashboard`**

### Option B: Connect to Teammate's Host Machine over Wi-Fi/LAN
If a teammate is already running the Docker container on the same local network:
1. Ask them for their local IP address (e.g., `192.168.1.50`).
2. Put in your `.env`:
   ```env
   QDRANT_URL=http://192.168.1.50:6333
   ```
3. You can query their vector store directly without re-embedding!

---

## 🛠️ Step-by-Step Implementation Tasks

You are the primary owner of **[`backend/services/retrieval.py`](../backend/services/retrieval.py)**.

```
                  RETRIEVAL FLOWCHART
┌────────────────────────────────────────────────────────┐
│ 1. Query Embedder ➔ Voyage AI (input_type="query")     │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Qdrant Vector Similarity Search (Top-K per hop)     │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Document Type & Metadata Filtering (Filter builder) │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Keyword / Entity Score Booster (Proper Nouns/Dates) │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 5. Multi-Hop Context Aggregator & Chunk Deduplicator   │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ 6. Output: List[SearchHit] & List[SourceCitation]      │
└────────────────────────────────────────────────────────┘
```

---

## 📦 Code Implementation Blueprint

Here is the complete blueprint for [`backend/services/retrieval.py`](../backend/services/retrieval.py) that you will build and refine:

```python
"""
Retrieval Service: Handles Voyage AI query embeddings, Qdrant vector search,
metadata filtering, keyword boosting, and multi-hop deduplication.
Owner: Person 2
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
        self.voyage_client = voyageai.Client(api_key=settings.voyage_api_key) if settings.voyage_api_key else None
        self.qdrant_client: Optional[QdrantClient] = None

    def get_qdrant(self) -> QdrantClient:
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
        """Embeds a search query with Voyage AI using input_type='query'."""
        if not self.voyage_client:
            raise ValueError("VOYAGE_API_KEY is not set.")
        res = self.voyage_client.embed(
            texts=[query],
            model=settings.voyage_model,
            input_type="query",
        )
        return res.embeddings[0]

    def _apply_keyword_boost(self, hits: List[SearchHit], query: str, boost: float = 0.05) -> List[SearchHit]:
        """
        Boosts the score of chunks that contain exact capitalized proper nouns,
        character names, or historical dates (e.g. '301 AS', 'Brannoc', 'Mournthrone').
        """
        # Extract potential proper nouns and dates from query
        keywords = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b\d{2,4}\s*AS\b", query)
        if not keywords:
            return hits

        for hit in hits:
            content_lower = hit.content.lower()
            matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            if matches > 0:
                hit.score += min(matches * boost, 0.15)

        # Re-sort descending by boosted score
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
        qdrant = self.get_qdrant()

        # Build Qdrant filter condition if doc_type requested
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

        # Sort merged list by score descending
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


# Export singleton instance
retrieval_service = RetrievalService()
```

---

## 🧪 Testing & Local Validation

You can test your retrieval changes locally by running the test script:

```powershell
uv run python scripts/test_search.py
```

### Writing a Quick Multi-Hop Test:
Create a temporary scratch script or test in python:
```python
from backend.services.retrieval import retrieval_service

# Test Multi-Hop Retrieval
hits = retrieval_service.multi_hop_retrieve(
    primary_query="Who was the Sellsword Captain called Red-Handed?",
    secondary_query="Brannoc Ironmere death in 300 AS Accord of Mournthrone",
    top_k_per_hop=3
)

print(f"Retrieved {len(hits)} unique deduplicated chunks:")
for h in hits:
    print(f"[{h.score:.4f}] {h.source} (Page {h.page}) - {h.chunk_id}")
```

---

## 🤝 Interface Contract with Person 1

Person 1 (Reasoning & Orchestration Lead) will call your retrieval methods inside `backend/services/orchestrator.py` like this:

```python
from backend.services.retrieval import retrieval_service

# Single-hop or simple lookup
hits = retrieval_service.search(query="Where is Crookvale located?", top_k=5)

# Multi-hop retrieval with decomposed query
multi_hits = retrieval_service.multi_hop_retrieve(
    primary_query=hop1_subquery,
    secondary_query=hop2_subquery,
    top_k_per_hop=4
)

# Converting hits to citations
citations = retrieval_service.to_citations(multi_hits)
```

---

## ✅ Definition of Done Checklist

Before submitting your PR / pushing your branch, make sure:
- [ ] Environment is running with `uv` and `.env` configured.
- [ ] Qdrant container is connected (`http://localhost:6333` or host IP).
- [ ] `retrieval.py` implements single-hop search, document type filtering, keyword boosting, and multi-hop deduplication.
- [ ] `to_citations()` returns valid `SourceCitation` objects from `backend.models`.
- [ ] Tested locally via `uv run python scripts/test_search.py` with no errors.
- [ ] Code committed and pushed to your git branch!
