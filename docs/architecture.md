# System Architecture & Technical Design
## The Ashen Era Archive — Intelligent Multi-Hop & Multimodal Assistant

---

## 1. Executive Summary

This document outlines the architectural design of the **Ashen Era Archive Assistant**, an advanced Multimodal Retrieval-Augmented Generation (RAG) system built for the **SLIIT Codefest 2026 AI Innovation Challenge**.

The system addresses two core competition tracks:
1. **Track 1A (Rich Answers & Visual Understanding):** Figure plate discovery, heraldry inspection, artifact identification, and character portrait analysis via Gemini 2.5 Flash Vision.
2. **Track 1B (Multi-Hop Reasoning & Conflict Resolution):** Multi-query decomposition, multi-hop evidence chaining across disparate documents, and dialectical conflict resolution between official codices and unreliable in-world ephemera.

---

## 2. High-Level System Architecture

```
                                  USER INTERFACE
                                 [ Streamlit App ]
                                         │
                                         │ HTTP JSON (POST /api/ask)
                                         ▼
                            FASTAPI BACKEND APPLICATION
                     ┌───────────────────────────────────────┐
                     │          AskRouter (/api/ask)         │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                             ORCHESTRATOR SERVICE
                     ┌───────────────────────────────────────┐
                     │ • Query Decomposition                 │
                     │ • Multi-Hop Orchestration             │
                     │ • Multimodal Plate Discovery          │
                     │ • Evidence Ranking & Context Fusion   │
                     └───────┬───────────────────────┬───────┘
                             │                       │
             ┌───────────────┴────────┐     ┌────────┴────────────────┐
             ▼                        ▼     ▼                         ▼
   RETRIEVAL SERVICE           REASONING SERVICE            STATIC FILE SERVER
 ┌──────────────────────┐   ┌───────────────────────┐     ┌───────────────────────┐
 │ • Voyage-3 Embedding │   │ • DeepSeek-V3 LLM     │     │ • /images/{plate}     │
 │ • Dense Vector Query │   │   - Query Planner     │     │ • 70 Archive Plates   │
 │ • Keyword Boosting   │   │   - Conflict Resolver │     └───────────────────────┘
 │ • Cosine Similarity  │   │   - Citation Formatter│
 └──────────┬───────────┘   │ • Gemini 2.5 Flash    │
            │               │   - Visual Plate QA   │
            ▼               └───────────────────────┘
    QDRANT VECTOR DB
 ┌──────────────────────┐
 │ • 3,507 Chunks       │
 │ • 1024-dim Vectors   │
 │ • In-Memory HNSW     │
 └──────────────────────┘
```

---

## 3. Detailed Component Breakdown

### 3.1 Data Pipeline & Ingestion Layer
* **Corpus Scope:** 340 archive files spanning 4 primary categories:
  * `chronicles/`: Narrative long-form historical novels (PDF & DOCX).
  * `codex/`: Official annals and lore records with embedded figure plates.
  * `wiki/`: Community wiki articles (Markdown) with character portraits and heraldry.
  * `ephemera/`: In-world letters, ledgers, petitions, and 17 scanned historical documents (`.scan.pdf`).
* **OCR & Vision Transcription:** Scanned historical documents were converted to high-resolution images and transcribed with Gemini 2.5 Flash Vision.
* **Recursive Contextual Chunking:** Large documents (up to 450,000 characters) were segmented into uniform RAG chunks (~1,500–2,000 characters) with a 200-character sliding overlap to maintain narrative continuity across boundaries.

### 3.2 Dense Retrieval Layer (Person 2)
* **Embedding Model:** **Voyage AI (`voyage-3`)** producing 1024-dimensional normalized vectors.
* **Vector Store:** **Qdrant** running in Docker on ports `6333` (REST/Web) and `6334` (gRPC).
* **Pre-Computed Embeddings Archive:** All 3,507 vectors are pre-computed and stored in `data/embeddings.npz` (12.4 MB). FastAPI automatically initializes the Qdrant collection in ~2 seconds upon startup.
* **Hybrid Scoring:** Vector cosine similarity is combined with proper noun and historical date keyword boosting.

### 3.3 Reasoning & Multi-Hop Synthesis Layer (Person 1)
* **Query Decomposition:** Complex multi-part queries are decomposed into 1 or 2 targeted search sub-queries using DeepSeek-V3.
* **Multimodal Visual Inspection (Track 1A):** When questions reference visual entities (banners, portraits, artifacts), the system:
  1. Scores and identifies matching figure plates from `data/images/`.
  2. Sends the image + query to **Gemini 2.5 Flash Vision**.
  3. Injects the visual findings as a synthetic high-relevance evidence chunk.
* **Conflict Resolution Engine (Track 1B):** When official codices conflict with private ephemera (e.g. bribes, rumors, diary entries), the synthesizer states official recorded history first, followed by explicit documentation of the counter-account with bracketed citations `[1]`, `[2]`.

### 3.4 API & User Interface Layer
* **FastAPI Backend:** Asynchronous REST API with automatic OpenAPI Swagger documentation (`/docs`), health check endpoint (`/health`), and static file serving for image plates (`/images`).
* **Streamlit Chat UI:** Interactive frontend displaying:
  * Evidence-backed answers.
  * Side-by-side visual plate rendering with responsive columns.
  * Expandable chain-of-thought reasoning traces (`Track 1B`).
  * Structured source citations with page numbers and document type badges.

---

## 4. Multi-Hop Data Flow Lifecycle

1. **User Question Ingestion**: The user submits a query through Streamlit or REST API `POST /api/ask`.
2. **Query Decomposition**: DeepSeek-V3 generates 1 or 2 search sub-queries if multi-hop deduction is needed.
3. **Multi-Hop Vector Retrieval**: Voyage-3 embeds sub-queries and fetches candidate passages from Qdrant with deduplication.
4. **Multimodal Plate Matching**: The orchestrator scans the question and retrieved chunks for visual plate references. If found, Gemini 2.5 Flash Vision extracts visual details.
5. **Context Assembly & Prompting**: Passages are ranked, tagged as *Official Codex* or *In-World Ephemera*, and provided to DeepSeek-V3.
6. **Synthesized Output**: Response returns the answer, source citations, visual images, and reasoning trace.
