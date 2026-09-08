# SLIIT Codefest 2026 – AI Innovation Challenge
## Intelligent Multi-Hop Document Assistant ("The Ashen Era Archive")

An advanced Multimodal Retrieval-Augmented Generation (RAG) and reasoning engine built for the **SLIIT Codefest 2026 AI Competition**. The system processes, indexes, and queries a complex, multi-format corpus consisting of fantasy novels, lore codexes, fan-wiki articles, letters, ledgers, scanned ephemera, and visual figure plates.

---

## 📑 Table of Contents

1. [Key Features & Capabilities](#-key-features--capabilities)
2. [Prerequisites](#-prerequisites)
3. [Quickstart Guide (Zero-API Vector Restore)](#-quickstart-guide-zero-api-vector-restore)
4. [Environment Configuration (`.env`)](#-environment-configuration-env)
5. [Running the Application](#-running-the-application)
6. [API Endpoints Reference](#-api-endpoints-reference)
7. [Multimodal Vision & Reasoning Engine](#-multimodal-vision--reasoning-engine)
8. [Data Pipeline & Vector Management](#-data-pipeline--vector-management)
9. [Project Directory Structure](#-project-directory-structure)
10. [Useful Commands Cheat Sheet](#-useful-commands-cheat-sheet)

---

## 🌟 Key Features & Capabilities

* **Multimodal Visual QA (Track 1A):** Automatic figure plate discovery and visual inspection via **Gemini 2.5 Flash Vision** (e.g. heraldry banners, artifact plates, character portraits).
* **Multi-Hop Reasoning & Conflict Resolution (Track 1B):** Decomposes complex multi-part queries and resolves contradictions between official codices and unreliable in-world ephemera with transparent chain-of-thought traces.
* **Dense Vector Search:** 1024-dimensional **Voyage AI (`voyage-3`)** embeddings stored in a persistent **Qdrant** vector database (3,507 indexed passages).
* **Instant Zero-API Teammate Onboarding:** Pre-computed embeddings archive (`data/embeddings.npz`, ~12.4 MB) automatically restores all 3,507 vectors in **~2 seconds** on server startup—no API tokens required.
* **Interactive Web Interface:** Modern **Streamlit** chat UI with side-by-side visual plate rendering, step-by-step reasoning expanders, and verified source citations.
* **FastAPI Backend:** Production-ready asynchronous REST API with Swagger documentation and health endpoints.

---

## ⚙ Prerequisites

Before starting, ensure you have the following installed on your machine:

* **Python 3.10+** (Tested on Python 3.13)
* **[uv](https://docs.astral.sh/uv/)** (Blazing fast Python package manager)
  ```powershell
  # Install uv on Windows (PowerShell):
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
* **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (for running Qdrant)

---

## 🚀 Quickstart Guide (Zero-API Vector Restore)

### 1. Clone the Repository
```bash
git clone https://github.com/Nehara-Jay/AI-Innovation-challenge.git
cd AI-Innovation-challenge
```

### 2. Install Dependencies
```powershell
uv venv
uv pip install -r requirements.txt
```

### 3. Start Qdrant in Docker
```powershell
docker compose up -d
```

### 4. Start the Backend (Auto-Populates Qdrant in 2 Seconds!)
When the backend boots, it detects that Qdrant is fresh and **automatically restores all 3,507 pre-computed vectors** from `data/embeddings.npz`:
```powershell
uv run uvicorn backend.main:app --reload --port 8000
```

> **Manual Restore (Optional):** You can also manually restore vectors at any time by running:
> ```powershell
> uv run python scripts/restore_vectors.py
> ```

---

## 🔑 Environment Configuration (`.env`)

Create a `.env` file in the root directory (or copy from `.env.example`):

```env
# Voyage AI Configuration (Optional for querying pre-built vectors)
VOYAGE_API_KEY=pa-your-voyage-api-key-here
VOYAGE_MODEL=voyage-3

# Qdrant Vector DB Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=ashen_era_corpus

# LLM / Reasoning Model (OpenRouter / DeepSeek / Gemini)
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key-here
LLM_MODEL=deepseek/deepseek-chat
TEMPERATURE=0.1
MAX_TOKENS=1024
```

---

## 🖥 Running the Application

### 1. Run the FastAPI Backend Server
```powershell
uv run uvicorn backend.main:app --reload --port 8000
```
* **Interactive API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### 2. Run the Streamlit Chat UI
In a separate terminal window:
```powershell
uv run streamlit run frontend/app.py
```
* **Web UI:** [http://localhost:8501](http://localhost:8501)

---

## 📡 API Endpoints Reference

### `POST /api/ask`
End-to-end Multi-Hop Question Answering with reasoning traces, source citations, and visual plates.

**Request Body:**
```json
{
  "question": "What is the central emblem on the banner of House Morvain?",
  "max_hops": 2,
  "top_k_per_hop": 5,
  "include_reasoning": true
}
```

**Response Payload:**
```json
{
  "question": "What is the central emblem on the banner of House Morvain?",
  "answer": "The central emblem on the banner of House Morvain is a shield featuring two crossed golden keys...",
  "confidence": 0.9,
  "reasoning_trace": [
    {
      "step_number": 1,
      "sub_query": "central emblem on the banner of House Morvain",
      "thought": "Executing primary search...",
      "evidence_found": ["..."]
    }
  ],
  "citations": [
    {
      "citation_id": 1,
      "source": "Visual Plate: atmo_heraldry_faction_house_morvain.png",
      "page": 1,
      "type": "document",
      "excerpt": "..."
    }
  ],
  "images": [
    "D:\\Projects\\CodeFest\\AI-Innovation-challenge\\data\\raw\\Ashen_Era_Archive\\wiki\\images\\atmo_heraldry_faction_house_morvain.png"
  ],
  "latency_ms": 1420.5,
  "model_used": "deepseek/deepseek-chat"
}
```

### `POST /api/search`
Direct dense semantic search against Qdrant.

---

## 🧠 Multimodal Vision & Reasoning Engine

```
User Query
   │
   ├─► Query Decomposition (Sub-queries 1 & 2)
   │
   ├─► Multi-Hop Dense Retrieval (Voyage-3 + Qdrant)
   │
   ├─► Image Discovery & Plate Relevance Scoring
   │      │
   │      └─► Gemini 2.5 Flash Vision Inspection (Track 1A)
   │
   ├─► Conflict Resolution Engine (Codices vs Ephemera)
   │
   └─► DeepSeek Synthesis ──► Streamlit UI (Answer + Plates + CoT + Citations)
```

---

## 📁 Project Directory Structure

```
AI-Innovation-challenge/
├── .env                              # Environment variables & API keys (ignored in git)
├── .env.example                      # Template for environment variables
├── .gitignore                        # Git ignore file
├── docker-compose.yml                # Docker configuration for local Qdrant container
├── README.md                         # Project documentation & startup guide
├── requirements.txt                  # Python dependencies
│
├── data/
│   ├── embeddings.npz                # Pre-computed Voyage-3 vectors (3,507 points, 12.4 MB)
│   ├── chunked_archive.json          # 3,507 normalized RAG chunks with metadata
│   ├── extracted_archive.json        # Raw extracted multimodal corpus
│   └── local_qdrant/                 # Local Qdrant volume storage (gitignored)
│
├── backend/                          # FastAPI Backend Application
│   ├── main.py                       # FastAPI entrypoint with automated Qdrant auto-init
│   ├── config.py                     # App settings & environment loader
│   ├── dependencies.py               # Qdrant, Voyage, & OpenRouter client singletons
│   ├── models.py                     # Pydantic schemas (AskRequest, AskResponse, etc.)
│   ├── routers/
│   │   └── ask.py                    # /api/ask and /api/search endpoints
│   └── services/
│       ├── retrieval.py              # Voyage-3 embedding & multi-hop Qdrant search
│       ├── reasoning.py              # LLM reasoning, conflict resolution & Gemini vision
│       └── orchestrator.py           # End-to-end multimodal multi-hop orchestrator
│
├── frontend/
│   └── app.py                        # Streamlit web UI with image plate rendering
│
└── scripts/                          # Utility & CLI scripts
    ├── restore_vectors.py            # Instant 2-second vector database restore
    ├── test_pipeline.py              # Terminal end-to-end QA pipeline test
    ├── embed_to_qdrant.py            # Full dataset embedding script with Voyage AI
    └── check_json.py                 # Dataset verification & chunk stats inspection
```

---

## ⚡ Useful Commands Cheat Sheet

| Action | Command |
| :--- | :--- |
| **Start Qdrant (Docker)** | `docker compose up -d` |
| **Stop Qdrant (Docker)** | `docker compose down` |
| **Open Qdrant Dashboard** | Visit `http://localhost:6333/dashboard` |
| **Start FastAPI Server** | `uv run uvicorn backend.main:app --reload --port 8000` |
| **Start Streamlit Web UI** | `uv run streamlit run frontend/app.py` |
| **Restore Qdrant Vectors** | `uv run python scripts/restore_vectors.py` |
| **Test QA Pipeline (Terminal)** | `uv run python scripts/test_pipeline.py` |
| **Install New Dependency** | `uv pip install <package_name>` |
