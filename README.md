# SLIIT Codefest 2026 – AI Innovation Challenge
## Intelligent Multi-Hop Document Assistant ("The Ashen Era Archive")

An advanced Multimodal Retrieval-Augmented Generation (RAG) and reasoning engine built for the **SLIIT Codefest 2026 AI Competition**. The system processes, indexes, and queries a complex, multi-format corpus consisting of fantasy novels, lore codexes, fan-wiki articles, letters, ledgers, and scanned ephemera.

---

## 📑 Table of Contents

1. [Architecture Overview](#-architecture-overview)
2. [Prerequisites](#-prerequisites)
3. [Initial Setup & Installation](#-initial-setup--installation)
4. [Environment Configuration (`.env`)](#-environment-configuration-env)
5. [Starting Qdrant with Docker](#-starting-qdrant-with-docker)
6. [Data Pipeline & Embedding Generation](#-data-pipeline--embedding-generation)
7. [Testing Semantic Retrieval](#-testing-semantic-retrieval)
8. [Team Collaboration & Sharing](#-team-collaboration--sharing)
9. [Project Directory Structure](#-project-directory-structure)
10. [Useful Commands Cheat Sheet](#-useful-commands-cheat-sheet)

---

## 🏛 Architecture Overview

* **Data Ingestion & OCR:** Multi-format document parser (`PyMuPDF`, `python-docx`, `PaddleOCR`) extracting PDFs, DOCX, Markdown, Text, and scanned images.
* **Smart Chunking:** Normalizes oversized documents (including 450k-character narrative files) into uniform, context-preserving RAG chunks (~1,500–2,000 characters) with overlap.
* **Dense Embeddings:** **Voyage AI (`voyage-3`)** 1024-dimensional vector embeddings with adaptive rate-limit handling and resume checkpoints.
* **Vector Database:** **Qdrant** running in Docker with persistent storage in `data/local_qdrant/` and web dashboard UI.
* **Backend API & Reasoning:** FastAPI service with multi-hop retrieval and LLM reasoning.

---

## ⚙ Prerequisites

Before starting, ensure you have the following installed on your machine:

* **Python 3.10+** (Tested on Python 3.13)
* **[uv](https://docs.astral.sh/uv/)** (Blazing fast Python package and environment manager)
  ```powershell
  # Install uv on Windows (PowerShell):
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
* **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (for running Qdrant)

---

## 🚀 Initial Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Nehara-Jay/AI-Innovation-challenge.git
cd AI-Innovation-challenge
```

### 2. Create Virtual Environment & Install Dependencies
Using `uv`, creating the virtual environment and installing all dependencies is instant:

```powershell
# Create virtual environment (.venv)
uv venv

# Install project dependencies
uv pip install -r requirements.txt
```

---

## 🔑 Environment Configuration (`.env`)

Create a `.env` file in the root directory (or copy from `.env.example`):

```env
# Voyage AI Configuration
VOYAGE_API_KEY=pa-your-voyage-api-key-here
VOYAGE_MODEL=voyage-3

# Qdrant Vector DB Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=ashen_era_corpus

# LLM / Reasoning Model (OpenRouter / DeepSeek / Gemini)
OPENROUTER_API_KEY=your-openrouter-key-here
LLM_MODEL=deepseek/deepseek-r1:free
```

> **Note:** To get a Voyage AI key, sign up at [dash.voyageai.com](https://dash.voyageai.com/). Free accounts receive 200M free tokens.

---

## 🐳 Starting Qdrant with Docker

The vector database is managed via Docker with persistent volume storage in `data/local_qdrant/`.

### 1. Start Qdrant
Ensure Docker Desktop is open and running, then execute in cmd:

```powershell
docker compose up -d
```

### 2. Access the Interactive Web Dashboard
Open your browser and navigate to:
👉 **[http://localhost:6333/dashboard](http://localhost:6333/dashboard)**

You can view the `ashen_era_corpus` collection, total points, payload attributes, and perform vector searches directly from the browser!

### 3. Stop / Restart Qdrant
```powershell
# Stop Qdrant container
docker compose down

# Check container status
docker ps
```

---

## 🔄 Data Pipeline & Embedding Generation

If you need to re-extract or re-embed the documents:

### 1. Ingest Documents & OCR (Optional if `extracted_archive.json` exists)
Extracts text and runs OCR on all raw archive files:
```powershell
uv run python scripts/extract_archive.py
```

### 2. Check Dataset Stats
Validates unique source files, chunk sizes, and extracted types:
```powershell
uv run python scripts/check_json.py
```

### 3. Generate Voyage AI Embeddings & Index to Qdrant
Chunks the dataset, generates `voyage-3` embeddings, and uploads points directly to Qdrant:

```powershell
# Fast batch run (3,464 chunks in ~30s with billing profile added)
uv run python scripts/embed_to_qdrant.py --batch-size 128 --delay 0

# Free-tier throttled run (3 RPM / 10k TPM limit)
uv run python scripts/embed_to_qdrant.py --batch-size 16 --delay 21

# Test / Dry run on first 10 chunks
uv run python scripts/embed_to_qdrant.py --limit 10
```

---

## 🔍 Testing Semantic Retrieval

Verify that similarity search is working against the Qdrant vector store:

```powershell
uv run python scripts/test_search.py
```

Sample code to search programmatically:
```python
from backend.services.retrieval import retrieval_service

# Search for relevant context
results = retrieval_service.search("Who commanded Blackford in 285 AS?", top_k=3)

for r in results:
    print(f"[{r['score']:.4f}] {r['source']} (Page {r['page']}) -> {r['content'][:150]}...")
```

---

## 👥 Team Collaboration & Sharing

Because Qdrant runs in Docker on port `6333`, team members on the same local network (Wi-Fi/LAN) can connect to your vector store without having to re-embed the dataset on their machines:

1. Find your host machine's local IP address:
   ```powershell
   ipconfig
   # Look for IPv4 Address, e.g. 192.168.1.50
   ```
2. Teammates update their `.env`:
   ```env
   QDRANT_URL=http://192.168.1.50:6333
   ```
3. Teammates can immediately query the vector database and access the dashboard at `http://192.168.1.50:6333/dashboard`.

---

## 📁 Project Directory Structure

```
AI-Innovation-challenge/
├── .env                              # Environment variables & API keys (ignored in git)
├── .env.example                      # Template for environment variables
├── .gitignore                        # Git ignore file
├── docker-compose.yml                # Docker configuration for shared Qdrant container
├── README.md                         # Project documentation & startup guide
├── requirements.txt                  # Python dependencies
├── ai_chat_log.md                    # Interaction logs
│
├── data/
│   ├── extracted_archive.json        # Raw extracted multimodal corpus (~5.6 MB)
│   ├── chunked_archive.json          # 3,464 normalized RAG chunks
│   └── local_qdrant/                 # Persistent Qdrant vector database storage
│       ├── aliases/
│       ├── collections/
│       │   └── ashen_era_corpus/     # Indexed 1024-dim Voyage-3 vectors & metadata
│       └── raft_state.json
│
├── data_pipeline/
│   ├── chunker.py                    # Recursive smart text chunker & normalizer
│   ├── embed_archive.py              # Embedding & Qdrant ingestion core pipeline
│   ├── check_json.py                 # Dataset verification script
│   └── scripts/
│       └── extract_archive.py        # Multimodal document & OCR extractor
│
├── backend/                          # FastAPI Backend Application
│   ├── main.py                       # FastAPI application entrypoint
│   ├── config.py                     # App settings & environment loader
│   ├── dependencies.py               # Dependency injection
│   ├── models.py                     # Pydantic schemas & response models
│   ├── routers/
│   │   ├── _init.py
│   │   └── ask.py                    # Query & Ask API routes
│   └── services/
│       ├── init_.py
│       ├── retrieval.py              # Semantic vector search service
│       ├── reasoning.py              # Multi-hop reasoning engine
│       └── orchestrator.py           # Pipeline orchestration
│
├── frontend/
│   └── app.py                        # Web UI application (Streamlit / Gradio)
│
└── scripts/                          # Utility & CLI scripts
    ├── check_json.py                 # Quick dataset inspection
    ├── embed_to_qdrant.py            # CLI tool to run Voyage AI embedding
    ├── extract_archive.py            # OCR & archive extraction tool
    └── test_search.py                # Semantic retrieval verification test
```

---

## ⚡ Useful Commands Cheat Sheet

| Action | Command |
| :--- | :--- |
| **Start Qdrant (Docker)** | `docker compose up -d` |
| **Stop Qdrant (Docker)** | `docker compose down` |
| **Open Qdrant Dashboard** | Visit `http://localhost:6333/dashboard` |
| **Run Embeddings Pipeline** | `uv run python scripts/embed_to_qdrant.py` |
| **Run Retrieval Search Test** | `uv run python scripts/test_search.py` |
| **Inspect Dataset Chunks** | `uv run python scripts/check_json.py` |
| **Install New Dependency** | `uv pip install <package_name>` |
