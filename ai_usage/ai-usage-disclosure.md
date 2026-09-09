# AI Usage Disclosure Statement
## SLIIT Codefest 2026 — AI Innovation Challenge (Section 4.1 & 5.2)

---

## 1. Statement of AI Usage & Transparency

In compliance with Section 4.1 (AI Usage Policy) and Section 5.2 of the SLIIT Codefest 2026 competition guidelines, this document formally discloses the tools, models, prompt strategies, and scopes of assistance utilized during the development of this project.

All core architectural decisions, data models, prompt designs, and integration workflows were designed, verified, and directed by human team members.

---

## 2. Models & Assistants Utilized

| AI Tool / Model | Role / Scope of Assistance | Primary Tasks |
| :--- | :--- | :--- |
| **Antigravity / Gemini 2.5 Pro** | Coding Assistant & Pair Programmer | Scaffolding FastAPI routes, writing unit test scripts, refactoring Streamlit layout, generating documentation. |
| **Voyage AI (`voyage-3`)** | Core System Embedding Engine | 1024-dimensional dense vector embeddings for 3,507 corpus chunks. |
| **DeepSeek V3 (`deepseek-chat`)** | Core System Reasoning LLM | Query decomposition, conflict resolution, and evidence synthesis. |
| **Google Gemini 2.5 Flash Vision** | Multimodal Vision & OCR Engine | Scanned PDF transcription during corpus extraction and real-time figure plate inspection (Track 1A). |

---

## 3. Scope of Generated Code vs. Human Engineering

* **Corpus Ingestion & Data Cleaning (90% Human-Directed / 10% AI):**
  Document extraction pipelines, chunking heuristics (2000 chars / 200 overlap), OCR verification, and JSON normalizers.
* **Vector Database & Zero-API Setup (85% Human-Directed / 15% AI):**
  Qdrant configuration, Docker Compose networking, `embeddings.npz` serialization, and FastAPI lifespan auto-init.
* **Reasoning & Multi-Hop Pipeline (80% Human-Directed / 20% AI):**
  Pydantic data schemas, multi-hop query planner, citation attribution, and dialectical conflict resolution prompt engineering.
* **Frontend & Static Serving (80% Human-Directed / 20% AI):**
  Streamlit chat UI, reasoning expanders, responsive plate columns, and FastAPI static file mounting.

---

## 4. Verification & Validation Process

Every piece of AI-assisted code was subjected to:
1. **Static Type Checking & Linter Verification**: Pydantic schema validation for API request/response contracts.
2. **End-to-End Execution**: Tested through `scripts/test_pipeline.py` and live Streamlit queries against the local Qdrant instance.
3. **Manual Factual Verification**: Comparison of synthesized answers against ground truth codex PDFs and ephemera documents.
