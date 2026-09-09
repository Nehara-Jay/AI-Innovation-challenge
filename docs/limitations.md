# System Limitations & Future Improvements

---

## 1. Current Limitations

### 1.1 Multi-Hop Latency & Rate Limits
* **Observation:** Complex multi-hop queries that execute query decomposition + 2 retrieval hops + Gemini Vision analysis + DeepSeek-V3 synthesis take between **8 to 25 seconds** depending on OpenRouter traffic.
* **Mitigation:** Query decomposition is bypassed for single-hop questions (`max_hops=1`), reducing response time to <3 seconds.

### 1.2 Vision Entity Disambiguation
* **Observation:** If multiple characters share similar names (e.g. *Ignatz Ashgrove* vs *Ignatz Cindervale* vs *Ignatz Fellgard*), image plate matching relies on compound keyword scoring (`ignatz` + `ashgrove`).
* **Mitigation:** Implemented weighted compound keyword matching so plates with exact surname matches receive higher priority.

### 1.3 Static Vector Embeddings
* **Observation:** Pre-computed embeddings in `data/embeddings.npz` are frozen for the existing 3,507 chunks. Adding new documents requires re-running `scripts/embed_to_qdrant.py`.
* **Mitigation:** The re-indexing script supports incremental upserts and batching.

---

## 2. Edge Cases Handled

| Edge Case | Potential Failure | System Handling |
| :--- | :--- | :--- |
| **Conflicting Dates / Accounts** | LLM picks one randomly or hallucinates. | Prompt forces dialectical resolution: official codex statement first, followed by in-world rumor/ephemera. |
| **Scanned Handwritten Ephemera** | Standard PDF extractors return empty text. | Transcribed via Gemini 2.5 Flash Vision OCR during corpus ingestion. |
| **Unindexed/Empty Qdrant Container** | Backend errors with empty collection. | Startup lifespan auto-detects 0 points and restores all 3,507 vectors in 2s from `embeddings.npz`. |
| **Missing Image on Remote Client** | Streamlit shows broken image icon. | `get_image_display_target()` falls back from local file path to backend static URL (`http://localhost:8000/images/...`). |

---

## 3. Future Roadmap

1. **Cross-Encoder Re-ranking:** Add a lightweight cross-encoder (e.g. `bge-reranker-large` or `cohere-rerank-v3`) to re-rank top-20 retrieval candidates down to top-5 prior to LLM synthesis.
2. **Graph-RAG Entity Linker:** Build an explicit Neo4j or NetworkX entity relationship graph mapping House lineages, battle dates, and artifact attunements.
3. **Automated Benchmark Evaluation Suite:** Expand automated test questions with automated BLEU/ROUGE/G-Eval scoring across competition criteria.
