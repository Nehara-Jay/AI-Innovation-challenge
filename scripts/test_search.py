import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.services.retrieval import retrieval_service

query = "Who commanded Blackford in 285 AS?"
print(f"Query: {query}\n")

results = retrieval_service.search(query, top_k=3)
print(f"Retrieved {len(results)} matching chunks:\n")
for i, r in enumerate(results):
    print(f"--- Result {i+1} (Score: {r['score']:.4f}) ---")
    print(f"Source: {r['source']} (Page {r['page']}) | Chunk: {r['chunk_id']}")
    print(f"Snippet: {r['content'][:250]}...\n")
