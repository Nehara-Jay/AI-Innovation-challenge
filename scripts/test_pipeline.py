"""
End-to-end multi-hop pipeline test script.
Tests Retrieval + DeepSeek reasoning on sample Ashen Era questions.
"""

import sys
from pathlib import Path

# Fix Windows console UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.models import AskRequest
from backend.services.orchestrator import orchestrator_service


def test_question(question: str):
    print("=" * 75)
    print(f"QUESTION: {question}")
    print("=" * 75)

    req = AskRequest(question=question, max_hops=2, top_k_per_hop=4)
    resp = orchestrator_service.ask(req)

    print(f"\nANSWER (Model: {resp.model_used}, Latency: {resp.latency_ms:.0f}ms):\n")
    print(resp.answer)

    print("\nREASONING TRACE:")
    for step in resp.reasoning_trace:
        print(f" - Step {step.step_number} [{step.sub_query}]: {step.thought}")

    print("\nCITATIONS:")
    for c in resp.citations:
        print(f" [{c.citation_id}] {c.source} (Page {c.page}, Type: {c.type}) - Score: {c.relevance_score}")
        print(f"     Excerpt: {c.excerpt[:140]}...\n")


if __name__ == "__main__":
    sample_queries = [
        "Who commanded Blackford in 285 AS and what was his epithet?",
        "What grimoire did Korvath carry and what was its recorded attunement cost?",
    ]

    for q in sample_queries:
        test_question(q)
        print("\n")
