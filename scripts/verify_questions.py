import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.models import AskRequest
from backend.services.orchestrator import orchestrator_service

# 1. Inspect Ground Truth in raw extracted data
with open(BASE_DIR / "data" / "extracted_archive.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 80)
print("🔍 MANUAL GROUND-TRUTH SEARCH IN RAW EXTRACTED ARCHIVE:")
print("=" * 80)

print("\n--- [Ground Truth for Q1: House Morvain Banner / Emblem] ---")
for c in data:
    c_low = c["content"].lower()
    if "morvain" in c_low and any(w in c_low for w in ["banner", "emblem", "sigil", "crest", "arms", "heraldry", "flag"]):
        print(f"File: {c['source']} (Page {c['page']})")
        for line in c["content"].split("\n"):
            if any(w in line.lower() for w in ["morvain", "banner", "emblem", "sigil", "crest", "arms", "flag"]):
                print(f"  > {line}")

print("\n--- [Ground Truth for Q2: Ignatz Ashgrove the Oathless Portrait] ---")
for c in data:
    c_low = c["content"].lower()
    if "ignatz" in c_low or "ashgrove" in c_low:
        print(f"File: {c['source']} (Page {c['page']})")
        for line in c["content"].split("\n"):
            if any(w in line.lower() for w in ["ignatz", "ashgrove", "portrait", "hold", "hand", "depict", "plate", "image"]):
                print(f"  > {line}")

print("\n" + "=" * 80)
print("🤖 RUNNING LLM RAG PIPELINE:")
print("=" * 80)

questions = [
    ("1a_v12", "What is the central emblem on the banner of House Morvain?"),
    ("1a_v06", "In the portrait of Ignatz Ashgrove the Oathless, what object are they holding?")
]

for qid, qtext in questions:
    print(f"\n[{qid}] QUESTION: {qtext}")
    req = AskRequest(question=qtext, max_hops=2, top_k_per_hop=5)
    resp = orchestrator_service.ask(req)

    print(f"\nLLM ANSWER (Latency: {resp.latency_ms:.0f}ms):")
    print(resp.answer)
    print("\nCITATIONS USED:")
    for cite in resp.citations:
        print(f" - [{cite.citation_id}] {cite.source} (p.{cite.page}): {cite.excerpt[:120]}...")
    print("-" * 60)
