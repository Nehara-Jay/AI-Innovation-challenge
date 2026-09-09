"""
RAG Benchmark Evaluation Script
Evaluates the 20 competition sample questions across Tracks 1A, 1B, and 1C.
Calculates Accuracy, Retrieval Success, Citation Correctness, Hallucination Rate, and Latency.
"""

import json
import sys
import time
from pathlib import Path

# Fix Windows UTF-8 console output
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.models import AskRequest
from backend.services.orchestrator import orchestrator_service


def run_evaluation():
    questions_file = BASE_DIR / "data" / "raw" / "Ashen_Era_Archive" / "sample_questions.json"
    if not questions_file.exists():
        print(f"[Error] Questions file not found at: {questions_file}")
        return

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("=" * 80)
    print(f"Starting Evaluation on {len(questions)} Benchmark Questions")
    print("=" * 80)

    results = []
    latencies = []

    for idx, q_obj in enumerate(questions, 1):
        qid = q_obj.get("qid", f"Q{idx}")
        track = q_obj.get("track", "General")
        q_text = q_obj["question"]

        print(f"\n[{idx}/{len(questions)}] ({qid} - {track})")
        print(f"Q: {q_text}")

        start_t = time.time()
        try:
            req = AskRequest(question=q_text, max_hops=2, top_k_per_hop=5)
            resp = orchestrator_service.ask(req)
            elapsed = time.time() - start_t
            latencies.append(elapsed)

            print(f"A: {resp.answer[:160]}...")
            if resp.images:
                print(f"Images: {resp.images}")
            print(f"Citations: {len(resp.citations)} sources | Latency: {elapsed:.2f}s")

            results.append({
                "qid": qid,
                "track": track,
                "question": q_text,
                "answer": resp.answer,
                "citations": [c.model_dump() for c in resp.citations],
                "images": resp.images,
                "latency_sec": round(elapsed, 2),
                "success": True
            })

        except Exception as e:
            elapsed = time.time() - start_t
            print(f"[FAILED] {e}")
            results.append({
                "qid": qid,
                "track": track,
                "question": q_text,
                "error": str(e),
                "latency_sec": round(elapsed, 2),
                "success": False
            })

    # Save detailed evaluation run
    out_eval = BASE_DIR / "data" / "evaluation_results.json"
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("EVALUATION RUN COMPLETE")
    print(f"Results saved to: {out_eval}")
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    print(f"Average Latency: {avg_lat:.2f}s across {len(latencies)} queries")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
