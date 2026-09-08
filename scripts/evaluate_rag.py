"""
RAG Benchmark Evaluation Script
--------------------------------

Evaluates the Ashen Era multi-hop RAG system against the benchmark
dataset in tests/benchmark_questions.json.

Metrics recorded:
- Answer correctness
- Expected-source retrieval recall
- Citation count
- Model confidence
- End-to-end latency
- API/server errors
- Accuracy by reasoning-hop count

The script communicates with the FastAPI /api/ask endpoint, so the
backend may run locally or on another team member's machine.

Examples:

    uv run python scripts/evaluate_rag.py

    uv run python scripts/evaluate_rag.py --split development

    uv run python scripts/evaluate_rag.py --split holdout

    uv run python scripts/evaluate_rag.py --run-name v1

    uv run python scripts/evaluate_rag.py \
        --api-url http://192.168.1.50:8000/api/ask
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import requests


# ------------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

BENCHMARK_FILE = BASE_DIR / "tests" / "benchmark_questions.json"

OUTPUT_DIR = BASE_DIR / "evaluation"


# ------------------------------------------------------------------
# Default Configuration
# ------------------------------------------------------------------

DEFAULT_API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000/api/ask",
)

DEFAULT_TIMEOUT_SECONDS = 180


# ------------------------------------------------------------------
# Text Normalization
# ------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for transparent benchmark comparison.

    - lowercase
    - remove punctuation
    - collapse repeated whitespace
    """

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def answer_matches(
    predicted_answer: str,
    gold_answer: str,
) -> bool | None:
    """
    Check whether the human-verified gold answer is present in the
    generated answer.

    Returns:
        True  -> gold answer found
        False -> gold answer not found
        None  -> no gold answer supplied

    The benchmark currently contains short factual gold answers,
    therefore normalized phrase containment is intentionally used
    instead of an LLM judge.
    """

    if not gold_answer:
        return None

    predicted = normalize_text(predicted_answer)
    gold = normalize_text(gold_answer)

    if not predicted or not gold:
        return False

    return gold in predicted


# ------------------------------------------------------------------
# Source Normalization
# ------------------------------------------------------------------

def normalize_source(source: str) -> str:
    """
    Normalize source filenames so paths and filename formatting do not
    cause false retrieval failures.

    Example:

        wiki/Ederon_Fellgard.md
        ederon_fellgard.md

    both normalize to:

        ederonfellgard
    """

    if not source:
        return ""

    filename = Path(str(source).replace("\\", "/")).name

    # Remove common document extensions.
    filename = re.sub(
        r"\.(pdf|docx|md|txt|png|jpg|jpeg)$",
        "",
        filename,
        flags=re.IGNORECASE,
    )

    filename = normalize_text(filename)

    # Ignore spaces, underscores and punctuation differences.
    filename = re.sub(
        r"[^a-z0-9]",
        "",
        filename,
    )

    return filename


def source_matches(
    expected_source: str,
    returned_source: str,
) -> bool:
    """
    Determine whether a returned citation corresponds to an expected
    source document.
    """

    expected = normalize_source(expected_source)
    returned = normalize_source(returned_source)

    if not expected or not returned:
        return False

    return (
        expected == returned
        or expected in returned
        or returned in expected
    )


# ------------------------------------------------------------------
# Benchmark Loading
# ------------------------------------------------------------------

def load_benchmark() -> list[dict[str, Any]]:
    """
    Load benchmark questions from JSON.
    """

    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {BENCHMARK_FILE}"
        )

    with open(
        BENCHMARK_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "benchmark_questions.json must contain a JSON list."
        )

    return questions


# ------------------------------------------------------------------
# Retrieval Evaluation
# ------------------------------------------------------------------

def calculate_source_recall(
    expected_sources: list[str],
    returned_sources: list[str],
) -> tuple[float | None, list[str]]:
    """
    Calculate expected-source retrieval recall.

    Example:

        expected sources = 2
        correct returned sources = 1

        recall = 0.5

    Returns:
        recall
        matched expected sources
    """

    if not expected_sources:
        return None, []

    matched_sources = []

    for expected in expected_sources:

        found = any(
            source_matches(
                expected,
                returned,
            )
            for returned in returned_sources
        )

        if found:
            matched_sources.append(expected)

    recall = (
        len(matched_sources)
        / len(expected_sources)
    )

    return recall, matched_sources


# ------------------------------------------------------------------
# Single Question Evaluation
# ------------------------------------------------------------------

def evaluate_question(
    item: dict[str, Any],
    api_url: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """
    Submit one benchmark question to /api/ask and evaluate the result.
    """

    question = item["question"]

    requested_hops = item.get(
        "hops",
        2,
    )

    # The current AskRequest contract allows 1-4 hops.
    requested_hops = max(
        1,
        min(
            int(requested_hops),
            4,
        ),
    )

    payload = {
        "question": question,
        "max_hops": requested_hops,
        "top_k_per_hop": 5,
        "include_reasoning": True,
    }

    start_time = time.perf_counter()

    response = requests.post(
        api_url,
        json=payload,
        timeout=timeout_seconds,
    )

    client_latency_seconds = (
        time.perf_counter()
        - start_time
    )

    response.raise_for_status()

    data = response.json()

    # --------------------------------------------------------------
    # Extract API response fields
    # --------------------------------------------------------------

    predicted_answer = str(
        data.get(
            "answer",
            "",
        )
    )

    confidence = data.get(
        "confidence"
    )

    server_latency_ms = data.get(
        "latency_ms"
    )

    model_used = data.get(
        "model_used",
        "",
    )

    reasoning_trace = data.get(
        "reasoning_trace",
        [],
    )

    citations = data.get(
        "citations",
        [],
    )

    images = data.get(
        "images",
        [],
    )

    # --------------------------------------------------------------
    # Answer correctness
    # --------------------------------------------------------------

    gold_answer = item.get(
        "gold_answer",
        "",
    )

    correct = answer_matches(
        predicted_answer,
        gold_answer,
    )

    # --------------------------------------------------------------
    # Citation/source evaluation
    # --------------------------------------------------------------

    returned_sources = []

    for citation in citations:

        source = citation.get(
            "source",
            "",
        )

        if source:
            returned_sources.append(
                source
            )

    expected_sources = item.get(
        "expected_sources",
        [],
    )

    source_recall, matched_sources = (
        calculate_source_recall(
            expected_sources,
            returned_sources,
        )
    )

    retrieval_success = None

    if source_recall is not None:
        retrieval_success = (
            source_recall == 1.0
        )

    # --------------------------------------------------------------
    # Preliminary failure classification
    # --------------------------------------------------------------

    failure_type = ""

    if correct is False:

        if (
            source_recall is not None
            and source_recall < 1.0
        ):
            failure_type = (
                "RETRIEVAL_FAILURE"
            )

        elif (
            source_recall is not None
            and source_recall == 1.0
        ):
            failure_type = (
                "REASONING_FAILURE"
            )

        else:
            failure_type = (
                "ANSWER_FAILURE"
            )

    elif correct is True:

        if (
            source_recall is not None
            and source_recall < 1.0
        ):
            failure_type = (
                "CITATION_SOURCE_MISMATCH"
            )

    # Hallucination requires human evidence review.
    hallucination_review = (
        "PENDING_MANUAL_REVIEW"
    )

    # Citation correctness also requires checking excerpts manually.
    citation_review = (
        "PENDING_MANUAL_REVIEW"
    )

    # --------------------------------------------------------------
    # Result
    # --------------------------------------------------------------

    return {
        "id": item.get("id", ""),
        "split": item.get("split", ""),
        "difficulty": item.get(
            "difficulty",
            "",
        ),
        "hops": item.get(
            "hops",
            "",
        ),
        "category": item.get(
            "category",
            "",
        ),
        "question": question,
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
        "correct": correct,
        "confidence": confidence,
        "expected_sources": " | ".join(
            expected_sources
        ),
        "returned_sources": " | ".join(
            returned_sources
        ),
        "matched_sources": " | ".join(
            matched_sources
        ),
        "source_recall": (
            round(
                source_recall,
                3,
            )
            if source_recall is not None
            else ""
        ),
        "retrieval_success": retrieval_success,
        "citation_count": len(
            citations
        ),
        "reasoning_step_count": len(
            reasoning_trace
        ),
        "image_count": len(
            images
        ),
        "server_latency_ms": server_latency_ms,
        "client_latency_seconds": round(
            client_latency_seconds,
            3,
        ),
        "model_used": model_used,
        "failure_type": failure_type,
        "citation_review": citation_review,
        "hallucination_review": hallucination_review,
        "error": "",
    }


# ------------------------------------------------------------------
# API Failure Result
# ------------------------------------------------------------------

def create_error_result(
    item: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    """
    Create a benchmark row when the API call fails.
    """

    return {
        "id": item.get("id", ""),
        "split": item.get("split", ""),
        "difficulty": item.get(
            "difficulty",
            "",
        ),
        "hops": item.get(
            "hops",
            "",
        ),
        "category": item.get(
            "category",
            "",
        ),
        "question": item.get(
            "question",
            "",
        ),
        "gold_answer": item.get(
            "gold_answer",
            "",
        ),
        "predicted_answer": "",
        "correct": False,
        "confidence": "",
        "expected_sources": " | ".join(
            item.get(
                "expected_sources",
                [],
            )
        ),
        "returned_sources": "",
        "matched_sources": "",
        "source_recall": "",
        "retrieval_success": False,
        "citation_count": 0,
        "reasoning_step_count": 0,
        "image_count": 0,
        "server_latency_ms": "",
        "client_latency_seconds": "",
        "model_used": "",
        "failure_type": "API_FAILURE",
        "citation_review": (
            "NOT_AVAILABLE"
        ),
        "hallucination_review": (
            "NOT_AVAILABLE"
        ),
        "error": str(error),
    }


# ------------------------------------------------------------------
# Summary Metrics
# ------------------------------------------------------------------

def build_summary(
    results: list[dict[str, Any]],
    run_name: str,
    split: str,
    api_url: str,
) -> dict[str, Any]:
    """
    Calculate overall benchmark metrics.
    """

    total_questions = len(
        results
    )

    successful_api_calls = [
        result
        for result in results
        if not result["error"]
    ]

    scored_results = [
        result
        for result in successful_api_calls
        if result["correct"]
        is not None
    ]

    correct_results = [
        result
        for result in scored_results
        if result["correct"] is True
    ]

    # --------------------------------------------------------------
    # Answer Accuracy
    # --------------------------------------------------------------

    answer_accuracy = None

    if scored_results:
        answer_accuracy = (
            len(correct_results)
            / len(scored_results)
        )

    # --------------------------------------------------------------
    # Retrieval Recall
    # --------------------------------------------------------------

    recall_values = []

    for result in successful_api_calls:

        value = result[
            "source_recall"
        ]

        if value != "":
            recall_values.append(
                float(value)
            )

    average_source_recall = None

    if recall_values:
        average_source_recall = (
            sum(recall_values)
            / len(recall_values)
        )

    # --------------------------------------------------------------
    # Latency
    # --------------------------------------------------------------

    client_latencies = [
        float(
            result[
                "client_latency_seconds"
            ]
        )
        for result in successful_api_calls
        if result[
            "client_latency_seconds"
        ] != ""
    ]

    average_latency = None
    median_latency = None
    max_latency = None

    if client_latencies:

        average_latency = (
            statistics.mean(
                client_latencies
            )
        )

        median_latency = (
            statistics.median(
                client_latencies
            )
        )

        max_latency = max(
            client_latencies
        )

    # --------------------------------------------------------------
    # Hop-specific Accuracy
    # --------------------------------------------------------------

    accuracy_by_hops = {}

    hop_values = sorted(
        {
            result["hops"]
            for result in scored_results
            if result["hops"] != ""
        }
    )

    for hop in hop_values:

        hop_results = [
            result
            for result in scored_results
            if result["hops"] == hop
        ]

        hop_correct = [
            result
            for result in hop_results
            if result["correct"] is True
        ]

        if hop_results:

            accuracy_by_hops[
                str(hop)
            ] = round(
                len(hop_correct)
                / len(hop_results),
                4,
            )

    # --------------------------------------------------------------
    # Failure Counts
    # --------------------------------------------------------------

    failure_counts = {}

    for result in results:

        failure = result[
            "failure_type"
        ]

        if failure:
            failure_counts[failure] = (
                failure_counts.get(
                    failure,
                    0,
                )
                + 1
            )

    return {
        "run_name": run_name,
        "split": split,
        "api_url": api_url,
        "total_questions": total_questions,
        "successful_api_calls": len(
            successful_api_calls
        ),
        "api_failures": (
            total_questions
            - len(
                successful_api_calls
            )
        ),
        "scored_questions": len(
            scored_results
        ),
        "correct_answers": len(
            correct_results
        ),
        "answer_accuracy": (
            round(
                answer_accuracy,
                4,
            )
            if answer_accuracy is not None
            else None
        ),
        "answer_accuracy_percent": (
            round(
                answer_accuracy * 100,
                2,
            )
            if answer_accuracy is not None
            else None
        ),
        "average_source_recall": (
            round(
                average_source_recall,
                4,
            )
            if average_source_recall
            is not None
            else None
        ),
        "average_source_recall_percent": (
            round(
                average_source_recall * 100,
                2,
            )
            if average_source_recall
            is not None
            else None
        ),
        "average_latency_seconds": (
            round(
                average_latency,
                3,
            )
            if average_latency is not None
            else None
        ),
        "median_latency_seconds": (
            round(
                median_latency,
                3,
            )
            if median_latency is not None
            else None
        ),
        "max_latency_seconds": (
            round(
                max_latency,
                3,
            )
            if max_latency is not None
            else None
        ),
        "accuracy_by_hops": (
            accuracy_by_hops
        ),
        "failure_counts": (
            failure_counts
        ),
        "manual_review_required_for": [
            "citation correctness",
            "hallucination detection",
            "partially correct narrative answers",
        ],
    }


# ------------------------------------------------------------------
# Output Helpers
# ------------------------------------------------------------------

def save_csv(
    results: list[dict[str, Any]],
    output_file: Path,
) -> None:
    """
    Save detailed benchmark results to CSV.
    """

    if not results:
        return

    fieldnames = list(
        results[0].keys()
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


def save_summary(
    summary: dict[str, Any],
    output_file: Path,
) -> None:
    """
    Save benchmark summary to JSON.
    """

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ------------------------------------------------------------------
# Console Summary
# ------------------------------------------------------------------

def print_summary(
    summary: dict[str, Any],
) -> None:
    """
    Print useful benchmark metrics to the terminal.
    """

    print("\n")
    print("=" * 70)
    print("ASHEN ERA RAG BENCHMARK SUMMARY")
    print("=" * 70)

    print(
        f"Run: "
        f"{summary['run_name']}"
    )

    print(
        f"Split: "
        f"{summary['split']}"
    )

    print(
        f"Questions: "
        f"{summary['total_questions']}"
    )

    print(
        f"Successful API calls: "
        f"{summary['successful_api_calls']}"
    )

    print(
        f"API failures: "
        f"{summary['api_failures']}"
    )

    accuracy = summary[
        "answer_accuracy_percent"
    ]

    if accuracy is not None:
        print(
            f"Answer accuracy: "
            f"{accuracy:.2f}%"
        )

    recall = summary[
        "average_source_recall_percent"
    ]

    if recall is not None:
        print(
            f"Average source recall: "
            f"{recall:.2f}%"
        )

    average_latency = summary[
        "average_latency_seconds"
    ]

    if average_latency is not None:
        print(
            f"Average latency: "
            f"{average_latency:.3f}s"
        )

    print(
        "Accuracy by hops: "
        f"{summary['accuracy_by_hops']}"
    )

    print(
        "Failures: "
        f"{summary['failure_counts']}"
    )

    print("=" * 70)


# ------------------------------------------------------------------
# Command-Line Arguments
# ------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Ashen Era "
            "multi-hop RAG system."
        )
    )

    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=(
            "FastAPI /api/ask endpoint. "
            "Defaults to RAG_API_URL or "
            "http://127.0.0.1:8000/api/ask"
        ),
    )

    parser.add_argument(
        "--split",
        choices=[
            "development",
            "holdout",
            "all",
        ],
        default="development",
        help=(
            "Benchmark split to evaluate."
        ),
    )

    parser.add_argument(
        "--run-name",
        default="v1",
        help=(
            "Name used for output files, "
            "for example v1, v2, or final."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optionally test only the first "
            "N selected questions."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "Request timeout in seconds."
        ),
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:

    args = parse_arguments()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    questions = load_benchmark()

    # --------------------------------------------------------------
    # Select requested benchmark split
    # --------------------------------------------------------------

    if args.split != "all":

        questions = [
            item
            for item in questions
            if item.get("split")
            == args.split
        ]

    if args.limit is not None:

        questions = questions[
            :args.limit
        ]

    if not questions:

        print(
            "No benchmark questions "
            "matched the selected split."
        )

        return

    print("=" * 70)
    print("ASHEN ERA RAG BENCHMARK")
    print("=" * 70)

    print(
        f"API URL: {args.api_url}"
    )

    print(
        f"Split: {args.split}"
    )

    print(
        f"Questions selected: "
        f"{len(questions)}"
    )

    print(
        f"Run name: {args.run_name}"
    )

    print("=" * 70)

    results = []

    # --------------------------------------------------------------
    # Run benchmark
    # --------------------------------------------------------------

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question_id = item.get(
            "id",
            f"Q{index}",
        )

        print(
            f"\n[{index}/{len(questions)}] "
            f"{question_id}"
        )

        print(
            f"Question: "
            f"{item['question']}"
        )

        try:

            result = evaluate_question(
                item=item,
                api_url=args.api_url,
                timeout_seconds=args.timeout,
            )

            results.append(
                result
            )

            if result["correct"] is True:

                status_text = "CORRECT"

            elif result["correct"] is False:

                status_text = "INCORRECT"

            else:

                status_text = "NOT SCORED"

            print(
                f"Result: "
                f"{status_text}"
            )

            print(
                f"Confidence: "
                f"{result['confidence']}"
            )

            print(
                f"Source recall: "
                f"{result['source_recall']}"
            )

            print(
                f"Latency: "
                f"{result['client_latency_seconds']}s"
            )

            if result[
                "failure_type"
            ]:

                print(
                    f"Failure type: "
                    f"{result['failure_type']}"
                )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

            results.append(
                create_error_result(
                    item,
                    error,
                )
            )

    # --------------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------------

    csv_file = (
        OUTPUT_DIR
        / f"results_{args.run_name}.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / f"summary_{args.run_name}.json"
    )

    save_csv(
        results,
        csv_file,
    )

    summary = build_summary(
        results=results,
        run_name=args.run_name,
        split=args.split,
        api_url=args.api_url,
    )

    save_summary(
        summary,
        summary_file,
    )

    print_summary(
        summary
    )

    print(
        f"\nDetailed results saved to:\n"
        f"{csv_file}"
    )

    print(
        f"\nSummary saved to:\n"
        f"{summary_file}"
    )


if __name__ == "__main__":
    main()