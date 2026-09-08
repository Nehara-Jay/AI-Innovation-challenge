# Evaluation Notes

This document records benchmark failures, observed issues,
fixes, and regression-test results.

## Failure Categories

- Retrieval Failure
- Reasoning Failure
- Citation Failure
- Hallucination
- API Failure
- Latency Issue
- Conflict Resolution Failure

# RAG Evaluation Notes

This document tracks evaluation findings, benchmark failures,
integration issues, fixes, and regression-test results.

## Failure Categories

### RETRIEVAL_FAILURE
The required supporting evidence was not retrieved.

### REASONING_FAILURE
The correct evidence was retrieved, but the final reasoning or
answer was incorrect.

### CITATION_FAILURE
The generated answer may be correct, but the returned citation
does not sufficiently support it.

### HALLUCINATION
The response contains a factual claim that is not supported by
the retrieved archive evidence.

### CONFLICT_RESOLUTION_FAILURE
The system handled contradictory archive sources incorrectly.

### API_FAILURE
The request failed due to an API or service error.

### LATENCY_ISSUE
The request completed but took an unexpectedly long time.

---

## Integration Issues

| ID | Issue | Owner | Status |
|---|---|---|---|
| I001 | Docker/Qdrant unavailable on Person 4 local machine | Person 4 | Open |

---

## Benchmark Failures

| Question ID | Failure Type | Observation | Owner | Status |
|---|---|---|---|---|
| - | - | No benchmark run completed yet | - | Pending |