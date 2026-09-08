"""
Unit tests for Pydantic API contracts.

These tests validate request/response constraints without requiring
Qdrant, Docker, Voyage AI, or OpenRouter.
"""

import pytest
from pydantic import ValidationError

from backend.models import (
    AskRequest,
    SearchRequest,
    SearchHit,
    SourceCitation,
    ReasoningStep,
)


# ---------------------------------------------------------
# AskRequest Tests
# ---------------------------------------------------------


def test_valid_ask_request():
    request = AskRequest(
        question="Which faction was associated with Ederon Fellgard?"
    )

    assert request.question == (
        "Which faction was associated with Ederon Fellgard?"
    )
    assert request.max_hops == 2
    assert request.top_k_per_hop == 5
    assert request.include_reasoning is True


def test_short_question_rejected():
    """
    AskRequest requires a minimum question length of 3 characters.
    """

    with pytest.raises(ValidationError):
        AskRequest(
            question="hi"
        )


def test_max_hops_lower_bound():
    """
    max_hops must be at least 1.
    """

    with pytest.raises(ValidationError):
        AskRequest(
            question="Who ruled Gloamreach?",
            max_hops=0,
        )


def test_max_hops_upper_bound():
    """
    max_hops must not exceed 4.
    """

    with pytest.raises(ValidationError):
        AskRequest(
            question="Who ruled Gloamreach?",
            max_hops=5,
        )


def test_valid_custom_hops():
    request = AskRequest(
        question="Which faction won the accord?",
        max_hops=3,
    )

    assert request.max_hops == 3


def test_top_k_per_hop_lower_bound():
    """
    top_k_per_hop must be at least 1.
    """

    with pytest.raises(ValidationError):
        AskRequest(
            question="Which faction won the accord?",
            top_k_per_hop=0,
        )


def test_top_k_per_hop_upper_bound():
    """
    top_k_per_hop must not exceed 20.
    """

    with pytest.raises(ValidationError):
        AskRequest(
            question="Which faction won the accord?",
            top_k_per_hop=21,
        )


def test_document_type_filter_can_be_set():
    request = AskRequest(
        question="What does the archive say about Gloamreach?",
        doc_type_filter="ephemera",
    )

    assert request.doc_type_filter == "ephemera"


# ---------------------------------------------------------
# SearchRequest Tests
# ---------------------------------------------------------


def test_valid_search_request():
    request = SearchRequest(
        query="Ederon Fellgard",
        top_k=5,
    )

    assert request.query == "Ederon Fellgard"
    assert request.top_k == 5


def test_search_top_k_lower_bound():
    with pytest.raises(ValidationError):
        SearchRequest(
            query="Gloamreach",
            top_k=0,
        )


def test_search_top_k_upper_bound():
    with pytest.raises(ValidationError):
        SearchRequest(
            query="Gloamreach",
            top_k=51,
        )


def test_valid_score_threshold():
    request = SearchRequest(
        query="Gloamreach",
        score_threshold=0.75,
    )

    assert request.score_threshold == 0.75


def test_score_threshold_below_zero_rejected():
    with pytest.raises(ValidationError):
        SearchRequest(
            query="Gloamreach",
            score_threshold=-0.1,
        )


def test_score_threshold_above_one_rejected():
    with pytest.raises(ValidationError):
        SearchRequest(
            query="Gloamreach",
            score_threshold=1.1,
        )


# ---------------------------------------------------------
# SearchHit Tests
# ---------------------------------------------------------


def test_search_hit_model():
    hit = SearchHit(
        id=1,
        score=0.91,
        chunk_id="chunk_00001",
        source="example_document.md",
        page=2,
        type="document",
        content="Example evidence from the archive.",
    )

    assert hit.id == 1
    assert hit.score == 0.91
    assert hit.chunk_id == "chunk_00001"
    assert hit.page == 2


# ---------------------------------------------------------
# Citation Tests
# ---------------------------------------------------------


def test_source_citation_model():
    citation = SourceCitation(
        citation_id=1,
        source="example_document.md",
        page=4,
        type="document",
        excerpt="Relevant supporting evidence.",
        relevance_score=0.88,
    )

    assert citation.citation_id == 1
    assert citation.source == "example_document.md"
    assert citation.relevance_score == 0.88


# ---------------------------------------------------------
# ReasoningStep Tests
# ---------------------------------------------------------


def test_reasoning_step_model():
    step = ReasoningStep(
        step_number=1,
        sub_query="Which faction is Ederon Fellgard associated with?",
        thought="The retrieved evidence identifies the relevant faction.",
        evidence_found=[
            "Ederon Fellgard is associated with the faction."
        ],
    )

    assert step.step_number == 1
    assert len(step.evidence_found) == 1