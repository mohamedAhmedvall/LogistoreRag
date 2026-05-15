"""Tests unitaires pour `ragtime.models`."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from ragtime.models import (
    Citation,
    IndexableDocument,
    SearchFilters,
    SearchQuery,
    SearchResponse,
    SearchResult,
    SynthesisRequest,
    SynthesisResponse,
    Ticket,
)

# --- Ticket ---------------------------------------------------------------- #


def _ticket_payload(**overrides: object) -> dict:
    base = {
        "ticket_id": 1,
        "product": "Web Portal",
        "category": "Account Suspension",
        "issue_description": "Payment was deducted but failed.",
        "resolution_notes": "Data sync restored.",
        "priority": "Urgent",
        "status": "Open",
        "channel": "Email",
        "region": "North America",
        "language": "French",
        "escalated": "No",
        "sla_breached": "Yes",
        "ticket_created_date": "2023-05-17",
    }
    base.update(overrides)
    return base


def test_ticket_parses_yesno_to_bool() -> None:
    t = Ticket(**_ticket_payload())
    assert t.escalated is False
    assert t.sla_breached is True


def test_ticket_accepts_real_bools() -> None:
    t = Ticket(**_ticket_payload(escalated=True, sla_breached=False))
    assert t.escalated is True
    assert t.sla_breached is False


def test_ticket_dates_parsed() -> None:
    t = Ticket(**_ticket_payload())
    assert t.ticket_created_date == date(2023, 5, 17)


def test_ticket_strips_whitespace() -> None:
    t = Ticket(**_ticket_payload(product="  Web Portal  "))
    assert t.product == "Web Portal"


def test_ticket_rejects_invalid_satisfaction() -> None:
    with pytest.raises(ValidationError):
        Ticket(**_ticket_payload(customer_satisfaction_score=10))


def test_ticket_rejects_negative_resolution_time() -> None:
    with pytest.raises(ValidationError):
        Ticket(**_ticket_payload(resolution_time_hours=-1.0))


def test_ticket_pii_fields_ignored() -> None:
    """Les champs PII passés en entrée sont silencieusement droppés."""
    t = Ticket(
        **_ticket_payload(),
        customer_name="Patricia Smith",
        customer_email="p@example.com",
    )
    assert not hasattr(t, "customer_name")
    assert not hasattr(t, "customer_email")


# --- IndexableDocument ---------------------------------------------------- #


def test_indexable_document_requires_non_empty_content() -> None:
    with pytest.raises(ValidationError):
        IndexableDocument(document_id="abc", source_ticket_id=1, content="")


def test_indexable_document_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        IndexableDocument(
            document_id="abc",
            source_ticket_id=1,
            content="hello",
            unknown="x",  # type: ignore[call-arg]
        )


# --- SearchQuery / Filters / Response ------------------------------------- #


def test_search_query_minimal() -> None:
    q = SearchQuery(query="payment refund")
    assert q.top_k == 10
    assert q.filters is None
    assert q.rerank is None


def test_search_query_with_filters() -> None:
    q = SearchQuery(
        query="login broken",
        top_k=5,
        filters=SearchFilters(category=["Login Issue"], escalated=True),
    )
    assert q.filters is not None
    assert q.filters.category == ["Login Issue"]
    assert q.filters.escalated is True


def test_search_query_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        SearchQuery(query="")


def test_search_query_rejects_topk_too_large() -> None:
    with pytest.raises(ValidationError):
        SearchQuery(query="x", top_k=1000)


def test_search_response_roundtrip() -> None:
    r = SearchResult(
        document_id="d1",
        source_ticket_id=42,
        score=0.91,
        content="snippet",
        payload={"category": "Refund Request"},
    )
    resp = SearchResponse(query="refund", total=1, took_ms=12.3, results=[r])
    dumped = resp.model_dump_json()
    assert "Refund Request" in dumped
    parsed = SearchResponse.model_validate_json(dumped)
    assert parsed.results[0].document_id == "d1"


# --- Synthesis ------------------------------------------------------------ #


def test_synthesis_request_requires_at_least_one_result() -> None:
    with pytest.raises(ValidationError):
        SynthesisRequest(query="x", results=[])


def test_synthesis_response_with_citations() -> None:
    cit = Citation(document_id="d1", source_ticket_id=42, span="extrait")
    resp = SynthesisResponse(
        query="x",
        answer="Voici la réponse",
        citations=[cit],
        model="anthropic/claude-3.5-sonnet",
        took_ms=420.0,
    )
    assert resp.citations[0].document_id == "d1"
    assert resp.generated_at is not None
