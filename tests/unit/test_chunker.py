"""Tests unitaires pour `ragtime.ingest.chunker`."""

from __future__ import annotations

from datetime import date

import pytest

from ragtime.ingest.chunker import (
    build_payload,
    compose_content,
    compute_document_id,
    to_indexable_document,
)
from ragtime.models import Ticket


def _make_ticket(**overrides: object) -> Ticket:
    base: dict[str, object] = {
        "ticket_id": 42,
        "product": "Mobile App",
        "category": "Bug Report",
        "issue_description": "App crashes on launch.",
        "resolution_notes": "Reinstall solved the issue.",
        "priority": "High",
        "status": "Resolved",
        "channel": "Email",
        "region": "Europe",
        "language": "English",
        "escalated": False,
        "sla_breached": False,
        "ticket_created_date": date(2024, 1, 15),
        "ticket_resolved_date": date(2024, 1, 17),
        "customer_satisfaction_score": 4,
        "issue_complexity_score": 2,
        "customer_segment": "Enterprise",
    }
    base.update(overrides)
    return Ticket.model_validate(base)


# --- compute_document_id -------------------------------------------------- #


def _doc_id(
    category="Bug Report",
    product="Mobile App",
    language="English",
    issue="hello",
    resolution="world",
) -> str:
    return compute_document_id(category, product, language, issue, resolution)


def test_document_id_is_deterministic() -> None:
    assert _doc_id() == _doc_id()
    assert len(_doc_id()) == 16


def test_document_id_differs_for_different_content() -> None:
    assert _doc_id(issue="hello") != _doc_id(issue="goodbye")
    assert _doc_id(resolution="r1") != _doc_id(resolution="r2")


def test_document_id_differs_for_different_context() -> None:
    """Même texte mais catégorie/produit/langue différents → ID différent."""
    assert _doc_id(category="Bug Report") != _doc_id(category="Login Issue")
    assert _doc_id(product="Mobile App") != _doc_id(product="Web Portal")
    assert _doc_id(language="English") != _doc_id(language="French")


def test_document_id_ignores_surrounding_whitespace() -> None:
    """Le strip dans le hash garantit qu'un trailing newline ne change pas l'ID."""
    assert _doc_id(issue="hello", resolution="world") == _doc_id(
        issue="  hello  ", resolution="\nworld\n"
    )


# --- compose_content ------------------------------------------------------ #


def test_compose_content_includes_all_tags_when_present() -> None:
    t = _make_ticket()
    content = compose_content(t)
    assert "[CATEGORY] Bug Report" in content
    assert "[PRODUCT] Mobile App" in content
    assert "[LANGUAGE] English" in content
    assert "[REGION] Europe" in content
    assert "[ISSUE] App crashes on launch." in content
    assert "[RESOLUTION] Reinstall solved the issue." in content


def test_compose_content_skips_empty_issue() -> None:
    t = _make_ticket(issue_description="")
    content = compose_content(t)
    assert "[ISSUE]" not in content
    assert "[RESOLUTION]" in content


def test_compose_content_skips_empty_resolution() -> None:
    t = _make_ticket(resolution_notes="")
    content = compose_content(t)
    assert "[ISSUE]" in content
    assert "[RESOLUTION]" not in content


# --- build_payload -------------------------------------------------------- #


def test_payload_contains_filterable_fields() -> None:
    t = _make_ticket()
    payload = build_payload(t)
    assert payload["ticket_id"] == 42
    assert payload["category"] == "Bug Report"
    assert payload["product"] == "Mobile App"
    assert payload["escalated"] is False
    assert payload["ticket_created_date"] == "2024-01-15"


def test_payload_excludes_pii() -> None:
    """Les champs PII ne doivent jamais apparaître dans le payload."""
    t = _make_ticket()
    payload = build_payload(t)
    assert "customer_name" not in payload
    assert "customer_email" not in payload


# --- to_indexable_document ----------------------------------------------- #


def test_to_indexable_document_roundtrip() -> None:
    t = _make_ticket()
    doc = to_indexable_document(t)
    assert doc.source_ticket_id == 42
    assert "Mobile App" in doc.content
    assert doc.payload["category"] == "Bug Report"
    assert len(doc.document_id) == 16


def test_to_indexable_document_idempotent() -> None:
    """Deux tickets avec exactement la même clé contextuelle → même document_id."""
    t1 = _make_ticket(ticket_id=1)
    t2 = _make_ticket(ticket_id=999)  # même category/product/language/issue/resolution
    assert to_indexable_document(t1).document_id == to_indexable_document(t2).document_id


def test_to_indexable_document_changes_on_content_change() -> None:
    t1 = _make_ticket()
    t2 = _make_ticket(issue_description="Different issue.")
    assert to_indexable_document(t1).document_id != to_indexable_document(t2).document_id


def test_to_indexable_document_changes_on_context_change() -> None:
    """Mêmes textes mais produit ou langue différents → docs distincts."""
    t1 = _make_ticket()
    t2 = _make_ticket(product="API Service")
    t3 = _make_ticket(language="French")
    ids = {to_indexable_document(t).document_id for t in (t1, t2, t3)}
    assert len(ids) == 3


@pytest.mark.parametrize("field", ["issue_description", "resolution_notes"])
def test_to_indexable_document_handles_empty_text_field(field: str) -> None:
    """Un seul champ texte non vide suffit (validé en amont par normalizer)."""
    t = _make_ticket(**{field: ""})
    doc = to_indexable_document(t)
    assert doc.content  # non vide
