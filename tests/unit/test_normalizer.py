"""Tests unitaires pour `ragtime.ingest.normalizer`."""

from __future__ import annotations

from ragtime.ingest.normalizer import normalize


def _record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "ticket_id": 1,
        "product": "Web Portal",
        "category": "Login Issue",
        "issue_description": "Cannot login.",
        "resolution_notes": "Reset password.",
        "priority": "High",
        "status": "Resolved",
        "channel": "Email",
        "region": "Europe",
        "language": "English",
        "escalated": "No",
        "sla_breached": "No",
    }
    base.update(overrides)
    return base


def test_normalize_yields_valid_tickets() -> None:
    records = [_record(), _record(ticket_id=2)]
    out = list(normalize(records))
    assert len(out) == 2
    assert out[0].ticket_id == 1


def test_normalize_drops_invalid_records() -> None:
    """Un dict sans ticket_id (ou avec un type cassé) est skip."""
    good = _record()
    bad = _record(customer_satisfaction_score=99)  # > 5, échec validation
    out = list(normalize([good, bad]))
    assert len(out) == 1
    assert out[0].ticket_id == 1


def test_normalize_filters_empty_content() -> None:
    """Un ticket sans issue ET sans resolution est filtré."""
    empty = _record(issue_description="", resolution_notes="")
    non_empty = _record(ticket_id=2)
    out = list(normalize([empty, non_empty]))
    assert len(out) == 1
    assert out[0].ticket_id == 2


def test_normalize_keeps_ticket_with_only_issue() -> None:
    only_issue = _record(resolution_notes="")
    out = list(normalize([only_issue]))
    assert len(out) == 1


def test_normalize_keeps_ticket_with_only_resolution() -> None:
    only_res = _record(issue_description="")
    out = list(normalize([only_res]))
    assert len(out) == 1


def test_normalize_drops_pii_silently() -> None:
    """customer_name / customer_email ne doivent jamais survivre à la normalisation."""
    rec = _record()
    rec["customer_name"] = "Alice"
    rec["customer_email"] = "alice@example.com"
    out = list(normalize([rec]))
    assert len(out) == 1
    t = out[0]
    assert not hasattr(t, "customer_name")
    assert not hasattr(t, "customer_email")
