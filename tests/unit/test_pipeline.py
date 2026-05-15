"""Tests unitaires pour `ragtime.ingest.pipeline` (avec un mini-CSV temporaire)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ragtime.exceptions import IngestionError
from ragtime.ingest.pipeline import run_pipeline


def _write_mini_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "ticket_id": 1,
        "customer_name": "Alice",
        "customer_email": "alice@example.com",
        "product": "Mobile App",
        "category": "Bug Report",
        "issue_description": "App crashes.",
        "resolution_notes": "Reinstalled.",
        "priority": "High",
        "status": "Resolved",
        "channel": "Email",
        "region": "Europe",
        "customer_age": 30,
        "customer_gender": "Female",
        "subscription_type": "Premium",
        "customer_tenure_months": 12,
        "previous_tickets": 0,
        "customer_satisfaction_score": 5,
        "first_response_time_hours": 1.5,
        "resolution_time_hours": 4.0,
        "ticket_created_date": "2024-01-01",
        "ticket_resolved_date": "2024-01-02",
        "escalated": "No",
        "sla_breached": "No",
        "operating_system": "iOS",
        "browser": "Safari",
        "payment_method": "Card",
        "language": "English",
        "preferred_contact_time": "Morning",
        "issue_complexity_score": 2,
        "customer_segment": "Consumer",
    }
    base.update(overrides)
    return base


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    csv = tmp_path / "tickets.csv"
    _write_mini_csv(csv, [_row(ticket_id=1), _row(ticket_id=2, issue_description="Other issue.")])
    docs = list(run_pipeline(csv))
    assert len(docs) == 2
    assert all(len(d.document_id) == 16 for d in docs)
    assert all("Mobile App" in d.content for d in docs)


def test_pipeline_deduplicates_same_content(tmp_path: Path) -> None:
    """Deux tickets avec même issue+resolution → 1 seul document."""
    csv = tmp_path / "tickets.csv"
    _write_mini_csv(csv, [_row(ticket_id=1), _row(ticket_id=2)])
    docs = list(run_pipeline(csv))
    assert len(docs) == 1


def test_pipeline_respects_sample_size(tmp_path: Path) -> None:
    csv = tmp_path / "tickets.csv"
    rows = [_row(ticket_id=i, issue_description=f"Issue {i}") for i in range(1, 11)]
    _write_mini_csv(csv, rows)
    docs = list(run_pipeline(csv, sample_size=3))
    assert len(docs) == 3


def test_pipeline_drops_invalid_and_empty_tickets(tmp_path: Path) -> None:
    csv = tmp_path / "tickets.csv"
    _write_mini_csv(
        csv,
        [
            _row(ticket_id=1),
            _row(ticket_id=2, issue_description="", resolution_notes=""),  # vide
            _row(ticket_id=3, customer_satisfaction_score=99, issue_description="Diff"),  # invalide
            _row(ticket_id=4, issue_description="Yet another"),
        ],
    )
    docs = list(run_pipeline(csv))
    ids = {d.source_ticket_id for d in docs}
    assert ids == {1, 4}


def test_pipeline_dropped_pii_not_in_payload(tmp_path: Path) -> None:
    csv = tmp_path / "tickets.csv"
    _write_mini_csv(csv, [_row()])
    [doc] = list(run_pipeline(csv))
    assert "customer_name" not in doc.payload
    assert "customer_email" not in doc.payload


def test_pipeline_missing_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        list(run_pipeline(tmp_path / "does_not_exist.csv"))
