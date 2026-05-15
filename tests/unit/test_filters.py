"""Tests unitaires pour `ragtime.search.filters`."""

from __future__ import annotations

from datetime import date

from qdrant_client.http.models import DatetimeRange, MatchAny, MatchValue

from ragtime.models import SearchFilters
from ragtime.search.filters import build_qdrant_filter


def test_no_filter_returns_none() -> None:
    assert build_qdrant_filter(None) is None


def test_empty_filter_returns_none() -> None:
    """SearchFilters() sans aucun champ rempli → None."""
    assert build_qdrant_filter(SearchFilters()) is None


def test_single_list_field_becomes_match_any() -> None:
    f = build_qdrant_filter(SearchFilters(category=["Login Issue", "Bug Report"]))
    assert f is not None
    assert len(f.must) == 1
    cond = f.must[0]
    assert cond.key == "category"
    assert isinstance(cond.match, MatchAny)
    assert cond.match.any == ["Login Issue", "Bug Report"]


def test_boolean_field_becomes_match_value() -> None:
    f = build_qdrant_filter(SearchFilters(escalated=True))
    assert f is not None
    assert f.must[0].key == "escalated"
    assert isinstance(f.must[0].match, MatchValue)
    assert f.must[0].match.value is True


def test_false_boolean_is_preserved() -> None:
    """False ≠ None, doit produire une condition."""
    f = build_qdrant_filter(SearchFilters(sla_breached=False))
    assert f is not None
    assert f.must[0].match.value is False


def test_date_range_combined() -> None:
    f = build_qdrant_filter(
        SearchFilters(created_after=date(2024, 1, 1), created_before=date(2024, 12, 31))
    )
    assert f is not None
    cond = f.must[0]
    assert cond.key == "ticket_created_date"
    assert isinstance(cond.range, DatetimeRange)
    assert cond.range.gte.date() == date(2024, 1, 1)
    assert cond.range.lte.date() == date(2024, 12, 31)


def test_only_created_after_set() -> None:
    f = build_qdrant_filter(SearchFilters(created_after=date(2023, 6, 1)))
    cond = f.must[0]
    assert cond.range.gte is not None
    assert cond.range.lte is None


def test_combined_filters_all_in_must() -> None:
    f = build_qdrant_filter(
        SearchFilters(
            category=["Login Issue"],
            language=["French", "English"],
            escalated=True,
            created_after=date(2024, 1, 1),
        )
    )
    assert f is not None
    keys = {c.key for c in f.must}
    assert keys == {"category", "language", "escalated", "ticket_created_date"}


def test_empty_list_is_ignored() -> None:
    """Une liste vide ne doit pas générer une condition (== pas de filtre)."""
    f = build_qdrant_filter(SearchFilters(category=[]))
    assert f is None
