"""Conversion des `SearchFilters` (Pydantic) en `qdrant_client.models.Filter`.

La séparation est nette : `SearchFilters` est le contrat API (côté client),
`build_qdrant_filter` est la traduction côté serveur. Les filtres sont
appliqués côté Qdrant (payload-aware), pas après récupération.

Convention :
- Listes (catégorie, produit, …) → `MatchAny` (OR sémantique).
- Booléens → `MatchValue`.
- Dates → `DatetimeRange` (gte / lte).
Toutes les conditions sont combinées en AND (placées dans `must`).
"""

from __future__ import annotations

from datetime import date, datetime

from qdrant_client.http.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

from ragtime.models import SearchFilters


def build_qdrant_filter(filters: SearchFilters | None) -> Filter | None:
    """Convertit un `SearchFilters` en `Filter` Qdrant (ou None si pas de filtres).

    Args:
        filters: Filtres applicatifs (Pydantic). None pour retrourner None.

    Returns:
        Un `Filter` Qdrant prêt à être passé à `query_points`, ou None
        s'il n'y a aucune condition.
    """
    if filters is None:
        return None

    conditions: list[FieldCondition] = []

    # Champs listes → MatchAny
    list_mapping: tuple[tuple[str, list[str] | None], ...] = (
        ("category", filters.category),
        ("product", filters.product),
        ("priority", filters.priority),
        ("status", filters.status),
        ("channel", filters.channel),
        ("region", filters.region),
        ("language", filters.language),
    )
    for field, values in list_mapping:
        if values:
            conditions.append(FieldCondition(key=field, match=MatchAny(any=list(values))))

    # Booléens → MatchValue
    if filters.escalated is not None:
        conditions.append(
            FieldCondition(key="escalated", match=MatchValue(value=filters.escalated))
        )
    if filters.sla_breached is not None:
        conditions.append(
            FieldCondition(key="sla_breached", match=MatchValue(value=filters.sla_breached))
        )

    # Dates de création → DatetimeRange
    if filters.created_after is not None or filters.created_before is not None:
        conditions.append(
            FieldCondition(
                key="ticket_created_date",
                range=DatetimeRange(
                    gte=_to_iso(filters.created_after),
                    lte=_to_iso(filters.created_before),
                ),
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)


def _to_iso(d: date | None) -> str | None:
    """Sérialise une date en ISO 8601 (datetime à minuit UTC) pour Qdrant."""
    if d is None:
        return None
    return datetime.combine(d, datetime.min.time()).isoformat() + "Z"
