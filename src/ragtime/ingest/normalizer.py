"""Normalisation des tickets (bronze → silver).

- Valide chaque dict via le modèle Pydantic `Ticket` (drop des PII + champs inconnus).
- Filtre les tickets sans contenu textuel (issue + resolution vides).
- Logue les compteurs (chargés, invalides, vides, conservés).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from pydantic import ValidationError

from ragtime.logging_setup import get_logger
from ragtime.models import Ticket

logger = get_logger(__name__)


def _has_content(t: Ticket) -> bool:
    """Vrai si le ticket a au moins un champ textuel non vide à indexer."""
    return bool(t.issue_description.strip() or t.resolution_notes.strip())


def normalize(records: Iterable[dict[str, Any]]) -> Iterator[Ticket]:
    """Normalise un flux de dicts en `Ticket` validés.

    Args:
        records: Itérable de dicts en provenance du loader.

    Yields:
        Tickets validés, non vides, prêts à être chunkés.
    """
    n_total = 0
    n_invalid = 0
    n_empty = 0
    n_kept = 0

    for record in records:
        n_total += 1
        try:
            ticket = Ticket.model_validate(record)
        except ValidationError as exc:
            n_invalid += 1
            logger.warning(
                "normalizer.invalid",
                ticket_id=record.get("ticket_id"),
                errors=exc.error_count(),
            )
            continue

        if not _has_content(ticket):
            n_empty += 1
            continue

        n_kept += 1
        yield ticket

    logger.info(
        "normalizer.done",
        total=n_total,
        invalid=n_invalid,
        empty=n_empty,
        kept=n_kept,
    )
