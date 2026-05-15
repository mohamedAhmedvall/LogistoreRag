"""Chargement du dataset Kaggle (CSV → itérateur de dicts).

Le loader est responsable de la lecture bas niveau : ouverture du fichier,
parsing CSV, échantillonnage optionnel. Il n'effectue **aucune** validation
métier — ce travail est délégué au `normalizer`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from ragtime.exceptions import IngestionError
from ragtime.logging_setup import get_logger

logger = get_logger(__name__)


# Colonnes texte du CSV — les NaN sont remplacés par "" pour éviter
# que Pydantic ne se plaigne d'un float `nan` quand il attend str.
_TEXT_COLUMNS: tuple[str, ...] = (
    "customer_name",
    "customer_email",
    "product",
    "category",
    "issue_description",
    "resolution_notes",
    "priority",
    "status",
    "channel",
    "region",
    "customer_gender",
    "subscription_type",
    "operating_system",
    "browser",
    "payment_method",
    "language",
    "preferred_contact_time",
    "customer_segment",
    "ticket_created_date",
    "ticket_resolved_date",
    "escalated",
    "sla_breached",
)


def _read_csv(path: Path, sample_size: int | None) -> pd.DataFrame:
    if not path.exists():
        raise IngestionError(f"Dataset introuvable : {path}")

    nrows = sample_size if sample_size is not None else None
    logger.info("loader.read_csv", path=str(path), nrows=nrows)
    df = pd.read_csv(path, nrows=nrows)

    # Normaliser les NaN texte → "" pour faciliter la validation Pydantic.
    for col in _TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df


def load_tickets(
    path: Path,
    sample_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Charge le CSV et yield des dicts (un par ligne).

    Args:
        path: Chemin absolu vers le fichier CSV.
        sample_size: Si fourni, ne lit que les `sample_size` premières lignes.

    Yields:
        Un dict par ligne avec les colonnes du CSV.

    Raises:
        IngestionError: Si le fichier n'existe pas ou est illisible.
    """
    df = _read_csv(path, sample_size)
    logger.info("loader.loaded", rows=len(df), columns=len(df.columns))

    yield from df.to_dict(orient="records")
