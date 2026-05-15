"""Pipeline d'ingestion : loader → normalizer → chunker, avec dédup.

Le pipeline est un générateur paresseux : il yield des `IndexableDocument`
au fur et à mesure. Cela permet de pipeliner avec l'indexation Qdrant
sans charger l'intégralité du dataset en mémoire.

Idempotence garantie par dédup sur `document_id` : si deux tickets ont
le même contenu textuel (issue + resolution), un seul document sort.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ragtime.ingest.chunker import to_indexable_document
from ragtime.ingest.loader import load_tickets
from ragtime.ingest.normalizer import normalize
from ragtime.logging_setup import get_logger
from ragtime.models import IndexableDocument

logger = get_logger(__name__)


def run_pipeline(
    csv_path: Path,
    sample_size: int | None = None,
) -> Iterator[IndexableDocument]:
    """Orchestre l'ingestion complète et yield des IndexableDocument uniques.

    Args:
        csv_path: Chemin du CSV source.
        sample_size: Limite éventuelle du nombre de lignes lues.

    Yields:
        IndexableDocument dédupliqués sur leur `document_id`.
    """
    seen: set[str] = set()
    n_duplicates = 0
    n_yielded = 0

    records = load_tickets(csv_path, sample_size=sample_size)
    tickets = normalize(records)

    for ticket in tickets:
        doc = to_indexable_document(ticket)
        if doc.document_id in seen:
            n_duplicates += 1
            continue
        seen.add(doc.document_id)
        n_yielded += 1
        yield doc

    logger.info(
        "pipeline.done",
        yielded=n_yielded,
        duplicates_dropped=n_duplicates,
    )
