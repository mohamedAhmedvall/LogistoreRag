"""Indexation idempotente de documents dans Qdrant.

Le `point_id` d'un document est un UUID5 dérivé du `document_id` (hash
SHA-256 tronqué). Cela garantit qu'une re-indexation des mêmes données
écrase le point existant plutôt que d'en créer un nouveau.

Le payload Qdrant porte également `document_id` et `source_ticket_id`
pour faciliter la traçabilité côté UI.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import SparseVector as QdrantSparseVector

from ragtime.config import get_settings
from ragtime.embeddings.sparse import SparseVector
from ragtime.exceptions import IndexError
from ragtime.index.schema import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from ragtime.logging_setup import get_logger

logger = get_logger(__name__)

# Namespace fixe pour la dérivation UUID5 → IDs stables et déterministes.
# Tout document `document_id` produit le même point_id sur toutes les machines.
_POINT_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000001")


def document_id_to_point_id(document_id: str) -> str:
    """Dérive un UUID5 stable à partir d'un `document_id` (hex tronqué)."""
    return str(uuid.uuid5(_POINT_NAMESPACE, document_id))


@dataclass(frozen=True)
class IndexableRow:
    """Représentation prête à upsert dans Qdrant."""

    document_id: str
    source_ticket_id: int
    content: str
    dense: list[float]
    sparse: SparseVector
    payload: dict[str, Any]

    def to_point(self) -> PointStruct:
        """Convertit en `PointStruct` (qdrant-client)."""
        full_payload = {
            **self.payload,
            "document_id": self.document_id,
            "source_ticket_id": self.source_ticket_id,
            "content": self.content,
        }
        return PointStruct(
            id=document_id_to_point_id(self.document_id),
            vector={
                DENSE_VECTOR_NAME: self.dense,
                SPARSE_VECTOR_NAME: QdrantSparseVector(
                    indices=self.sparse.indices, values=self.sparse.values
                ),
            },
            payload=full_payload,
        )


def _batched(items: Iterable[IndexableRow], size: int) -> Iterator[list[IndexableRow]]:
    batch: list[IndexableRow] = []
    for it in items:
        batch.append(it)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def upsert_documents(
    client: QdrantClient,
    rows: Iterable[IndexableRow],
    collection: str | None = None,
    batch_size: int = 128,
) -> int:
    """Upsert un flux de documents dans Qdrant, par batches.

    Args:
        client: Client Qdrant connecté.
        rows: Itérable de `IndexableRow`.
        collection: Nom de la collection. Si None, depuis Settings.
        batch_size: Taille des lots envoyés à Qdrant.

    Returns:
        Le nombre total de points upserted.

    Raises:
        IndexError: En cas d'erreur Qdrant.
    """
    settings = get_settings()
    name = collection or settings.qdrant_collection_name
    total = 0
    for batch in _batched(rows, batch_size):
        try:
            client.upsert(
                collection_name=name,
                points=[r.to_point() for r in batch],
                wait=True,
            )
        except Exception as exc:
            raise IndexError(f"Échec d'upsert (batch de {len(batch)}): {exc}") from exc
        total += len(batch)
        logger.info("qdrant.upsert.batch", count=len(batch), total=total)

    logger.info("qdrant.upsert.done", collection=name, total=total)
    return total
