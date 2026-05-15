"""Définition et création de la collection Qdrant.

Une collection unique `tickets_support` contenant deux vecteurs nommés
(`dense` et `sparse`) — c'est l'approche moderne préconisée par Qdrant
pour la recherche hybride.

Les champs catégoriels du payload sont indexés (payload indexes) pour
permettre un filtrage rapide côté serveur.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from ragtime.config import get_settings
from ragtime.exceptions import IndexError
from ragtime.logging_setup import get_logger

logger = get_logger(__name__)

# Nom du vecteur dense / sparse dans la collection.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Champs filtrables côté Qdrant (payload indexes).
# Mappage : (champ payload, type Qdrant).
_PAYLOAD_INDEXES: tuple[tuple[str, PayloadSchemaType], ...] = (
    ("category", PayloadSchemaType.KEYWORD),
    ("product", PayloadSchemaType.KEYWORD),
    ("priority", PayloadSchemaType.KEYWORD),
    ("status", PayloadSchemaType.KEYWORD),
    ("channel", PayloadSchemaType.KEYWORD),
    ("region", PayloadSchemaType.KEYWORD),
    ("language", PayloadSchemaType.KEYWORD),
    ("escalated", PayloadSchemaType.BOOL),
    ("sla_breached", PayloadSchemaType.BOOL),
    ("ticket_created_date", PayloadSchemaType.DATETIME),
    ("ticket_resolved_date", PayloadSchemaType.DATETIME),
    ("customer_segment", PayloadSchemaType.KEYWORD),
)


def collection_exists(client: QdrantClient, name: str) -> bool:
    """Indique si une collection existe déjà dans Qdrant."""
    try:
        return client.collection_exists(collection_name=name)
    except Exception as exc:  # pragma: no cover (dépend du serveur)
        raise IndexError(f"Échec de vérification d'existence de '{name}': {exc}") from exc


def create_collection(
    client: QdrantClient,
    name: str | None = None,
    dense_dim: int = 1024,
    recreate: bool = False,
) -> str:
    """Crée la collection Qdrant pour les tickets de support.

    Args:
        client: Client Qdrant déjà construit.
        name: Nom de la collection. Si None, lu depuis Settings.
        dense_dim: Dimension des vecteurs dense (1024 pour E5-large).
        recreate: Si True, supprime la collection existante avant de la recréer.

    Returns:
        Le nom effectif de la collection.

    Raises:
        IndexError: En cas d'échec d'appel à Qdrant.
    """
    settings = get_settings()
    collection_name = name or settings.qdrant_collection_name

    try:
        exists = collection_exists(client, collection_name)
        if exists and recreate:
            logger.info("qdrant.collection.delete", name=collection_name)
            client.delete_collection(collection_name=collection_name)
            exists = False
        if exists:
            logger.info("qdrant.collection.exists", name=collection_name)
            return collection_name

        logger.info("qdrant.collection.create", name=collection_name, dense_dim=dense_dim)
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=dense_dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
        )

        for field, schema in _PAYLOAD_INDEXES:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema,
            )
        logger.info(
            "qdrant.collection.indexed_payload_fields", fields=[f for f, _ in _PAYLOAD_INDEXES]
        )
        return collection_name
    except IndexError:
        raise
    except Exception as exc:
        raise IndexError(f"Échec de création de la collection '{collection_name}': {exc}") from exc
