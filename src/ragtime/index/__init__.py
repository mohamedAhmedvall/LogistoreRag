"""Couche d'interaction avec Qdrant : client, schema, indexer."""

from ragtime.index.client import build_qdrant_client, get_qdrant_client
from ragtime.index.indexer import IndexableRow, document_id_to_point_id, upsert_documents
from ragtime.index.schema import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    collection_exists,
    create_collection,
)

__all__ = [
    "DENSE_VECTOR_NAME",
    "SPARSE_VECTOR_NAME",
    "IndexableRow",
    "build_qdrant_client",
    "collection_exists",
    "create_collection",
    "document_id_to_point_id",
    "get_qdrant_client",
    "upsert_documents",
]
