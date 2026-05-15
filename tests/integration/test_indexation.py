"""Tests d'intégration de la couche index (Qdrant in-memory).

Ces tests utilisent le mode `:memory:` de `qdrant-client`, qui simule
un serveur Qdrant localement sans réseau. Ils valident le cycle complet
création de collection → upsert → comptage.
"""

from __future__ import annotations

import pytest

from ragtime.embeddings.sparse import SparseVector
from ragtime.index import (
    IndexableRow,
    build_qdrant_client,
    collection_exists,
    create_collection,
    document_id_to_point_id,
    upsert_documents,
)

pytestmark = pytest.mark.integration

COLLECTION = "test_tickets"
DENSE_DIM = 1024


@pytest.fixture
def client():
    """Client Qdrant in-memory, frais à chaque test."""
    return build_qdrant_client(url=":memory:")


def _make_row(idx: int, **overrides) -> IndexableRow:
    base = {
        "document_id": f"doc{idx:08x}beefcafe",
        "source_ticket_id": idx,
        "content": f"[CATEGORY] Bug Report\n[ISSUE] Issue {idx}",
        "dense": [0.001 * idx] * DENSE_DIM,
        "sparse": SparseVector(indices=[idx, idx + 100], values=[1.0, 0.5]),
        "payload": {"category": "Bug Report", "language": "English"},
    }
    base.update(overrides)
    return IndexableRow(**base)


def test_create_collection_idempotent(client) -> None:
    """Créer une collection deux fois ne lève pas d'erreur."""
    name1 = create_collection(client, name=COLLECTION, dense_dim=DENSE_DIM)
    name2 = create_collection(client, name=COLLECTION, dense_dim=DENSE_DIM)
    assert name1 == name2 == COLLECTION
    assert collection_exists(client, COLLECTION)


def test_create_collection_recreate_clears_data(client) -> None:
    create_collection(client, name=COLLECTION, dense_dim=DENSE_DIM)
    upsert_documents(client, [_make_row(1)], collection=COLLECTION)
    assert client.get_collection(COLLECTION).points_count == 1

    create_collection(client, name=COLLECTION, dense_dim=DENSE_DIM, recreate=True)
    assert client.get_collection(COLLECTION).points_count == 0


def test_upsert_then_count(client) -> None:
    create_collection(client, name=COLLECTION, dense_dim=DENSE_DIM)
    rows = [_make_row(i) for i in range(1, 11)]
    total = upsert_documents(client, rows, collection=COLLECTION, batch_size=3)
    assert total == 10
    assert client.get_collection(COLLECTION).points_count == 10


def test_upsert_is_idempotent_on_same_document_id(client) -> None:
    """Re-upserter les mêmes docs ne crée pas de doublons."""
    create_collection(client, name=COLLECTION, dense_dim=DENSE_DIM)
    rows = [_make_row(i) for i in range(1, 6)]
    upsert_documents(client, rows, collection=COLLECTION)
    upsert_documents(client, rows, collection=COLLECTION)
    assert client.get_collection(COLLECTION).points_count == 5


def test_payload_is_retrievable_by_point_id(client) -> None:
    create_collection(client, name=COLLECTION, dense_dim=DENSE_DIM)
    row = _make_row(42)
    upsert_documents(client, [row], collection=COLLECTION)
    point_id = document_id_to_point_id(row.document_id)
    points = client.retrieve(collection_name=COLLECTION, ids=[point_id])
    assert len(points) == 1
    assert points[0].payload["source_ticket_id"] == 42
    assert points[0].payload["document_id"] == row.document_id
    assert points[0].payload["category"] == "Bug Report"
