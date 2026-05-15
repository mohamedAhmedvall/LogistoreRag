"""Tests unitaires pour `ragtime.index.indexer` (sans serveur Qdrant)."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from ragtime.embeddings.sparse import SparseVector
from ragtime.exceptions import IndexError
from ragtime.index.indexer import (
    IndexableRow,
    _batched,
    document_id_to_point_id,
    upsert_documents,
)
from ragtime.index.schema import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

# --- document_id_to_point_id --------------------------------------------- #


def test_point_id_is_uuid_string() -> None:
    pid = document_id_to_point_id("abc1234567890def")
    # Doit être un UUID valide
    parsed = uuid.UUID(pid)
    assert isinstance(parsed, uuid.UUID)


def test_point_id_is_deterministic() -> None:
    assert document_id_to_point_id("xyz") == document_id_to_point_id("xyz")


def test_point_id_differs_for_different_document_id() -> None:
    assert document_id_to_point_id("a") != document_id_to_point_id("b")


# --- IndexableRow.to_point ------------------------------------------------ #


def _make_row(**overrides) -> IndexableRow:
    base = {
        "document_id": "deadbeefcafe1234",
        "source_ticket_id": 7,
        "content": "[CATEGORY] X",
        "dense": [0.1] * 1024,
        "sparse": SparseVector(indices=[1, 2], values=[0.5, 0.3]),
        "payload": {"category": "X", "language": "English"},
    }
    base.update(overrides)
    return IndexableRow(**base)


def test_to_point_has_named_vectors() -> None:
    pt = _make_row().to_point()
    assert DENSE_VECTOR_NAME in pt.vector
    assert SPARSE_VECTOR_NAME in pt.vector


def test_to_point_payload_includes_document_id_and_content() -> None:
    pt = _make_row().to_point()
    assert pt.payload["document_id"] == "deadbeefcafe1234"
    assert pt.payload["source_ticket_id"] == 7
    assert pt.payload["content"] == "[CATEGORY] X"
    assert pt.payload["category"] == "X"  # champ utilisateur conservé


def test_to_point_id_matches_doc_id_uuid5() -> None:
    pt = _make_row(document_id="hello").to_point()
    assert pt.id == document_id_to_point_id("hello")


# --- _batched ------------------------------------------------------------ #


def test_batched_returns_full_then_partial() -> None:
    rows = [_make_row(document_id=f"d{i:02d}") for i in range(7)]
    batches = list(_batched(rows, 3))
    assert [len(b) for b in batches] == [3, 3, 1]


def test_batched_empty_input() -> None:
    assert list(_batched([], 3)) == []


# --- upsert_documents (avec QdrantClient mocké) -------------------------- #


def test_upsert_documents_calls_upsert_in_batches() -> None:
    client = MagicMock()
    rows = [_make_row(document_id=f"d{i:02d}") for i in range(5)]
    total = upsert_documents(client, rows, collection="X", batch_size=2)
    assert total == 5
    # 5 rows en batches de 2 → 3 appels
    assert client.upsert.call_count == 3


def test_upsert_documents_propagates_error_as_index_error() -> None:
    client = MagicMock()
    client.upsert.side_effect = RuntimeError("network down")
    with pytest.raises(IndexError):
        upsert_documents(client, [_make_row()], collection="X")
