"""Tests unitaires pour `ragtime.embeddings` (avec mock du modèle ONNX).

Les tests ne déclenchent **aucun** téléchargement de modèle : le client
fastembed est mocké via `monkeypatch` pour rester rapide et hors-ligne.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ragtime.embeddings.dense import DenseEmbedder, _prefix_passage, _prefix_query
from ragtime.embeddings.sparse import SparseEmbedder, SparseVector
from ragtime.exceptions import RAGtimeError

# --- Helpers ------------------------------------------------------------- #


class _FakeArray:
    """Mini ndarray-like avec .tolist()."""

    def __init__(self, data: list) -> None:
        self._data = data

    def tolist(self) -> list:
        return list(self._data)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._data),)


# --- DenseEmbedder ------------------------------------------------------- #


def test_prefix_helpers() -> None:
    assert _prefix_passage(["a", "b"]) == ["passage: a", "passage: b"]
    assert _prefix_query(["q"]) == ["query: q"]


def test_dense_embedder_uses_settings_by_default() -> None:
    e = DenseEmbedder()
    assert e.model_name  # vient de settings, non vide
    assert e.batch_size > 0


def test_dense_embed_documents_applies_passage_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []
    fake_model = MagicMock()
    fake_model.embed = lambda texts, batch_size: (
        captured.append(list(texts)) or iter([_FakeArray([0.1] * 8) for _ in texts])
    )
    embedder = DenseEmbedder(model_name="intfloat/multilingual-e5-large", batch_size=8)
    monkeypatch.setattr(embedder, "_get_model", lambda: fake_model)

    out = list(embedder.embed_documents(["hello", "world"]))
    assert captured[0] == ["passage: hello", "passage: world"]
    assert len(out) == 2
    assert out[0] == [0.1] * 8


def test_dense_embed_queries_applies_query_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []
    fake_model = MagicMock()
    fake_model.embed = lambda texts, batch_size: (
        captured.append(list(texts)) or iter([_FakeArray([0.2] * 4) for _ in texts])
    )
    embedder = DenseEmbedder(model_name="intfloat/multilingual-e5-large")
    monkeypatch.setattr(embedder, "_get_model", lambda: fake_model)

    out = list(embedder.embed_queries(["search this"]))
    assert captured[0] == ["query: search this"]
    assert out == [[0.2] * 4]


def test_dense_model_is_loaded_lazily() -> None:
    """`_model` n'est initialisé qu'au premier appel — pas à l'instanciation."""
    e = DenseEmbedder()
    assert e._model is None


def test_dense_dimension_unknown_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    e = DenseEmbedder(model_name="not/a/real/model")
    fake = MagicMock()
    fake.list_supported_models = lambda: [{"model": "some/other/model", "dim": 768}]
    monkeypatch.setattr(e, "_get_model", lambda: fake)
    with pytest.raises(RAGtimeError):
        _ = e.dimension


def test_dense_dimension_known_model(monkeypatch: pytest.MonkeyPatch) -> None:
    e = DenseEmbedder(model_name="x/y")
    fake = MagicMock()
    fake.list_supported_models = lambda: [{"model": "x/y", "dim": 1024}]
    monkeypatch.setattr(e, "_get_model", lambda: fake)
    assert e.dimension == 1024


# --- SparseEmbedder + SparseVector --------------------------------------- #


def test_sparse_vector_validates_alignment() -> None:
    with pytest.raises(RAGtimeError):
        SparseVector(indices=[1, 2], values=[0.5])


def test_sparse_vector_nnz() -> None:
    v = SparseVector(indices=[3, 7, 12], values=[1.0, 0.5, 0.2])
    assert v.nnz == 3


def test_sparse_embedder_yields_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Emb:
        indices = _FakeArray([5, 9, 13])
        values = _FakeArray([0.9, 0.4, 0.1])

    fake_model = MagicMock()
    fake_model.embed = lambda texts, batch_size: iter([_Emb() for _ in texts])

    embedder = SparseEmbedder(model_name="Qdrant/bm25")
    monkeypatch.setattr(embedder, "_get_model", lambda: fake_model)

    out = list(embedder.embed(["t1", "t2"]))
    assert len(out) == 2
    assert out[0].indices == [5, 9, 13]
    assert out[0].values == [0.9, 0.4, 0.1]
    assert out[0].nnz == 3


def test_sparse_embedder_lazy_load() -> None:
    e = SparseEmbedder()
    assert e._model is None


# --- Conversion helpers (couvre les deux chemins ndarray / list) -------- #


def test_int_list_from_ndarray_or_list() -> None:
    from ragtime.embeddings.sparse import _to_float_list, _to_int_list

    assert _to_int_list(_FakeArray([1.0, 2.0, 3.0])) == [1, 2, 3]
    assert _to_int_list([4, 5]) == [4, 5]
    assert _to_float_list(_FakeArray([1, 2])) == [1.0, 2.0]
    assert _to_float_list([3, 4]) == [3.0, 4.0]


def test_ndarray_to_list_helper() -> None:
    from ragtime.embeddings.dense import _ndarray_to_list

    assert _ndarray_to_list(_FakeArray([0.1, 0.2])) == [0.1, 0.2]
    assert _ndarray_to_list([0.3, 0.4]) == [0.3, 0.4]
