"""Embeddings : wrappers dense (E5-large) et sparse (BM25).

Voir `docs/ADR/001-embeddings-substitut.md`.
"""

from ragtime.embeddings.dense import DenseEmbedder, DenseVector, get_dense_embedder
from ragtime.embeddings.sparse import SparseEmbedder, SparseVector, get_sparse_embedder

__all__ = [
    "DenseEmbedder",
    "DenseVector",
    "SparseEmbedder",
    "SparseVector",
    "get_dense_embedder",
    "get_sparse_embedder",
]
