"""Wrapper d'embedding sparse (BM25 via fastembed).

BM25 produit pour chaque texte un vecteur creux représenté par deux listes
parallèles : `indices` (ids de tokens) et `values` (poids BM25).
C'est le format attendu par Qdrant pour les vecteurs sparse.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ragtime.config import get_settings
from ragtime.exceptions import RAGtimeError
from ragtime.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SparseVector:
    """Représentation d'un vecteur sparse, format compatible Qdrant."""

    indices: list[int]
    values: list[float]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise RAGtimeError(
                f"Tailles incohérentes: indices={len(self.indices)} values={len(self.values)}"
            )

    @property
    def nnz(self) -> int:
        """Nombre d'éléments non nuls."""
        return len(self.indices)


class SparseEmbedder:
    """Wrapper paresseux autour de `fastembed.SparseTextEmbedding`."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.sparse_embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.cache_dir = cache_dir
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed import SparseTextEmbedding
            except ImportError as exc:  # pragma: no cover
                raise RAGtimeError("fastembed n'est pas installé. Voir requirements.txt.") from exc
            logger.info("sparse.load_model", model=self.model_name)
            self._model = SparseTextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._model

    def embed(self, texts: Iterable[str]) -> Iterator[SparseVector]:
        """Embed une séquence de textes en vecteurs sparse.

        Args:
            texts: Itérable de chaînes à embarquer.

        Yields:
            Des `SparseVector` (indices + values), un par texte.
        """
        model = self._get_model()
        for emb in model.embed(list(texts), batch_size=self.batch_size):
            # fastembed renvoie un objet avec .indices et .values (np arrays).
            indices = _to_int_list(emb.indices)
            values = _to_float_list(emb.values)
            yield SparseVector(indices=indices, values=values)


def _to_int_list(x: Any) -> list[int]:
    try:
        return [int(v) for v in x.tolist()]
    except AttributeError:
        return [int(v) for v in x]


def _to_float_list(x: Any) -> list[float]:
    try:
        return [float(v) for v in x.tolist()]
    except AttributeError:
        return [float(v) for v in x]


@lru_cache(maxsize=1)
def get_sparse_embedder() -> SparseEmbedder:
    """Singleton d'embedder sparse."""
    return SparseEmbedder()
