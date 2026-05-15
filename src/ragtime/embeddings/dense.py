"""Wrapper d'embedding dense (E5-large multilingue via fastembed).

Voir `docs/ADR/001-embeddings-substitut.md` pour la justification du
remplacement de BGE-M3 par `intfloat/multilingual-e5-large`.

Les modèles E5 attendent un préfixe selon le mode :
- Indexation : "passage: <texte>"
- Requête    : "query: <texte>"
Le préfixe est appliqué automatiquement par `embed_documents` /
`embed_queries`. Les appelants ne doivent **jamais** préfixer eux-mêmes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from ragtime.config import get_settings
from ragtime.exceptions import RAGtimeError
from ragtime.logging_setup import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)

# Vecteur dense = liste de floats (ou np.ndarray). On expose list[float] côté API
# pour découpler le code applicatif de numpy / Qdrant.
DenseVector = list[float]


def _prefix_passage(texts: Iterable[str]) -> list[str]:
    return [f"passage: {t}" for t in texts]


def _prefix_query(texts: Iterable[str]) -> list[str]:
    return [f"query: {t}" for t in texts]


class DenseEmbedder:
    """Wrapper paresseux autour de `fastembed.TextEmbedding`.

    Le modèle ONNX est téléchargé et chargé au premier appel
    (`_get_model()`). Cela évite de payer ~3 s de chargement à l'import.

    Le batching est délégué à fastembed (`batch_size` paramètre interne).
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.cache_dir = cache_dir
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover (env-dependant)
                raise RAGtimeError("fastembed n'est pas installé. Voir requirements.txt.") from exc

            logger.info("dense.load_model", model=self.model_name)
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._model

    @property
    def dimension(self) -> int:
        """Dimensionnalité des vecteurs produits par le modèle courant."""
        model = self._get_model()
        # fastembed >= 0.3 expose les méta dans le ModelDescription.
        for m in model.list_supported_models():
            if m["model"] == self.model_name:
                dim = m.get("dim")
                if dim is None:  # pragma: no cover
                    raise RAGtimeError(f"Modèle '{self.model_name}' sans dim métadata.")
                return int(dim)
        raise RAGtimeError(f"Modèle d'embedding inconnu : {self.model_name}")

    def _embed(self, texts: list[str]) -> Iterator[DenseVector]:
        model = self._get_model()
        # `embed` accepte batch_size en kwarg ; il yield des np.ndarray.
        for vec in model.embed(texts, batch_size=self.batch_size):
            yield _ndarray_to_list(vec)

    def embed_documents(self, texts: Iterable[str]) -> Iterator[DenseVector]:
        """Embed des passages destinés à être indexés (préfixe 'passage:')."""
        prefixed = _prefix_passage(texts)
        yield from self._embed(prefixed)

    def embed_queries(self, texts: Iterable[str]) -> Iterator[DenseVector]:
        """Embed des requêtes utilisateur (préfixe 'query:')."""
        prefixed = _prefix_query(texts)
        yield from self._embed(prefixed)


def _ndarray_to_list(vec: np.ndarray | list[float]) -> DenseVector:
    """Convertit un np.ndarray en list[float] (idempotent si déjà liste)."""
    try:
        return vec.tolist()  # type: ignore[union-attr]
    except AttributeError:
        return list(vec)


@lru_cache(maxsize=1)
def get_dense_embedder() -> DenseEmbedder:
    """Singleton d'embedder dense pour éviter de recharger le modèle ONNX."""
    return DenseEmbedder()
