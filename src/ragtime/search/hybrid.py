"""Recherche hybride dense + sparse + Reciprocal Rank Fusion (RRF).

Une seule requête Qdrant via `query_points(prefetch=..., query=FusionQuery(rrf))` :
- Prefetch dense : top-`top_k_retrieve` plus proches voisins (cosine).
- Prefetch sparse : top-`top_k_retrieve` selon BM25.
- Fusion RRF côté serveur → top-`top_k_final` global.

C'est l'approche moderne préconisée par Qdrant pour le hybride. Aucune
fusion côté client : Qdrant calcule la fusion sur ses propres résultats
de prefetch (plus rapide et plus cohérent qu'un merge Python).
"""

from __future__ import annotations

import time

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Fusion,
    FusionQuery,
    Prefetch,
)
from qdrant_client.http.models import (
    SparseVector as QdrantSparseVector,
)

from ragtime.config import get_settings
from ragtime.embeddings import get_dense_embedder, get_sparse_embedder
from ragtime.exceptions import SearchError
from ragtime.index.client import get_qdrant_client
from ragtime.index.schema import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from ragtime.logging_setup import get_logger
from ragtime.models import SearchFilters, SearchQuery, SearchResponse, SearchResult
from ragtime.search.filters import build_qdrant_filter

logger = get_logger(__name__)


def hybrid_search(
    query: SearchQuery,
    client: QdrantClient | None = None,
    collection: str | None = None,
) -> SearchResponse:
    """Exécute une recherche hybride RRF sur la collection Qdrant.

    Args:
        query: Requête API (contient le texte, top_k, filtres, flag rerank).
        client: Client Qdrant (override pour tests). Si None, singleton.
        collection: Nom de collection (override). Si None, depuis Settings.

    Returns:
        `SearchResponse` avec la liste ordonnée des `SearchResult`.
    """
    settings = get_settings()
    qclient = client or get_qdrant_client()
    coll = collection or settings.qdrant_collection_name

    t0 = time.perf_counter()

    # 1) Embedder la requête (dense + sparse).
    dense_embedder = get_dense_embedder()
    sparse_embedder = get_sparse_embedder()
    [dense_vec] = list(dense_embedder.embed_queries([query.query]))
    [sparse_vec] = list(sparse_embedder.embed([query.query]))

    qfilter = build_qdrant_filter(query.filters)
    top_k_retrieve = settings.hybrid_top_k_retrieve

    try:
        result = qclient.query_points(
            collection_name=coll,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using=DENSE_VECTOR_NAME,
                    limit=top_k_retrieve,
                    filter=qfilter,
                ),
                Prefetch(
                    query=QdrantSparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                    using=SPARSE_VECTOR_NAME,
                    limit=top_k_retrieve,
                    filter=qfilter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=query.top_k,
            with_payload=True,
        )
    except Exception as exc:
        raise SearchError(f"Échec hybrid_search: {exc}") from exc

    points = result.points
    took_ms = (time.perf_counter() - t0) * 1000.0

    results: list[SearchResult] = []
    for p in points:
        payload = dict(p.payload or {})
        content = payload.pop("content", "")
        document_id = payload.pop("document_id", str(p.id))
        source_ticket_id = int(payload.pop("source_ticket_id", 0))
        results.append(
            SearchResult(
                document_id=document_id,
                source_ticket_id=source_ticket_id,
                score=float(p.score),
                content=content,
                payload=payload,
            )
        )

    logger.info(
        "search.hybrid.done",
        query_len=len(query.query),
        results=len(results),
        took_ms=round(took_ms, 1),
        has_filters=qfilter is not None,
    )

    return SearchResponse(
        query=query.query,
        total=len(results),
        took_ms=took_ms,
        results=results,
        reranked=False,
    )


def search(
    text: str,
    top_k: int = 10,
    filters: SearchFilters | None = None,
) -> SearchResponse:
    """Sucre syntaxique : construit la `SearchQuery` et appelle `hybrid_search`."""
    return hybrid_search(SearchQuery(query=text, top_k=top_k, filters=filters))
