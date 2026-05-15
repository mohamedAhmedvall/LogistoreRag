"""Test d'intégration : hybrid_search bout-en-bout sur les données réelles
indexées dans le container Docker Qdrant.

**Pré-requis** : avoir lancé en amont :
    1. scripts/run_ingestion.py
    2. scripts/run_embedding.py --limit 1000
    3. scripts/run_indexation.py

Ces tests sont automatiquement skip si la collection est introuvable ou
si Qdrant n'est pas joignable, pour ne pas casser une CI sans services.
"""

from __future__ import annotations

import pytest

from ragtime.config import get_settings
from ragtime.index.client import build_qdrant_client
from ragtime.models import SearchFilters, SearchQuery
from ragtime.search.hybrid import hybrid_search

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    settings = get_settings()
    try:
        c = build_qdrant_client()
        # Sanity check : la collection doit exister.
        if not c.collection_exists(settings.qdrant_collection_name):
            pytest.skip("Collection tickets_support introuvable. Lancer run_indexation.py d'abord.")
        info = c.get_collection(settings.qdrant_collection_name)
        if info.points_count == 0:
            pytest.skip("Collection vide.")
        return c
    except Exception as exc:  # pragma: no cover - dépend du serveur
        pytest.skip(f"Qdrant indisponible : {exc}")


def test_search_returns_results(client) -> None:
    """Une requête anglaise simple doit retourner ≥ 1 résultat."""
    resp = hybrid_search(SearchQuery(query="cannot login to my account", top_k=5), client=client)
    assert resp.total >= 1
    assert all(r.score > 0 for r in resp.results)
    assert resp.took_ms > 0


def test_search_with_filter_respects_category(client) -> None:
    """Filtre catégorie : seuls les tickets de la catégorie demandée."""
    resp = hybrid_search(
        SearchQuery(
            query="account access blocked",
            top_k=5,
            filters=SearchFilters(category=["Login Issue"]),
        ),
        client=client,
    )
    assert resp.total >= 1
    for r in resp.results:
        assert r.payload.get("category") == "Login Issue"


def test_search_multilingual_french(client) -> None:
    """Une requête française doit retourner des résultats pertinents."""
    resp = hybrid_search(
        SearchQuery(query="mot de passe oublié connexion impossible", top_k=5),
        client=client,
    )
    assert resp.total >= 1


def test_search_returns_results_ordered_by_score(client) -> None:
    """Les résultats sont triés par score décroissant (RRF)."""
    resp = hybrid_search(SearchQuery(query="payment failed transaction", top_k=10), client=client)
    scores = [r.score for r in resp.results]
    assert scores == sorted(scores, reverse=True)
