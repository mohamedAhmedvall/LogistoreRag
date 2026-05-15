"""Wrapper léger autour de `qdrant_client.QdrantClient`.

Centralise la construction du client à partir des settings (URL, API key,
timeout) et fournit un point d'entrée unique pour les autres modules
(`schema`, `indexer`, `search`).

Le client supporte trois modes :
- HTTP/REST distant : `settings.qdrant_url = "http://localhost:6333"`
- In-memory : `url = ":memory:"` (pour les tests d'intégration)
- Embedded file : `url = "file:///path/to/qdrant_storage"` (rarement utilisé)
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient

from ragtime.config import get_settings
from ragtime.logging_setup import get_logger

logger = get_logger(__name__)


def build_qdrant_client(
    url: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> QdrantClient:
    """Construit un `QdrantClient` à partir des settings (ou des overrides).

    Args:
        url: URL du serveur Qdrant. Si None, lue depuis Settings.
        api_key: Clé d'API Qdrant (cloud). Si None, lue depuis Settings.
        timeout: Timeout HTTP en secondes.

    Returns:
        Une instance de `QdrantClient` prête à l'emploi.
    """
    settings = get_settings()
    target_url = url or settings.qdrant_url
    target_key = api_key or (
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    )

    # Mode in-memory : pas d'URL HTTP à passer.
    if target_url == ":memory:":
        logger.info("qdrant.client.in_memory")
        return QdrantClient(location=":memory:")

    logger.info("qdrant.client.connect", url=target_url, has_api_key=target_key is not None)
    return QdrantClient(url=target_url, api_key=target_key, timeout=timeout)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Singleton du client Qdrant pour le code applicatif (API, scripts)."""
    return build_qdrant_client()
