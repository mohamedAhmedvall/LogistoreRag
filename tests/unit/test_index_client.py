"""Tests unitaires pour `ragtime.index.client` (sans serveur)."""

from __future__ import annotations

from unittest.mock import patch

from qdrant_client import QdrantClient

from ragtime.index.client import build_qdrant_client


def test_build_qdrant_client_in_memory_mode() -> None:
    """`:memory:` doit déclencher le mode local de qdrant-client."""
    client = build_qdrant_client(url=":memory:")
    assert isinstance(client, QdrantClient)
    # Sanity : on peut lister les collections (vide).
    cols = client.get_collections()
    assert cols.collections == []


def test_build_qdrant_client_remote_signature() -> None:
    """Construit un client HTTP avec les bons args (mocké pour ne pas pinger un serveur)."""
    with patch("ragtime.index.client.QdrantClient") as mock_cls:
        build_qdrant_client(url="http://example.test:6333", api_key="secret", timeout=12.0)
        mock_cls.assert_called_once_with(
            url="http://example.test:6333",
            api_key="secret",
            timeout=12.0,
        )
