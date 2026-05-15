"""Fixtures pytest partagées entre tests unitaires et d'intégration."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from ragtime.config import Settings, get_settings


@pytest.fixture
def settings() -> Settings:
    """Settings effectifs (lus depuis .env ou variables d'environnement)."""
    return get_settings()


@pytest.fixture
def clean_settings_cache() -> Iterator[None]:
    """Vide le cache lru_cache de get_settings entre deux tests qui le mutent."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
