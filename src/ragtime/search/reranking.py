"""Reranking optionnel via BGE-reranker-v2-m3 (Incrément ultérieur).

Stub pour conserver la signature attendue par CLAUDE.md §3. L'implémentation
réelle sera ajoutée à l'Incrément 9 (avec téléchargement du cross-encoder).
"""

from __future__ import annotations

from ragtime.models import SearchResult


def rerank(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Rerank les résultats par cross-encoder (non implémenté à ce stade).

    À l'Incrément 9, on remplacera ce stub par un appel à BGE-reranker-v2-m3.
    Pour l'instant, on retourne les résultats inchangés.
    """
    _ = query  # placeholder
    return results
