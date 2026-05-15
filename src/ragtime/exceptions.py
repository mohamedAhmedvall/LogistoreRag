"""Hiérarchie d'exceptions custom pour RAG-time / LogiStore.

Toute exception applicative hérite de `RAGtimeError`. Cela permet aux
appelants de filtrer proprement les erreurs métier des erreurs système.
"""

from __future__ import annotations


class RAGtimeError(Exception):
    """Exception racine pour toutes les erreurs applicatives RAG-time."""


class ConfigError(RAGtimeError):
    """Erreur de configuration (variable manquante, valeur invalide)."""


class IngestionError(RAGtimeError):
    """Erreur durant le pipeline d'ingestion (chargement, normalisation, chunking)."""


class IndexError(RAGtimeError):  # noqa: A001 (shadowing builtin is intentional and scoped)
    """Erreur d'interaction avec le vector store (Qdrant)."""


class SearchError(RAGtimeError):
    """Erreur durant la phase de recherche (hybride, rerank, filtres)."""


class LLMError(RAGtimeError):
    """Erreur d'appel au LLM (OpenRouter, synthèse, jugement)."""


class EvaluationError(RAGtimeError):
    """Erreur durant la campagne d'évaluation."""
