"""Tests unitaires pour la hiérarchie d'exceptions."""

from __future__ import annotations

import pytest

from ragtime.exceptions import (
    ConfigError,
    EvaluationError,
    IndexError,
    IngestionError,
    LLMError,
    RAGtimeError,
    SearchError,
)


@pytest.mark.parametrize(
    "exc_cls",
    [ConfigError, IngestionError, IndexError, SearchError, LLMError, EvaluationError],
)
def test_all_errors_inherit_from_ragtime_error(exc_cls: type[Exception]) -> None:
    """Toute exception métier doit hériter de RAGtimeError."""
    assert issubclass(exc_cls, RAGtimeError)


def test_exception_can_be_raised_and_caught() -> None:
    """Sanity check : on peut lever et attraper en remontant à la racine."""
    with pytest.raises(RAGtimeError):
        raise IngestionError("boom")
