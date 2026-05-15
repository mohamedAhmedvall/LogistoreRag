"""Tests unitaires pour `ragtime.config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragtime.config import PROJECT_ROOT, Settings, get_settings


def test_settings_defaults_load() -> None:
    """Les valeurs par défaut se chargent sans erreur."""
    s = Settings(_env_file=None)
    assert s.qdrant_collection_name == "tickets_support"
    # Voir docs/ADR/001-embeddings-substitut.md
    assert s.embedding_model == "intfloat/multilingual-e5-large"
    assert s.sparse_embedding_model == "Qdrant/bm25"
    assert s.embedding_device in {"cpu", "cuda", "mps"}
    assert s.api_port == 8000
    assert s.hybrid_top_k_final <= s.hybrid_top_k_retrieve


def test_settings_validation_rejects_bad_device() -> None:
    """Le champ embedding_device est contraint."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, embedding_device="tpu")  # type: ignore[arg-type]


def test_settings_paths_resolved_absolute() -> None:
    """Les chemins relatifs doivent être résolus par rapport au projet."""
    s = Settings(_env_file=None)
    assert s.dataset_path.is_absolute()
    assert s.eval_golden_path.is_absolute()
    assert s.eval_output_dir.is_absolute()


def test_settings_paths_under_project_root() -> None:
    """Les chemins relatifs résolus restent sous PROJECT_ROOT."""
    s = Settings(_env_file=None)
    assert PROJECT_ROOT in s.dataset_path.parents
    assert PROJECT_ROOT in s.eval_golden_path.parents


def test_settings_absolute_path_preserved(tmp_path: Path) -> None:
    """Un chemin absolu fourni est conservé tel quel."""
    abs_csv = tmp_path / "elsewhere.csv"
    s = Settings(_env_file=None, dataset_path=abs_csv)
    assert s.dataset_path == abs_csv


def test_get_settings_is_cached() -> None:
    """get_settings retourne toujours la même instance."""
    a = get_settings()
    b = get_settings()
    assert a is b


def test_topk_constraints_enforced() -> None:
    """top_k_final ne peut pas dépasser les bornes définies."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, hybrid_top_k_final=0)
    with pytest.raises(ValueError):
        Settings(_env_file=None, hybrid_top_k_final=999)
