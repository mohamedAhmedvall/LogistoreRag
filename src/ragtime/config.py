"""Configuration centralisée via pydantic-settings.

Toutes les valeurs sont lues depuis l'environnement et/ou un fichier `.env`
à la racine du projet. Aucune valeur sensible n'est hardcodée.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configuration applicative complète.

    Champs groupés par domaine : Qdrant, embeddings, reranker, LLM,
    API, dataset, recherche, évaluation.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Qdrant ---
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: SecretStr | None = Field(default=None)
    qdrant_collection_name: str = Field(default="tickets_support")

    # --- Embeddings ---
    # Voir docs/ADR/001-embeddings-substitut.md pour le choix du modèle.
    embedding_model: str = Field(default="intfloat/multilingual-e5-large")
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    embedding_device: Literal["cpu", "cuda", "mps"] = Field(default="cpu")
    sparse_embedding_model: str = Field(default="Qdrant/bm25")

    # --- Reranker ---
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    reranker_enabled: bool = Field(default=True)

    # --- LLM (OpenRouter) ---
    openrouter_api_key: SecretStr | None = Field(default=None)
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    llm_model: str = Field(default="anthropic/claude-3.5-sonnet")

    # --- API FastAPI ---
    api_host: str = Field(default="0.0.0.0")  # noqa: S104 (bind dev/docker)
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_key: SecretStr = Field(default=SecretStr("change-me-in-production"))
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")

    # --- Dataset ---
    dataset_name: str = Field(
        default="mirzayasirabdullah07/customer-support-tickets-dataset-200k-records"
    )
    dataset_path: Path = Field(default=Path("data/raw/customer_support_tickets.csv"))
    dataset_sample_size: int | None = Field(default=10_000, ge=1)

    # --- Recherche hybride ---
    hybrid_top_k_retrieve: int = Field(default=50, ge=1, le=500)
    hybrid_top_k_final: int = Field(default=10, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1)

    # --- Évaluation ---
    eval_golden_path: Path = Field(default=Path("data/golden/queries.json"))
    eval_judge_model: str = Field(default="anthropic/claude-3.5-sonnet")
    eval_output_dir: Path = Field(default=Path("data/evaluation"))

    @field_validator("dataset_path", "eval_golden_path", "eval_output_dir", mode="after")
    @classmethod
    def _resolve_relative_to_project(cls, v: Path) -> Path:
        """Résout les chemins relatifs par rapport à la racine du projet."""
        return v if v.is_absolute() else PROJECT_ROOT / v

    @property
    def project_root(self) -> Path:
        """Chemin absolu vers la racine du projet."""
        return PROJECT_ROOT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retourne l'instance singleton des settings (cacheée)."""
    return Settings()
