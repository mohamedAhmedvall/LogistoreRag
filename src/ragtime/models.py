"""Schémas Pydantic — contrats de données du projet.

Ces modèles définissent le contrat entre les couches : ingestion → index →
recherche → API → frontend. Toute évolution de schéma doit passer par ce
fichier (et non par des dicts opaques).

Convention :
- `Ticket` : forme source (bronze) — proche du CSV Kaggle, après dedup.
- `IndexableDocument` : forme gold — prête à être indexée dans Qdrant.
- `SearchQuery` / `SearchResult` / `SearchResponse` : contrats de l'API search.
- `SynthesisRequest` / `SynthesisResponse` : contrats de l'API synthesize.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator

# --------------------------------------------------------------------------- #
# Énumérations métier (calées sur les valeurs du dataset)                     #
# --------------------------------------------------------------------------- #


class Priority(StrEnum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    URGENT = "Urgent"


class TicketStatus(StrEnum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    PENDING = "Pending"
    RESOLVED = "Resolved"
    CLOSED = "Closed"


class Channel(StrEnum):
    EMAIL = "Email"
    PHONE = "Phone"
    CHAT = "Chat"
    SOCIAL = "Social"
    WEB = "Web"


# --------------------------------------------------------------------------- #
# Modèle source (bronze → silver)                                             #
# --------------------------------------------------------------------------- #


class Ticket(BaseModel):
    """Ticket brut tel que chargé depuis le dataset Kaggle, après normalisation.

    Les champs PII (`customer_name`, `customer_email`) ne sont **pas** conservés
    par défaut : ils sont droppés à l'étape `normalizer`.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    ticket_id: int = Field(..., description="Identifiant numérique source")
    product: str
    category: str
    issue_description: str = Field(default="")
    resolution_notes: str = Field(default="")
    priority: str
    status: str
    channel: str
    region: str

    # Métadonnées techniques / contextuelles
    operating_system: str | None = None
    browser: str | None = None
    payment_method: str | None = None
    language: str = Field(default="English")
    preferred_contact_time: str | None = None

    # Mesures business
    customer_age: int | None = Field(default=None, ge=0, le=120)
    customer_gender: str | None = None
    subscription_type: str | None = None
    customer_tenure_months: NonNegativeInt | None = None
    previous_tickets: NonNegativeInt = Field(default=0)
    customer_satisfaction_score: int | None = Field(default=None, ge=1, le=5)
    first_response_time_hours: float | None = Field(default=None, ge=0)
    resolution_time_hours: float | None = Field(default=None, ge=0)
    issue_complexity_score: int | None = Field(default=None, ge=1, le=10)
    customer_segment: str | None = None

    # Dates
    ticket_created_date: date | None = None
    ticket_resolved_date: date | None = None

    # Flags
    escalated: bool = Field(default=False)
    sla_breached: bool = Field(default=False)

    @field_validator("escalated", "sla_breached", mode="before")
    @classmethod
    def _yesno_to_bool(cls, v: Any) -> Any:
        """Convertit les "Yes"/"No" du CSV en bool."""
        if isinstance(v, str):
            return v.strip().lower() in {"yes", "true", "1", "y"}
        return v


# --------------------------------------------------------------------------- #
# Modèle gold (prêt à indexer)                                                #
# --------------------------------------------------------------------------- #


class IndexableDocument(BaseModel):
    """Document prêt à être upserté dans Qdrant.

    Le `document_id` est un hash déterministe du contenu : relancer
    l'ingestion sur les mêmes données ne crée pas de doublons.

    Le `content` est le texte qui sera embarqué (BGE-M3). Les champs
    discriminants (subject, description, resolution) sont préservés
    séparément dans `payload` pour pouvoir surligner les zones de match.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(..., description="Hash déterministe SHA-256 (préfixe)")
    source_ticket_id: int
    content: str = Field(..., min_length=1, description="Texte concaténé à embarquer")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées filtrables côté Qdrant",
    )


# --------------------------------------------------------------------------- #
# API : recherche                                                              #
# --------------------------------------------------------------------------- #


class SearchFilters(BaseModel):
    """Filtres optionnels appliqués côté Qdrant (payload-aware)."""

    model_config = ConfigDict(extra="forbid")

    category: list[str] | None = None
    product: list[str] | None = None
    priority: list[str] | None = None
    status: list[str] | None = None
    channel: list[str] | None = None
    region: list[str] | None = None
    language: list[str] | None = None
    escalated: bool | None = None
    sla_breached: bool | None = None
    created_after: date | None = None
    created_before: date | None = None


class SearchQuery(BaseModel):
    """Requête de recherche envoyée à l'API."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: SearchFilters | None = None
    rerank: bool | None = Field(
        default=None,
        description="Force l'activation/désactivation du rerank (None = défaut config).",
    )


class SearchResult(BaseModel):
    """Résultat unitaire de recherche."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_ticket_id: int
    score: float = Field(..., description="Score de pertinence final (post-fusion/rerank)")
    score_dense: float | None = None
    score_sparse: float | None = None
    score_rerank: float | None = None
    content: str
    payload: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Réponse complète de l'API search."""

    model_config = ConfigDict(extra="forbid")

    query: str
    total: int
    took_ms: float
    results: list[SearchResult]
    reranked: bool = False


# --------------------------------------------------------------------------- #
# API : synthèse LLM                                                          #
# --------------------------------------------------------------------------- #


class SynthesisRequest(BaseModel):
    """Requête de synthèse LLM sur des résultats de recherche."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000)
    results: list[SearchResult] = Field(..., min_length=1, max_length=20)
    max_tokens: int = Field(default=800, ge=64, le=4000)


class Citation(BaseModel):
    """Citation d'un document source dans la synthèse."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_ticket_id: int
    span: str | None = Field(default=None, description="Extrait textuel cité")


class SynthesisResponse(BaseModel):
    """Réponse synthétisée par le LLM, accompagnée des citations."""

    model_config = ConfigDict(extra="forbid")

    query: str
    answer: str
    citations: list[Citation]
    model: str
    took_ms: float
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=None))
