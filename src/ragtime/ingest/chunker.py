"""Composition d'un IndexableDocument par ticket.

Notre dataset est constitué de **tickets courts** (issue + resolution
chacun < 200 caractères en moyenne), bien en-deçà de la fenêtre BGE-M3.
Le "chunking" se réduit donc à une stratégie de composition du texte
à embarquer, et non à un découpage.

Stratégie : un IndexableDocument par ticket avec :
- `content` = texte concaténé tagué `[CATEGORY] ... [PRODUCT] ... [ISSUE] ... [RESOLUTION] ...`
- `payload` = tous les champs filtrables (préservés tels quels).
- `document_id` = sha256(issue + "\\n" + resolution)[:16] → déterministe, dédup naturelle.
"""

from __future__ import annotations

import hashlib

from ragtime.models import IndexableDocument, Ticket

_DOC_ID_LEN = 16  # 16 hex chars = 64 bits → collisions improbables sur 200K docs.


def compute_document_id(
    category: str,
    product: str,
    language: str,
    issue: str,
    resolution: str,
) -> str:
    """Hash SHA-256 déterministe sur la combinaison contextuelle, tronqué à 16 hex chars.

    Le dataset Kaggle étant extrêmement répétitif (10 issues × 10 resolutions
    uniques pour 200K tickets), on enrichit la clé de dédup avec la catégorie,
    le produit et la langue. Cela permet d'obtenir un index riche tout en
    conservant l'idempotence : deux tickets avec exactement la même tuple
    (category, product, language, issue, resolution) produisent le même ID.
    """
    payload = "|".join(s.strip() for s in (category, product, language, issue, resolution)).encode()
    return hashlib.sha256(payload).hexdigest()[:_DOC_ID_LEN]


def compose_content(ticket: Ticket) -> str:
    """Construit le texte à embarquer à partir d'un ticket.

    Les champs contextuels (catégorie, produit, langue, région) sont
    intégrés au content : ils enrichissent l'embedding et reflètent le
    fait qu'une même paire (issue, resolution) peut avoir un sens
    différent selon le produit ou la région.
    """
    parts: list[str] = [
        f"[CATEGORY] {ticket.category}",
        f"[PRODUCT] {ticket.product}",
        f"[LANGUAGE] {ticket.language}",
        f"[REGION] {ticket.region}",
    ]
    if ticket.issue_description.strip():
        parts.append(f"[ISSUE] {ticket.issue_description.strip()}")
    if ticket.resolution_notes.strip():
        parts.append(f"[RESOLUTION] {ticket.resolution_notes.strip()}")
    return "\n".join(parts)


def build_payload(ticket: Ticket) -> dict[str, object]:
    """Construit le payload filtrable côté Qdrant.

    On préserve :
    - les champs textuels de catégorisation (filtres + surlignage),
    - les flags business (escalated, sla_breached),
    - les dates et métriques.
    On exclut les champs PII (déjà droppés au niveau du modèle Ticket).
    """
    return {
        "ticket_id": ticket.ticket_id,
        "category": ticket.category,
        "product": ticket.product,
        "priority": ticket.priority,
        "status": ticket.status,
        "channel": ticket.channel,
        "region": ticket.region,
        "language": ticket.language,
        "issue_description": ticket.issue_description,
        "resolution_notes": ticket.resolution_notes,
        "escalated": ticket.escalated,
        "sla_breached": ticket.sla_breached,
        "ticket_created_date": (
            ticket.ticket_created_date.isoformat() if ticket.ticket_created_date else None
        ),
        "ticket_resolved_date": (
            ticket.ticket_resolved_date.isoformat() if ticket.ticket_resolved_date else None
        ),
        "customer_satisfaction_score": ticket.customer_satisfaction_score,
        "issue_complexity_score": ticket.issue_complexity_score,
        "customer_segment": ticket.customer_segment,
    }


def to_indexable_document(ticket: Ticket) -> IndexableDocument:
    """Convertit un Ticket validé en IndexableDocument prêt à indexer."""
    document_id = compute_document_id(
        category=ticket.category,
        product=ticket.product,
        language=ticket.language,
        issue=ticket.issue_description,
        resolution=ticket.resolution_notes,
    )
    return IndexableDocument(
        document_id=document_id,
        source_ticket_id=ticket.ticket_id,
        content=compose_content(ticket),
        payload=build_payload(ticket),
    )
