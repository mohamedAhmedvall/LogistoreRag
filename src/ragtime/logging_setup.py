"""Configuration du logging structuré via structlog.

Émet des logs JSON sur stdout pour faciliter l'agrégation (ELK, Loki, …).
À appeler **une seule fois** au démarrage de l'application (API, scripts).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from ragtime.config import get_settings


def _drop_color_message_key(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Supprime la clé `color_message` (uvicorn) pour ne pas polluer le JSON."""
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging(log_level: str | None = None) -> None:
    """Configure structlog + stdlib logging pour émettre du JSON structuré.

    Args:
        log_level: Niveau de log (DEBUG/INFO/WARNING/ERROR/CRITICAL).
            Si None, utilise la valeur de `Settings.log_level`.
    """
    level_name = (log_level or get_settings().log_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        _drop_color_message_key,
        timestamper,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).handlers.clear()
        logging.getLogger(noisy).propagate = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Retourne un logger structlog (à utiliser à la place de logging.getLogger)."""
    return structlog.stdlib.get_logger(name)
