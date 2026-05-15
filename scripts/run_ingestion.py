"""Script d'ingestion : lit le CSV, normalise, compose les documents,
dump en JSONL dans data/processed/.

L'indexation Qdrant sera branchée à l'Incrément 4. Pour l'instant, ce script
sert à valider le pipeline d'ingestion bout-en-bout sur l'échantillon
configuré (`DATASET_SAMPLE_SIZE`).

Usage :
    python scripts/run_ingestion.py [--sample-size N] [--output PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Permet d'exécuter le script directement (sans pip install -e .) en l'ajoutant au path.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragtime.config import get_settings  # noqa: E402
from ragtime.ingest.pipeline import run_pipeline  # noqa: E402
from ragtime.logging_setup import configure_logging, get_logger  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingestion des tickets de support")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limiter le nombre de tickets lus (override Settings).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Chemin du JSONL de sortie (default: data/processed/indexable.jsonl).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    configure_logging()
    logger = get_logger("scripts.run_ingestion")

    settings = get_settings()
    sample_size = args.sample_size if args.sample_size is not None else settings.dataset_sample_size
    output_path = args.output or (settings.project_root / "data/processed/indexable.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "ingestion.start",
        csv_path=str(settings.dataset_path),
        sample_size=sample_size,
        output=str(output_path),
    )

    started = time.perf_counter()
    n = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for doc in run_pipeline(settings.dataset_path, sample_size=sample_size):
            fh.write(doc.model_dump_json() + "\n")
            n += 1

    took = time.perf_counter() - started
    logger.info("ingestion.complete", documents=n, took_sec=round(took, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
