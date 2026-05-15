"""Script d'indexation : lit le Parquet d'embeddings et upsert dans Qdrant.

Le script est **idempotent** : ré-exécuté sur le même fichier, il écrase
les points existants (UUID5 dérivé du document_id) sans créer de doublon.

Usage :
    python scripts/run_indexation.py [--input PATH] [--recreate-collection]
    python scripts/run_indexation.py --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragtime.config import get_settings  # noqa: E402
from ragtime.embeddings.sparse import SparseVector  # noqa: E402
from ragtime.index import (  # noqa: E402
    IndexableRow,
    build_qdrant_client,
    create_collection,
    upsert_documents,
)
from ragtime.logging_setup import configure_logging, get_logger  # noqa: E402

DENSE_DIM_E5_LARGE = 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indexation Qdrant des embeddings")
    parser.add_argument("--input", type=Path, default=None, help="Chemin du Parquet")
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Supprime puis recrée la collection (efface tout)",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Taille des batches upsert")
    return parser.parse_args()


def _row_to_indexable(row: dict[str, Any]) -> IndexableRow:
    sparse = SparseVector(
        indices=list(row["sparse_indices"]),
        values=list(row["sparse_values"]),
    )
    payload = (
        json.loads(row["payload"]) if isinstance(row["payload"], str) else dict(row["payload"])
    )
    return IndexableRow(
        document_id=row["document_id"],
        source_ticket_id=int(row["source_ticket_id"]),
        content=row["content"],
        dense=list(row["dense"]),
        sparse=sparse,
        payload=payload,
    )


def main() -> int:
    args = _parse_args()
    configure_logging()
    logger = get_logger("scripts.run_indexation")

    settings = get_settings()
    input_path = args.input or (settings.project_root / "data/processed/embeddings.parquet")
    if not input_path.exists():
        logger.error("indexation.missing_input", path=str(input_path))
        return 1

    logger.info("indexation.start", input=str(input_path), batch_size=args.batch_size)
    df = pd.read_parquet(input_path)
    logger.info("indexation.loaded", rows=len(df))

    client = build_qdrant_client()
    name = create_collection(
        client,
        dense_dim=DENSE_DIM_E5_LARGE,
        recreate=args.recreate_collection,
    )

    t0 = time.perf_counter()
    rows = (_row_to_indexable(r) for r in df.to_dict(orient="records"))
    total = upsert_documents(client, rows, collection=name, batch_size=args.batch_size)

    logger.info(
        "indexation.complete",
        collection=name,
        total=total,
        took_sec=round(time.perf_counter() - t0, 2),
    )

    # Affiche les stats finales de la collection.
    info = client.get_collection(collection_name=name)
    logger.info(
        "indexation.collection_info", points_count=info.points_count, status=str(info.status)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
