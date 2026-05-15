"""Script d'embedding : lit le JSONL produit par `run_ingestion.py`,
calcule les vecteurs dense (E5-large) et sparse (BM25), et persiste le
résultat dans un fichier Parquet à `data/processed/embeddings.parquet`.

Conserver les embeddings sur disque évite de recalculer à chaque essai
sur Qdrant (Incrément 4+). Le Parquet contient :

- document_id (str)
- source_ticket_id (int)
- content (str)
- dense (list[float], 1024 dims pour E5-large)
- sparse_indices (list[int])
- sparse_values (list[float])
- payload (string JSON)

Usage :
    python scripts/run_embedding.py [--input PATH] [--output PATH] [--limit N]
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
from ragtime.embeddings import (  # noqa: E402
    get_dense_embedder,
    get_sparse_embedder,
)
from ragtime.logging_setup import configure_logging, get_logger  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding dense + sparse des tickets indexables")
    parser.add_argument("--input", type=Path, default=None, help="Chemin du JSONL d'entrée")
    parser.add_argument("--output", type=Path, default=None, help="Chemin du Parquet de sortie")
    parser.add_argument("--limit", type=int, default=None, help="Limite le nombre de docs traités")
    return parser.parse_args()


def _load_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            docs.append(json.loads(line))
    return docs


def main() -> int:
    args = _parse_args()
    configure_logging()
    logger = get_logger("scripts.run_embedding")

    settings = get_settings()
    input_path = args.input or (settings.project_root / "data/processed/indexable.jsonl")
    output_path = args.output or (settings.project_root / "data/processed/embeddings.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("embedding.start", input=str(input_path), output=str(output_path), limit=args.limit)

    docs = _load_jsonl(input_path, args.limit)
    logger.info("embedding.loaded", count=len(docs))
    if not docs:
        logger.warning("embedding.empty_input")
        return 0

    contents = [d["content"] for d in docs]

    # Dense
    dense_embedder = get_dense_embedder()
    t0 = time.perf_counter()
    dense_vecs = list(dense_embedder.embed_documents(contents))
    logger.info(
        "embedding.dense_done",
        count=len(dense_vecs),
        dims=len(dense_vecs[0]) if dense_vecs else 0,
        took_sec=round(time.perf_counter() - t0, 2),
    )

    # Sparse
    sparse_embedder = get_sparse_embedder()
    t0 = time.perf_counter()
    sparse_vecs = list(sparse_embedder.embed(contents))
    avg_nnz = sum(s.nnz for s in sparse_vecs) / max(len(sparse_vecs), 1)
    logger.info(
        "embedding.sparse_done",
        count=len(sparse_vecs),
        avg_nnz=round(avg_nnz, 1),
        took_sec=round(time.perf_counter() - t0, 2),
    )

    # Construire le DataFrame
    df = pd.DataFrame(
        {
            "document_id": [d["document_id"] for d in docs],
            "source_ticket_id": [d["source_ticket_id"] for d in docs],
            "content": contents,
            "dense": dense_vecs,
            "sparse_indices": [s.indices for s in sparse_vecs],
            "sparse_values": [s.values for s in sparse_vecs],
            "payload": [json.dumps(d["payload"], ensure_ascii=False) for d in docs],
        }
    )

    df.to_parquet(output_path, compression="zstd", index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "embedding.complete",
        rows=len(df),
        output=str(output_path),
        size_mb=round(size_mb, 2),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
