"""Loads the trusted-source corpus the retriever searches over."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from loguru import logger

DATA_DIR = Path(__file__).parent / "data"
CORPUS_PATH = DATA_DIR / "trusted_health_corpus.jsonl"     # curated seed (committed)
INGESTED_DIR = DATA_DIR / "ingested"                        # grown by the pipeline (gitignored)


def _read_jsonl(path: Path) -> list[dict]:
    docs: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("text"):
                docs.append(row)
    return docs


@lru_cache(maxsize=1)
def load_corpus() -> list[dict]:
    """
    Load the trusted-source documents: the curated seed plus everything the
    ingestion pipeline has added under data/ingested/. Each row:
        {id, topic, source, url, text}

    The seed is hand-curated WHO/CDC/NHS/NIH guidance; ingested docs come from
    app/rag/ingest.py (local files, PubMed, …). Call refresh_corpus() after
    ingesting to pick up new docs in a running process.
    """
    docs = _read_jsonl(CORPUS_PATH) if CORPUS_PATH.exists() else []
    n_seed = len(docs)
    if INGESTED_DIR.exists():
        for p in sorted(INGESTED_DIR.glob("*.jsonl")):
            docs.extend(_read_jsonl(p))
    logger.info(f"[rag] corpus: {n_seed} seed + {len(docs) - n_seed} ingested "
                f"= {len(docs)} trusted-source documents")
    return docs


def refresh_corpus() -> None:
    """Drop the cached corpus + retriever so newly-ingested docs are picked up."""
    load_corpus.cache_clear()
    try:
        import app.rag.retriever as r
        r._retriever = None
    except Exception:
        pass
