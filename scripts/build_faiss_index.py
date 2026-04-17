"""
scripts/build_faiss_index.py
==============================
Build the FAISS vector index from your health content database.
Run once after setup, then nightly via Celery beat scheduler.

Usage:
    python scripts/build_faiss_index.py
    python scripts/build_faiss_index.py --content-file ./data/content.json
"""

import os
import sys
import json
import argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from loguru import logger
import faiss
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings

settings = get_settings()


def build_index(content_items: list[dict]) -> None:
    """
    Embed all content items and build a FAISS IVF index.
    IVF (Inverted File) index: fast approximate nearest-neighbour search,
    much faster than flat search at scale (millions of vectors).

    Args:
        content_items: list of {id, text, type, ...} dicts from your DB
    """
    logger.info(f"Building FAISS index for {len(content_items)} items...")

    # ── Load embedder ─────────────────────────────────────────────────────────
    embedder = SentenceTransformer(
        settings.HF_EMBEDDING_MODEL,
        cache_folder=os.path.join(settings.MODELS_DIR, "embedders"),
    )

    # ── Embed content in batches ──────────────────────────────────────────────
    texts = [item["text"] for item in content_items]
    logger.info("Embedding content...")
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalise for cosine similarity via dot product
        convert_to_numpy=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    n = len(embeddings)

    # ── Build IVF index ───────────────────────────────────────────────────────
    # nlist = number of Voronoi cells. sqrt(n) is a good rule of thumb.
    nlist = max(1, int(np.sqrt(n)))
    quantizer = faiss.IndexFlatIP(dim)                    # Inner product (= cosine after normalise)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    logger.info(f"Training IVF index (nlist={nlist})...")
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = min(10, nlist)    # Search this many cells at query time (speed/accuracy tradeoff)

    # ── Save index + metadata ─────────────────────────────────────────────────
    Path(settings.FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, settings.FAISS_INDEX_PATH)

    with open(settings.FAISS_METADATA_PATH, "w") as f:
        json.dump(content_items, f)

    logger.info(f"FAISS index saved: {index.ntotal} vectors, dim={dim}")
    logger.info(f"  Index: {settings.FAISS_INDEX_PATH}")
    logger.info(f"  Metadata: {settings.FAISS_METADATA_PATH}")


def load_content_from_file(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_demo_content() -> list[dict]:
    """
    Demo content for testing. Replace with a real DB query in production:
        SELECT id, title || ' ' || body AS text, content_type AS type FROM health_articles;
    """
    return [
        {"id": "c001", "text": "Managing type 2 diabetes with diet and regular exercise programmes", "type": "article"},
        {"id": "c002", "text": "Understanding hypertension blood pressure management lifestyle", "type": "article"},
        {"id": "c003", "text": "Mental health tips managing anxiety stress depression Nigeria", "type": "post"},
        {"id": "c004", "text": "Cardiovascular health nutrition guide heart disease prevention", "type": "video"},
        {"id": "c005", "text": "Blood sugar monitoring home diabetes self management", "type": "guide"},
        {"id": "c006", "text": "Malaria prevention treatment Nigeria antimalarial medication", "type": "article"},
        {"id": "c007", "text": "Pregnancy nutrition prenatal care antenatal vitamins", "type": "guide"},
        {"id": "c008", "text": "Exercise routines weight loss obesity management fitness", "type": "video"},
        {"id": "c009", "text": "Kidney disease chronic renal failure prevention dialysis", "type": "article"},
        {"id": "c010", "text": "Childhood vaccination immunisation schedule Nigeria NPI", "type": "guide"},
        {"id": "c011", "text": "HIV AIDS treatment antiretroviral therapy ART adherence", "type": "article"},
        {"id": "c012", "text": "Sickle cell disease management pain crisis hydroxyurea", "type": "guide"},
        {"id": "c013", "text": "Stroke prevention risk factors high blood pressure smoking", "type": "article"},
        {"id": "c014", "text": "Tuberculosis TB treatment directly observed therapy DOTS", "type": "article"},
        {"id": "c015", "text": "Mental health awareness depression stigma Nigeria support", "type": "post"},
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-file", help="Path to JSON file with content items")
    args = parser.parse_args()

    if args.content_file:
        content = load_content_from_file(args.content_file)
    else:
        logger.info("No --content-file specified. Using demo content.")
        content = load_demo_content()

    build_index(content)
