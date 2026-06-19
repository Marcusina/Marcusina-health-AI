"""
Content vector store — the engine behind semantic search and recommendations.

Holds embedded content items the backend pushes in (it never reaches into the
backend's MongoDB). Embeddings come from the shared CPU sentence-transformer.
Brute-force cosine is plenty at this scale; swap in FAISS/pgvector here if the
catalog grows to many thousands of items.

Persistence & multi-worker correctness: disk is the source of truth. Each worker
keeps an in-memory copy for fast cosine, but:
  * writes go through a cross-process file lock and re-read the latest on-disk
    state before applying their delta, then save atomically (temp file + os.replace).
    This stops one worker's save from clobbering content another worker added.
  * reads call reload_if_changed() first — a cheap mtime check that reloads only
    when another worker has written since. This gives cross-worker freshness on a
    single host with no extra infra. (Multi-HOST scaling still wants a shared
    store like Redis/pgvector — swap the persistence layer here.)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from filelock import FileLock
from loguru import logger

from app.core.config import get_settings
from app.embeddings import Embedder, embed, l2_normalize

settings = get_settings()

INDEX_DIR = Path(settings.MODELS_DIR) / "search"
SEED_PATH = Path(__file__).parent / "data" / "seed_content.jsonl"

# Serializes writers across processes so concurrent upserts don't lose data.
_LOCK_PATH = INDEX_DIR / "index.lock"


def index_write_lock(timeout: float = 15.0) -> FileLock:
    """Cross-process lock held around a read-latest → mutate → save cycle."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return FileLock(str(_LOCK_PATH), timeout=timeout)


class VectorStore:
    def __init__(self, embedder: Embedder | None = None):
        self._embedder = embedder
        self._records: dict[str, dict] = {}        # id -> {id, type, text, metadata}
        self._vectors: dict[str, np.ndarray] = {}  # id -> normalized vector
        self._ids: list[str] = []                  # cached order, aligned to _matrix
        self._matrix: np.ndarray | None = None     # cache, invalidated on change
        self._loaded_mtime: float | None = None    # mtime of the on-disk index we last loaded

    # ── mutations ───────────────────────────────────────────────────────────────

    def upsert(self, items: list[dict]) -> int:
        """Add or replace content items. Each: {id, text, type?, metadata?}."""
        texts, ids = [], []
        for it in items:
            if not it.get("id") or not it.get("text"):
                continue
            ids.append(it["id"])
            texts.append(it["text"])
        if not ids:
            return 0
        vecs = l2_normalize(embed(texts, self._embedder))
        for it, _id, vec in zip(items, ids, vecs):
            self._records[_id] = {"id": _id, "type": it.get("type", "content"),
                                  "text": it["text"], "metadata": it.get("metadata", {})}
            self._vectors[_id] = vec
        self._matrix = None
        return len(ids)

    def remove(self, ids: list[str]) -> int:
        n = 0
        for _id in ids:
            if self._records.pop(_id, None) is not None:
                self._vectors.pop(_id, None)
                n += 1
        self._matrix = None
        return n

    def __len__(self) -> int:
        return len(self._records)

    # ── query ─────────────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        self._ids = list(self._records.keys())
        if self._ids:
            self._matrix = np.vstack([self._vectors[i] for i in self._ids])
        else:
            self._matrix = np.empty((0, 0), dtype="float32")

    def search(self, query: str, k: int = 10, type_filter: str | None = None) -> list[dict]:
        if not self._records:
            return []
        if self._matrix is None:
            self._rebuild()
        q = l2_normalize(embed([query], self._embedder))[0]
        scores = self._matrix @ q
        order = np.argsort(-scores)
        out: list[dict] = []
        for idx in order:
            rec = self._records[self._ids[idx]]
            if type_filter and rec["type"] != type_filter:
                continue
            out.append({**rec, "score": round(float(scores[idx]), 4)})
            if len(out) >= k:
                break
        return out

    # ── persistence ─────────────────────────────────────────────────────────────

    def save(self, directory: Path | None = None) -> None:
        d = directory or INDEX_DIR
        d.mkdir(parents=True, exist_ok=True)
        ids = list(self._records.keys())
        mat = np.vstack([self._vectors[i] for i in ids]) if ids else np.empty((0, 0), "float32")

        # Write to temp files then atomically replace, so a concurrent reader never
        # sees a half-written index. Replace vectors FIRST and records LAST —
        # records.jsonl's mtime is the reload trigger, so by the time a reader sees
        # the new records, the matching vectors are already in place.
        vtmp, rtmp = d / "vectors.npy.tmp", d / "records.jsonl.tmp"
        with open(vtmp, "wb") as vf:           # file handle → np.save won't munge the name
            np.save(vf, mat)
        with open(rtmp, "w", encoding="utf-8") as f:
            for i in ids:
                f.write(json.dumps(self._records[i]) + "\n")
        os.replace(vtmp, d / "vectors.npy")
        os.replace(rtmp, d / "records.jsonl")
        self._loaded_mtime = (d / "records.jsonl").stat().st_mtime   # don't reload our own write
        logger.info(f"[search] saved {len(ids)} items to {d}")

    def load(self, directory: Path | None = None) -> bool:
        d = directory or INDEX_DIR
        vpath, rpath = d / "vectors.npy", d / "records.jsonl"
        if not (vpath.exists() and rpath.exists()):
            return False
        mat = np.load(vpath)
        with open(rpath, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
        self._records, self._vectors = {}, {}
        for rec, vec in zip(recs, mat):
            self._records[rec["id"]] = rec
            self._vectors[rec["id"]] = vec
        self._matrix = None
        self._loaded_mtime = rpath.stat().st_mtime
        logger.info(f"[search] loaded {len(recs)} items from {d}")
        return True

    def reload_if_changed(self, directory: Path | None = None) -> bool:
        """Reload from disk only if another process has written since we last loaded."""
        d = directory or INDEX_DIR
        rpath = d / "records.jsonl"
        try:
            mtime = rpath.stat().st_mtime
        except FileNotFoundError:
            return False
        if self._loaded_mtime is None or mtime > self._loaded_mtime:
            return self.load(d)
        return False

    def load_seed(self) -> int:
        with open(SEED_PATH, encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        n = self.upsert(items)
        logger.info(f"[search] seeded {n} demo content items")
        return n


_store: VectorStore | None = None


def get_store() -> VectorStore:
    """
    Process-wide store. Loads the persisted index if present; otherwise seeds demo
    content so search/recommend work out of the box (the backend then ingests real
    content via the index endpoints).
    """
    global _store
    if _store is None:
        _store = VectorStore()
        if not _store.load():
            _store.load_seed()
    return _store
