"""
Content vector store — the engine behind semantic search and recommendations.

Holds embedded content items the backend pushes in (it never reaches into the
backend's MongoDB). Embeddings come from the shared CPU sentence-transformer.
Brute-force cosine is plenty at this scale; swap in FAISS/pgvector here if the
catalog grows to many thousands of items.

Persistence is pluggable via an IndexBackend, chosen by `settings.SEARCH_BACKEND`:

  * "disk"  — local files (vectors.npy + records.jsonl), a cross-process file lock,
    and an mtime-based reload. Correct for multiple workers on ONE host.
  * "redis" — the shared REDIS_URL holds the index (one hash + a version counter)
    and a Redis lock serializes writers. Correct across MULTIPLE hosts, so you can
    run duplicate servers behind a load balancer and they stay consistent.

Either way the pattern is the same: each worker keeps an in-memory copy for fast
cosine; writes take a lock, re-read the latest, apply their delta, and publish
atomically; reads reload only when the shared version/mtime has moved.
"""

from __future__ import annotations

import contextlib
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

# Disk backend: cross-process file lock so concurrent upserts don't lose data.
_LOCK_PATH = INDEX_DIR / "index.lock"

# Redis backend keys (shared across all hosts).
_REDIS_ITEMS = "health_ai:search:items"        # hash: id -> {"r": record, "v": vector}
_REDIS_VERSION = "health_ai:search:version"     # int, bumped on every write
_REDIS_LOCK = "health_ai:search:lock"


class VectorStore:
    """In-memory cosine engine. Persistence is delegated to an IndexBackend."""

    def __init__(self, embedder: Embedder | None = None):
        self._embedder = embedder
        self._records: dict[str, dict] = {}        # id -> {id, type, text, metadata}
        self._vectors: dict[str, np.ndarray] = {}  # id -> normalized vector
        self._ids: list[str] = []                  # cached order, aligned to _matrix
        self._matrix: np.ndarray | None = None     # cache, invalidated on change
        self._loaded_sig: tuple | None = None      # disk backend: (mtime_ns, size) we last loaded
        self._loaded_version: str | None = None    # redis backend: version we last loaded

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

    def snapshot(self) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
        """The current records + vectors (for a backend to persist)."""
        return self._records, self._vectors

    def replace(self, records: dict[str, dict], vectors: dict[str, np.ndarray]) -> None:
        """Swap in a freshly-loaded dataset (used by a backend on reload)."""
        self._records = records
        self._vectors = vectors
        self._matrix = None

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

    # ── disk persistence (used by DiskBackend and tests) ─────────────────────────

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
        self._loaded_sig = self._disk_sig(d / "records.jsonl")   # don't reload our own write
        logger.info(f"[search] saved {len(ids)} items to {d}")

    @staticmethod
    def _disk_sig(path: Path) -> tuple | None:
        # (mtime_ns, size): catches rapid back-to-back writes that share an mtime
        # tick but change content — mtime alone is too coarse on some filesystems.
        try:
            st = path.stat()
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size)

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
        self._loaded_sig = self._disk_sig(rpath)
        logger.info(f"[search] loaded {len(recs)} items from {d}")
        return True

    def reload_if_changed(self, directory: Path | None = None) -> bool:
        """Disk: reload only if the file changed since we last loaded."""
        d = directory or INDEX_DIR
        sig = self._disk_sig(d / "records.jsonl")
        if sig is None:
            return False
        if sig != self._loaded_sig:
            return self.load(d)
        return False

    def load_seed(self) -> int:
        with open(SEED_PATH, encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        n = self.upsert(items)
        logger.info(f"[search] seeded {n} demo content items")
        return n


# ── persistence backends ──────────────────────────────────────────────────────

class DiskBackend:
    """Local-disk persistence + file lock. Multi-worker safe on a single host."""
    name = "disk"

    def lock(self, timeout: float = 15.0):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        return FileLock(str(_LOCK_PATH), timeout=timeout)

    def initial_load(self, store: VectorStore) -> bool:
        return store.load()

    def reload_if_changed(self, store: VectorStore) -> bool:
        return store.reload_if_changed()

    def persist(self, store: VectorStore) -> None:
        store.save()


class RedisBackend:
    """Shared-Redis persistence + Redis lock. Consistent across multiple hosts."""
    name = "redis"

    @staticmethod
    def _redis():
        from app.utils.cache import _get_sync_redis
        return _get_sync_redis()

    def lock(self, timeout: float = 15.0):
        r = self._redis()
        if r is None:
            logger.warning("[search] Redis unavailable — index write is uncoordinated")
            return contextlib.nullcontext()
        # auto-expiring lock so a crashed holder can't deadlock the cluster
        return r.lock(_REDIS_LOCK, timeout=30.0, blocking_timeout=timeout)

    def initial_load(self, store: VectorStore) -> bool:
        return self.reload_if_changed(store)

    def reload_if_changed(self, store: VectorStore) -> bool:
        r = self._redis()
        if r is None:
            return False
        version = r.get(_REDIS_VERSION)
        if version is None or version == store._loaded_version:
            return False
        raw = r.hgetall(_REDIS_ITEMS)
        records: dict[str, dict] = {}
        vectors: dict[str, np.ndarray] = {}
        for _id, payload in raw.items():
            obj = json.loads(payload)
            records[_id] = obj["r"]
            vectors[_id] = np.asarray(obj["v"], dtype="float32")
        store.replace(records, vectors)
        store._loaded_version = version
        logger.info(f"[search] loaded {len(records)} items from Redis (v{version})")
        return True

    def persist(self, store: VectorStore) -> None:
        r = self._redis()
        if r is None:
            logger.warning("[search] Redis unavailable — index change not shared")
            return
        records, vectors = store.snapshot()
        tmp = f"{_REDIS_ITEMS}:tmp"
        r.delete(tmp)
        if records:
            mapping = {_id: json.dumps({"r": records[_id], "v": vectors[_id].tolist()})
                       for _id in records}
            r.hset(tmp, mapping=mapping)
            r.rename(tmp, _REDIS_ITEMS)        # atomic swap — readers never see a partial hash
        else:
            r.delete(_REDIS_ITEMS)
        version = r.incr(_REDIS_VERSION)
        store._loaded_version = str(version)
        logger.info(f"[search] persisted {len(records)} items to Redis (v{version})")


def get_backend():
    """Pick the persistence backend from config (disk by default)."""
    return RedisBackend() if settings.SEARCH_BACKEND == "redis" else DiskBackend()


def index_write_lock(timeout: float = 15.0):
    """Lock for a read-latest → mutate → persist cycle (disk or Redis backed)."""
    return get_backend().lock(timeout)


_store: VectorStore | None = None


def get_store() -> VectorStore:
    """
    Process-wide store. Loads the shared index if present; otherwise seeds demo
    content (and publishes it so peers share the same seed) so search/recommend
    work out of the box. The backend then receives real content via the endpoints.
    """
    global _store
    if _store is None:
        store = VectorStore()
        backend = get_backend()
        backend.initial_load(store)
        if len(store) == 0:
            with index_write_lock():
                backend.reload_if_changed(store)      # a peer may have seeded under the lock
                if len(store) == 0:
                    store.load_seed()
                    backend.persist(store)
        _store = store
    return _store
