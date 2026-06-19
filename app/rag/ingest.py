"""
Trusted-source corpus ingestion.

The RAG misinfo checker is only as good as its evidence base — the eval showed
the engine is sound but coverage gates recall (out-of-corpus claims come back
`unsupported`). This pipeline grows the corpus from real trusted sources.

Sources:
  * local files  — .jsonl (pre-formatted) or .txt/.md (chunked). No network.
  * PubMed       — abstracts via NCBI E-utilities (free, no key). Needs network.

Everything is normalized to {id, topic, source, url, text}, de-duplicated by
content hash, and written to app/rag/data/ingested/ (which load_corpus reads
alongside the curated seed).

CLI:
    python -m app.rag.ingest files <path-or-dir> --source WHO --topic vaccines
    python -m app.rag.ingest pubmed "covid-19 vaccine safety" --max 20 --topic covid-19
    python -m app.rag.ingest stats
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from app.rag.corpus import load_corpus, refresh_corpus, INGESTED_DIR

_TEXT_EXT = {".txt", ".md", ".markdown"}


# ── chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> list[str]:
    """
    Split a long document into retrieval-sized passages. Packs whole paragraphs
    up to max_chars; hard-splits any single paragraph that's too long. A small
    char overlap keeps context across boundaries.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if len(para) > max_chars:
            if buf:
                chunks.append(buf); buf = ""
            for i in range(0, len(para), max_chars - overlap):
                chunks.append(para[i:i + max_chars])
            continue
        if len(buf) + len(para) + 1 <= max_chars:
            buf = f"{buf}\n{para}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c.strip()) >= 40]   # drop trivially short scraps


# ── normalization ─────────────────────────────────────────────────────────────

def _hash(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()


def make_doc(text: str, *, source: str, url: str | None = None,
             topic: str = "general", doc_id: str | None = None) -> dict:
    text = text.strip()
    return {
        "id": doc_id or f"{source.lower().replace(' ', '-')}-{_hash(text)[:10]}",
        "topic": topic, "source": source, "url": url, "text": text,
    }


# ── local files ───────────────────────────────────────────────────────────────

def ingest_files(path: str | Path, *, source: str = "local",
                 topic: str = "general") -> list[dict]:
    """Ingest a file or a directory of files. .jsonl is taken as-is; .txt/.md is chunked."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.suffix.lower() in _TEXT_EXT | {".jsonl"})
    docs: list[dict] = []
    for f in files:
        if f.suffix.lower() == ".jsonl":
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("text"):
                    docs.append(make_doc(row["text"], source=row.get("source", source),
                                         url=row.get("url"), topic=row.get("topic", topic),
                                         doc_id=row.get("id")))
        else:
            file_url = f.resolve().as_uri()   # resolve() so relative paths work too
            for chunk in chunk_text(f.read_text(encoding="utf-8")):
                docs.append(make_doc(chunk, source=source, topic=topic, url=file_url))
    logger.info(f"[ingest] read {len(docs)} docs from {len(files)} file(s)")
    return docs


# ── PubMed (NCBI E-utilities) ─────────────────────────────────────────────────

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def ingest_pubmed(query: str, *, max_results: int = 20, topic: str | None = None) -> list[dict]:
    """Pull abstracts from PubMed for a query (free, no API key; needs network)."""
    import httpx
    from xml.etree import ElementTree as ET

    with httpx.Client(timeout=30.0) as client:
        ids = client.get(f"{_EUTILS}/esearch.fcgi", params={
            "db": "pubmed", "term": query, "retmax": max_results, "retmode": "json",
        }).json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            logger.warning(f"[ingest] PubMed returned no results for {query!r}")
            return []
        xml = client.get(f"{_EUTILS}/efetch.fcgi", params={
            "db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml",
        }).text

    docs: list[dict] = []
    root = ET.fromstring(xml)
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID") or ""
        title = (art.findtext(".//ArticleTitle") or "").strip()
        abstract = " ".join(t.text or "" for t in art.findall(".//AbstractText")).strip()
        if not abstract:
            continue
        body = f"{title} {abstract}".strip()
        docs.append(make_doc(body, source="PubMed", topic=topic or query,
                             url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                             doc_id=f"pubmed-{pmid}"))
    logger.info(f"[ingest] fetched {len(docs)} PubMed abstracts for {query!r}")
    return docs


# ── batch recipe (reproducible corpus growth) ─────────────────────────────────

def ingest_batch(recipe_path: str | Path, *, refresh: bool = True,
                 pause_seconds: float = 0.4) -> dict:
    """
    Run a checked-in recipe of ingestion jobs so the corpus can be regrown
    deterministically (re-runs are safe — add_documents dedups by content hash).

    Recipe = a .jsonl, one job per line:
        {"source": "pubmed", "query": "...", "topic": "vaccines", "max": 8}
        {"source": "files",  "path": "docs/who/", "name": "WHO", "topic": "covid-19"}

    PubMed jobs are spaced by `pause_seconds` to stay under NCBI's keyless
    rate limit (3 req/s). A failing job is logged and skipped, not fatal —
    one bad query shouldn't sink the whole batch.
    """
    recipe_path = Path(recipe_path)
    jobs = [json.loads(l) for l in
            recipe_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    all_docs: list[dict] = []
    for i, job in enumerate(jobs):
        src = job.get("source", "pubmed")
        topic = job.get("topic")
        try:
            if src == "pubmed":
                if i:
                    time.sleep(pause_seconds)
                all_docs += ingest_pubmed(job["query"], max_results=job.get("max", 20),
                                          topic=topic)
            elif src == "files":
                all_docs += ingest_files(job["path"], source=job.get("name", "local"),
                                         topic=topic or "general")
            else:
                logger.warning(f"[ingest] batch: unknown source {src!r}, skipping")
        except Exception as e:   # one bad job shouldn't sink the batch
            logger.warning(f"[ingest] batch job {job!r} failed: {e}")

    logger.info(f"[ingest] batch: {len(jobs)} job(s) -> {len(all_docs)} docs collected")
    return add_documents(all_docs, refresh=refresh)


# ── persist (with dedup) ──────────────────────────────────────────────────────

def add_documents(docs: list[dict], *, refresh: bool = True) -> dict:
    """
    Append new docs to the ingested corpus, skipping any whose text already
    exists (by content hash) in the seed or prior ingestions. Returns a summary.
    """
    existing = {_hash(d["text"]) for d in load_corpus()}
    fresh, seen = [], set()
    for d in docs:
        h = _hash(d["text"])
        if h in existing or h in seen:
            continue
        seen.add(h)
        fresh.append(d)

    if fresh:
        INGESTED_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out = INGESTED_DIR / f"ingest_{stamp}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for d in fresh:
                f.write(json.dumps(d) + "\n")
        logger.info(f"[ingest] wrote {len(fresh)} new docs -> {out.name}")
        if refresh:
            refresh_corpus()

    return {"received": len(docs), "added": len(fresh),
            "skipped_duplicates": len(docs) - len(fresh),
            "corpus_total": len(load_corpus())}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main(argv: list[str] | None = None) -> int:
    import argparse, sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Trusted-source corpus ingestion")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("files", help="ingest a file or directory")
    pf.add_argument("path")
    pf.add_argument("--source", default="local")
    pf.add_argument("--topic", default="general")

    pp = sub.add_parser("pubmed", help="ingest PubMed abstracts for a query")
    pp.add_argument("query")
    pp.add_argument("--max", type=int, default=20)
    pp.add_argument("--topic", default=None)

    pb = sub.add_parser("batch", help="run a .jsonl recipe of ingestion jobs")
    pb.add_argument("recipe", nargs="?",
                    default=str(Path(__file__).parent / "data" / "ingest_queries.jsonl"))

    sub.add_parser("stats", help="show current corpus size")
    args = ap.parse_args(argv)

    if args.cmd == "stats":
        corpus = load_corpus()
        by_source: dict[str, int] = {}
        for d in corpus:
            by_source[d.get("source", "?")] = by_source.get(d.get("source", "?"), 0) + 1
        print(f"corpus total: {len(corpus)}")
        for s, n in sorted(by_source.items(), key=lambda x: -x[1]):
            print(f"  {s}: {n}")
        return 0

    if args.cmd == "batch":
        print(ingest_batch(args.recipe))
        return 0

    if args.cmd == "files":
        docs = ingest_files(args.path, source=args.source, topic=args.topic)
    else:  # pubmed
        docs = ingest_pubmed(args.query, max_results=args.max, topic=args.topic)

    print(add_documents(docs))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
