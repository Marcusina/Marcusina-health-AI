"""
Tests for the trusted-source corpus ingestion pipeline (app/rag/ingest).

All filesystem writes are isolated to tmp dirs — these never touch the real
seed corpus or the ingested directory.
"""

from __future__ import annotations

import json

import app.rag.corpus as corpus
from app.rag.ingest import chunk_text, make_doc, ingest_files, add_documents


# ── chunking ──────────────────────────────────────────────────────────────────

def test_chunk_text_packs_paragraphs():
    text = "Para one is short.\n\nPara two is also short.\n\n" + ("x" * 50)
    chunks = chunk_text(text, max_chars=60)
    assert len(chunks) >= 2
    assert all(len(c) <= 60 for c in chunks)


def test_chunk_text_splits_long_paragraph():
    chunks = chunk_text("y" * 2000, max_chars=500, overlap=50)
    assert len(chunks) >= 4
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_text_drops_trivial_scraps():
    assert chunk_text("ok\n\nhi") == []      # both under the 40-char floor


# ── normalization ─────────────────────────────────────────────────────────────

def test_make_doc_stable_id_and_shape():
    a = make_doc("Vaccines are safe.", source="WHO", topic="vaccines", url="http://x")
    b = make_doc("Vaccines are safe.", source="WHO", topic="vaccines")
    assert a["id"] == b["id"]                # id is content-stable per source
    assert a["source"] == "WHO" and a["topic"] == "vaccines"
    assert set(a) == {"id", "topic", "source", "url", "text"}


# ── local file ingestion ──────────────────────────────────────────────────────

def test_ingest_files_txt_and_jsonl(tmp_path):
    (tmp_path / "doc.txt").write_text(
        "Handwashing reduces the spread of many infectious diseases and is "
        "one of the most effective public-health measures available.", encoding="utf-8")
    (tmp_path / "facts.jsonl").write_text(
        json.dumps({"id": "f1", "text": "Smoking causes lung cancer.",
                    "source": "CDC", "topic": "cancer"}) + "\n", encoding="utf-8")
    docs = ingest_files(tmp_path, source="local", topic="hygiene")
    texts = " ".join(d["text"] for d in docs)
    assert "Handwashing" in texts and "Smoking causes lung cancer" in texts
    # the .jsonl row keeps its own id/source
    assert any(d["id"] == "f1" and d["source"] == "CDC" for d in docs)


def test_ingest_files_relative_path(tmp_path, monkeypatch):
    # Path.as_uri() throws on relative paths — ingest must resolve() first.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rel.txt").write_text(
        "Tetanus is prevented by vaccination and is caused by a bacterial toxin "
        "that affects the nervous system; it is not contagious between people.",
        encoding="utf-8")
    docs = ingest_files("rel.txt", source="WHO", topic="infectious-disease")
    assert docs and docs[0]["url"].startswith("file:")


# ── dedup + persist ───────────────────────────────────────────────────────────

def test_add_documents_dedup(tmp_path, monkeypatch):
    monkeypatch.setattr("app.rag.ingest.INGESTED_DIR", tmp_path)
    # pretend the existing corpus already has "Old fact"
    monkeypatch.setattr("app.rag.ingest.load_corpus",
                        lambda: [{"text": "Old fact"}])
    docs = [make_doc("New fact A", source="X"),
            make_doc("New fact A", source="X"),     # dup within batch
            make_doc("Old fact", source="X"),       # dup vs existing corpus
            make_doc("New fact B", source="X")]
    res = add_documents(docs, refresh=False)
    assert res["added"] == 2
    assert res["skipped_duplicates"] == 2
    written = list(tmp_path.glob("*.jsonl"))
    assert len(written) == 1
    lines = written[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


# ── seed + ingested merge ─────────────────────────────────────────────────────

def test_load_corpus_merges_seed_and_ingested(tmp_path, monkeypatch):
    seed = tmp_path / "seed.jsonl"
    seed.write_text(json.dumps({"id": "s1", "text": "seed fact", "source": "WHO"}) + "\n",
                    encoding="utf-8")
    ing = tmp_path / "ingested"; ing.mkdir()
    (ing / "a.jsonl").write_text(
        json.dumps({"id": "i1", "text": "ingested fact", "source": "PubMed"}) + "\n",
        encoding="utf-8")
    monkeypatch.setattr(corpus, "CORPUS_PATH", seed)
    monkeypatch.setattr(corpus, "INGESTED_DIR", ing)
    corpus.load_corpus.cache_clear()
    try:
        texts = {d["text"] for d in corpus.load_corpus()}
        assert texts == {"seed fact", "ingested fact"}
    finally:
        corpus.load_corpus.cache_clear()   # don't leak the patched corpus to other tests
