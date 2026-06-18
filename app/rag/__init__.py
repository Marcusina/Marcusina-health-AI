"""
RAG misinformation lane.

The prior misinfo approach was a generic fake-news classifier; on a health-claim
benchmark it scored 0.557 precision (barely above chance) — it flagged most true
health information as misinfo. A classifier can't know *facts*; it only learns
surface patterns. So this lane does it the right way:

    claim ──embed──> retrieve trusted evidence (WHO/CDC/PubMed snapshot)
                      └─> LLM judges the claim *against that evidence* only,
                          returning a verdict + the citations it used.

The model is not asked "is this true?" from memory — it is asked "does this
evidence support or contradict this claim?", which is checkable and grounded.
Per the safety posture, the output is an **advisory flag for human review**,
never an automatic removal. If the LLM is unavailable, the verdict is
"unverified" + needs_human_review, never a silent pass/fail.

Public surface:
    check_claim(text, k=...) -> dict matching MisinfoResult
    get_retriever()          -> process-wide Retriever
"""

from app.rag.misinfo import check_claim
from app.rag.retriever import get_retriever
from app.rag.corpus import load_corpus, refresh_corpus
from app.rag.ingest import ingest_files, ingest_pubmed, add_documents

__all__ = [
    "check_claim", "get_retriever", "load_corpus", "refresh_corpus",
    "ingest_files", "ingest_pubmed", "add_documents",
]
