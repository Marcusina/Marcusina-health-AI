"""
Marcusina AI — Local Playground server.

A lightweight test harness that calls the AI capabilities **in-process**, so you
can try them with your own data WITHOUT standing up Postgres / Redis / RabbitMQ /
Celery. It bypasses the async queue and the auth layer — it is a localhost dev
tool, NOT the production API (that's app.main).

Run:
    .venv/Scripts/python -m playground.server
    # then open http://localhost:8800

Capabilities that use the LLM (misinfo, SOAP, summary, support) need a local
model server (e.g. `ollama serve` + `ollama pull mistral`). Without one they
return their fail-safe / degraded result — which is itself worth seeing.

Handlers are sync `def` so FastAPI runs them in a threadpool; a slow LLM call
won't freeze the page.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

HERE = Path(__file__).parent
app = FastAPI(title="Marcusina AI Playground", docs_url="/docs")


def _safe(fn, *args, **kwargs):
    """Run a capability; never 500 — surface errors to the UI as JSON."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — playground must always answer
        logger.exception("playground call failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/play/llm-health")
def llm_health():
    from app.llm import get_llm
    return _safe(get_llm().health)


# ── Tier-1 sync ───────────────────────────────────────────────────────────────

class TriageIn(BaseModel):
    symptoms: str
    age: Optional[int] = None
    medical_history: Optional[list[str]] = None
    use_llm: bool = True


@app.post("/play/triage")
def triage(body: TriageIn):
    from app.safety import assess_triage
    return _safe(assess_triage, body.symptoms, body.age, body.medical_history, body.use_llm)


class ModerateIn(BaseModel):
    text: str
    context: str = "post"
    deep_scan: bool = False


@app.post("/play/moderate")
def moderate(body: ModerateIn):
    from app.safety import assess_moderation
    return _safe(assess_moderation, body.text, body.context, body.deep_scan)


# ── RAG misinfo ───────────────────────────────────────────────────────────────

class MisinfoIn(BaseModel):
    text: str
    k: int = 4


@app.post("/play/misinfo")
def misinfo(body: MisinfoIn):
    from app.rag import check_claim
    return _safe(check_claim, body.text, body.k)


# ── Clinical generation ───────────────────────────────────────────────────────

class SoapIn(BaseModel):
    transcript: str
    specialty: Optional[str] = None


@app.post("/play/soap")
def soap(body: SoapIn):
    from app.clinical import generate_soap
    return _safe(generate_soap, body.transcript, specialty=body.specialty)


class SummaryIn(BaseModel):
    transcript: str


@app.post("/play/summary")
def summary(body: SummaryIn):
    from app.clinical import generate_summary
    return _safe(generate_summary, body.transcript)


# ── Support assist ────────────────────────────────────────────────────────────

class SupportIn(BaseModel):
    subject: str = ""
    message: str
    category_hint: Optional[str] = None


@app.post("/play/support")
def support(body: SupportIn):
    from app.support import draft_support_reply
    return _safe(draft_support_reply, body.subject, body.message, body.category_hint)


# ── Search & recommend ────────────────────────────────────────────────────────

class SearchIn(BaseModel):
    query: str
    k: int = 10
    content_type: Optional[str] = None


@app.post("/play/search")
def search(body: SearchIn):
    from app.search import semantic_search
    return _safe(semantic_search, body.query, body.k, body.content_type)


class RecommendIn(BaseModel):
    user_interests: list[str] = []
    user_conditions: list[str] = []
    context: str = ""
    k: int = 10
    exclude: list[str] = []


@app.post("/play/recommend")
def recommend(body: RecommendIn):
    from app.search import recommend as _recommend
    return _safe(_recommend, body.user_interests, body.user_conditions,
                 body.context, body.k, body.exclude)


class ContentItemIn(BaseModel):
    id: str
    text: str
    type: str = "content"
    metadata: dict = {}


class IndexIn(BaseModel):
    items: list[ContentItemIn]


@app.post("/play/index")
def index_content(body: IndexIn):
    from app.search import index_content as _index
    return _safe(_index, [i.model_dump() for i in body.items])


if __name__ == "__main__":
    import uvicorn
    logger.info("Playground at http://localhost:8800  (LLM features need a local model server)")
    uvicorn.run(app, host="127.0.0.1", port=8800, log_level="info")
