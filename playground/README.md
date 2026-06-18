# Marcusina AI — Local Playground

A hands-on test harness for the AI capabilities. It calls the AI functions
**in-process**, so you do **not** need Postgres, Redis, RabbitMQ, or Celery
running — just the Python venv (and, for the LLM features, a local model server).

This is a localhost dev tool, **not** the production API (that's `app.main`, which
adds the async queue + auth).

## Run

```bash
# from the repo root, with the venv:
.venv/Scripts/python -m playground.server
```

Then open **http://localhost:8800**.

## What works without a model server

These run on rules / CPU and work immediately:

- **Triage** — red-flag rules → emergency (LLM only refines non-red-flag cases)
- **Moderation** — toxicity keywords + distress patterns
- **Semantic search** / **Recommendations** / **Add content** — embeddings (CPU)

With the LLM **off**, the LLM-powered tabs still respond, but with their
fail-safe / degraded result (which is worth seeing).

## What needs a local model server

**Misinfo (RAG)**, **SOAP note**, **Visit summary**, **Support assist** call the
LLM. Start one first (free, no API key):

```bash
ollama serve            # if not already running
ollama pull mistral     # one-time
```

The LLM status badge (top-right) shows **online / offline** and the configured
model. Point at a different server/model via `LLM_BASE_URL` / `LLM_MODEL` in `.env`.

## Testing with your own data

- **Add content** tab → paste a JSON array of your items → they're indexed, then
  use **Search** / **Recommendations** over them. (This mirrors how the backend
  would push real content — the AI service never reads your database.)
- Paste real (de-identified) transcripts into **SOAP** / **Summary**, real claims
  into **Misinfo**, real tickets into **Support**.
