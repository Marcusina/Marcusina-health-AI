# Free-tier deploy (backend contract testing)

A trimmed stack so the backend team can integrate and test the AI API on a
small/free host, **before** paying for GPU. The full ML quality needs a GPU; this
profile gives you the real endpoints, auth, rate limiting, and the
enqueue → callback flow with everything that doesn't need a GPU working fully.

## Run it

```bash
cp .env.free.example .env.free
# edit .env.free → set API_SECRET_KEY (and the callback URL/secret)
docker compose -f docker-compose.free.yml --env-file .env.free up -d
# API is now at http://<host>:8001  (point the backend's AI_SERVICE_URL here)
curl http://localhost:8001/health
```

Services: `api` + one combined `worker` + `redis` + `rabbitmq` + `postgres`.
No nginx/flower/beat. ~2–4 GB RAM is comfortable.

## What works vs. what degrades

| Works fully (no GPU/LLM) | Degrades without an LLM |
|---|---|
| Triage (red-flag rules), distress + toxicity moderation, **quarantine policy** | SOAP notes, visit summaries |
| Drug-interaction check (#12) | Support draft replies |
| Semantic search + recommendations (#10/#11) | LLM-graded misinfo (returns "unverified" → human review) |
| Symptom intake (rules skeleton; #13) | The LLM "polish" of intake/triage rationale |
| PII redaction, audit, callbacks, rate limiting | |

"Degrades" = the endpoint still responds with `degraded: true` (or a fail-safe
verdict), never a crash — the service is built to be GPU-optional.

Whisper transcription runs on CPU with `WHISPER_MODEL_SIZE=base` (slow but
functional); leave audio out of the first test pass if you like.

## Free hosting options

- **Compute:** Oracle Cloud **Always-Free** ARM VM (up to 4 cores / 24 GB) runs
  this whole compose. (Render/Railway/Fly free tiers are usually too small / sleep.)
- **Don't want to self-host the backing services?** Point the env at managed free
  tiers instead and drop those containers: **Upstash** (Redis), **CloudAMQP**
  "Little Lemur" (RabbitMQ), **Neon/Supabase** (Postgres).
- **Even lighter (sync-only):** the sync endpoints (triage, moderate, search,
  recommend, medications, intake) need only `api` + `redis` — Postgres/RabbitMQ
  are only required for the async enqueue→callback lane.

## Going to production

No code changes — just config:
- Point `LLM_BASE_URL` at a GPU-served vLLM/Ollama and set `LLM_MODEL`.
- `WHISPER_MODEL_SIZE=large-v3`, `WHISPER_DEVICE=cuda`.
- Switch to the full `docker-compose.yml` (nginx load balancer, replicas,
  `SEARCH_BACKEND=redis`) — see its header for scaling.
