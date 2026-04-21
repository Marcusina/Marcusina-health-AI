# Marcusina Health AI Service

High-throughput AI microservice bridging the **Fastify JS backend** with AI inference tasks.
Built for **50,000+ req/sec**, **5M+ users**, **24/7 uptime**.

---

## Architecture
Client
└─ Fastify (JS backend)
└─ POST /api/v1/* + X-AI-Secret   (returns task_id in <5ms)
└─ FastAPI (Gunicorn + uvicorn workers)
├─ Cache hit → return result immediately
└─ Cache miss → enqueue Celery task
├─ RabbitMQ queue (emergency / realtime / normal / batch)
│     └─ Celery worker (loads models once, processes tasks)
│           ├─ ONNX Runtime (NLP inference — NER, triage, misinfo, sentiment)
│           ├─ faster-whisper (speech transcription)
│           ├─ FAISS (semantic recommendations)
│           └─ Presidio (PII detection/redaction)
└─ Result → Redis → webhook to Fastify
---

## AI Models Used

| Task | HuggingFace ID | Size | Labels |
|---|---|---|---|
| Biomedical NER | `d4data/biomedical-ner-all` | 67 MB | 84 entity types |
| Triage classification | `Yuvrajxms09/biobert-triage-classifier` | 1.1 MB | urgent / non_urgent |
| Health misinformation | `jy46604790/Fake-News-Bert-Detect` | 1.2 MB | REAL / FAKE |
| Sentiment analysis | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 1.2 MB | negative / neutral / positive |
| Speech-to-text | `Systran/faster-whisper-base` | 145 MB | multilingual |
| Semantic embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 87 MB | 384-dim |

All models are exported to ONNX for 4-8× faster CPU inference vs raw PyTorch.

---

## Project Structure
Marcusina AI/
├── app/
│   ├── api/routes/        # FastAPI route handlers
│   ├── core/              # Settings, Celery, ONNX model registry
│   ├── models/            # Pydantic schemas
│   ├── tasks/             # Celery task definitions
│   ├── utils/             # Audit, cache, config_loader, exceptions
│   └── main.py            # FastAPI app entrypoint
├── config/                # Runtime-editable configuration (no code changes needed)
│   ├── red_flags.json         # Emergency symptom keywords
│   ├── health_claim_patterns.json  # Regex for unverified health claims
│   ├── distress_patterns.json      # Regex for mental-health distress
│   ├── toxic_keywords.json         # Toxicity keywords
│   ├── icd_map.json                # Symptom → ICD-10 mapping
│   └── specialty_map.json          # Symptom → medical specialty
├── models/
│   ├── onnx/{ner,triage,misinfo,sentiment}/   # Exported ONNX models
│   ├── whisper/                               # faster-whisper cache
│   ├── embedders/                             # Sentence transformer cache
│   └── faiss/content_index.bin                # FAISS index
├── scripts/
│   ├── export_onnx.py                 # Export HF models to ONNX
│   ├── build_faiss_index.py           # Build semantic recommendation index
│   ├── verify_labels.py               # Sanity-check id2label maps
│   ├── inspect_onnx.py                # Debug ONNX graph (for dev only)
│   ├── test_endpoints.py              # End-to-end test suite
│   └── fastify_integration.js         # Drop into your Fastify project
├── deploy/
│   └── nginx.conf
├── tests/
├── logs/
│   └── audit.log                      # NDJSON audit log (365-day retention)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example

---

## Setup — Production (Linux)

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env — set API_SECRET_KEY, FASTIFY_CALLBACK_URL, REDIS_URL, RABBITMQ_URL

# 2. Generate a strong secret
python -c "import secrets; print(secrets.token_hex(32))"   # paste into API_SECRET_KEY

# 3. Run setup (installs deps, downloads models, exports ONNX, builds FAISS)
chmod +x scripts/setup.sh && ./scripts/setup.sh

# 4. Start everything
docker compose up --build -d

# 5. Verify
docker compose ps
curl http://localhost:8001/health
```

---

## Setup — Development (Windows)

Windows requires specific workarounds: `--pool solo` for Celery, int32→int64 tokenizer casts, etc. All handled automatically by the code — just follow the steps below.

### Step 1: Install dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2: Configure environment

```bash
copy .env.example .env
# Edit .env — minimum: set API_SECRET_KEY to any string for local dev
```

### Step 3: Export AI models to ONNX

```bash
python scripts/export_onnx.py        # exports all 4 models (~5-10 minutes)
python scripts/build_faiss_index.py  # builds FAISS index
python scripts/verify_labels.py      # confirms id2label maps are correct
```

The `verify_labels.py` output should show all four models with correct labels:
### Step 4: Start infrastructure (Redis + RabbitMQ)

```bash
docker run -d --name health_redis -p 6379:6379 redis:7-alpine
docker run -d --name health_rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management-alpine
```

Verify they're running:
```bash
docker ps
docker exec health_redis redis-cli ping           # expect: PONG
docker exec health_rabbitmq rabbitmq-diagnostics ping   # expect: Ping succeeded
```

### Step 5: Run the service (three terminals)

**Terminal 1 — FastAPI server:**
```bash
uvicorn app.main:app --reload --port 8001
```

**Terminal 2 — Celery worker:**
```bash
celery -A app.core.celery_app worker --queues emergency,realtime,normal,batch --concurrency 2 --loglevel info --pool solo
```

Note: `--pool solo` is **required on Windows**. The default `prefork` pool uses `fork()` which doesn't exist on Windows.

**Terminal 3 — Run tests:**
```bash
python scripts/test_endpoints.py
```

All 10 tests should pass.

---

## Runtime Configuration (no-code changes)

All runtime rules live in `/config` as JSON. Edit any file and either restart the worker or call `config_loader.reload_all()` — no code changes or redeploys needed.

| File | Purpose | Shape |
|---|---|---|
| `red_flags.json` | Emergency symptom keywords | `{"_comment": "...", "keywords": [...]}` |
| `health_claim_patterns.json` | Regex patterns for misinfo | `{"_comment": "...", "patterns": [...]}` |
| `distress_patterns.json` | Regex patterns for mental-health distress | `{"_comment": "...", "patterns": [...]}` |
| `toxic_keywords.json` | Toxicity phrases | `{"_comment": "...", "keywords": [...]}` |
| `icd_map.json` | Symptom keyword → ICD-10 code | `{"_comment": "...", "mappings": {...}}` |
| `specialty_map.json` | Symptom keyword → medical specialty | `{"_comment": "...", "mappings": {...}}` |

Example: to add "seizure" as a red-flag, open `config/red_flags.json`, append to the `keywords` array, save, and restart the Celery worker.

---

## API Reference

### Authentication

All endpoints require the `X-AI-Secret` header matching `API_SECRET_KEY` in `.env`:

### Async flow

All AI endpoints return a `task_id` immediately:
```json
{ "task_id": "uuid-here", "status": "queued", "priority": "emergency" }
```

If the result is cached:
```json
{ "task_id": "cached", "status": "complete", "result": { ... } }
```

### Poll for result
Returns:
```json
{ "task_id": "...", "status": "complete|pending|failed", "result": { ... } }
```

### E-Consultation endpoints

| Endpoint | Queue | Typical latency |
|---|---|---|
| `POST /api/v1/consultation/transcribe` | realtime | 2–10s (audio length dependent) |
| `POST /api/v1/consultation/triage` | emergency or normal | 200–500ms |
| `POST /api/v1/consultation/soap-note` | normal | 3–8s |

Triage is routed to the `emergency` queue automatically when red-flag symptoms are detected.

### Health Social Media endpoints

| Endpoint | Queue | Typical latency |
|---|---|---|
| `POST /api/v1/social/moderate` | normal | 100–300ms |
| `POST /api/v1/social/recommend` | batch | 50–200ms |
| `POST /api/v1/social/sentiment` | batch | 50–150ms |

---

## Fastify Integration

Copy `scripts/fastify_integration.js` into your Fastify project. It provides:
- `AIClient` class with methods for every endpoint
- Automatic retry with exponential backoff
- Webhook handler for `/internal/ai-callback`
- Type-safe request validation

Register it in your Fastify server:
```javascript
const { registerAIRoutes } = require('./ai-integration');
registerAIRoutes(fastify, { aiBaseUrl: 'http://localhost:8001', aiSecret: process.env.AI_SECRET });
```

---

## Scaling Guide

| Users | Strategy |
|---|---|
| < 10k/day | 1 FastAPI server, 1 Celery worker per queue |
| 100k/day | 2 FastAPI replicas, 2–4 workers per queue |
| 1M/day | Kubernetes, HPA on worker pods, Redis Cluster |
| 50k req/sec | K8s + KEDA autoscaling on RabbitMQ queue depth |

For production, also consider:
- Upgrade `WHISPER_MODEL_SIZE` from `base` to `medium` or `large-v3` for better transcription
- Enable Redis persistence with AOF
- Use RabbitMQ cluster with quorum queues
- Run Celery with `prefork` pool on Linux (`--concurrency=<num_cores>`)

---

## Monitoring

- **Flower** (Celery task monitor): http://localhost:5555
- **RabbitMQ management UI**: http://localhost:15672 (guest/guest)
- **Prometheus metrics**: http://localhost:8001/metrics
- **Audit logs**: `logs/audit.log` (NDJSON format, 365-day retention per HIPAA guidance)

---

## Troubleshooting

### Celery worker won't start on Windows
Use `--pool solo`. The default `prefork` pool doesn't work on Windows because it needs `fork()`.

### `ShapeInferenceError` during INT8 quantization
Ignore — float32 ONNX is already 4–8× faster than raw PyTorch. INT8 is an optional optimization.

### `Unexpected input data type. Actual: tensor(int32), expected: tensor(int64)`
The model registry casts all int inputs to int64 automatically. If this error appears, make sure you're on the latest `app/core/model_registry.py` with the `.astype("int64")` line in `run_onnx_classifier`.

### Presidio takes 500+ seconds on first moderation
The first call downloads `en_core_web_lg` (587 MB). Fixed by `_get_presidio_engines()` caching — only happens once per worker process. Use `en_core_web_sm` in production to avoid the 587 MB download.

### `KeyError` on config loader
Your JSON file has the wrong top-level key. Every config file must be a dict with `_comment` and one of `keywords` / `patterns` / `mappings` depending on the file. Verify with:
```bash
python -c "import json,os; [print(f, '->', list(json.load(open('config/'+f)).keys())) for f in sorted(os.listdir('config')) if f.endswith('.json')]"
```

### `Connection refused` to localhost:6379 / localhost:5672
Redis / RabbitMQ is not running. Start with Docker:
```bash
docker run -d --name health_redis -p 6379:6379 redis:7-alpine
docker run -d --name health_rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management-alpine
```

### Callback warnings `[WinError 10061]` in worker logs
Expected until Fastify is running on port 3000. Results are still stored in Redis — Fastify can fetch them by polling `GET /api/v1/task/{task_id}`.

---

## License

Internal use — Marcusina AI Department.