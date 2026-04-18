# Health AI Service 

High-throughput AI microservice bridging your **Fastify JS backend** with AI inference.
Built for **50,000+ req/sec**, **5M+ users**, **24/7 uptime**.



## Architecture

```
Client
  └─ Fastify (JS backend)
        └─ POST /api/v1/* + X-AI-Secret   (returns task_id in <5ms)
              └─ FastAPI (Gunicorn + uvicorn workers)
                    ├─ Cache hit → return result immediately
                    └─ Cache miss → enqueue Celery task
                          ├─ RabbitMQ queue (emergency / realtime / normal / batch)
                          │     └─ Celery worker (loads models once, processes tasks)
                          │           ├─ ONNX Runtime (NLP inference)
                          │           ├─ faster-whisper (speech)
                          │           └─ FAISS (recommendations)
                          └─ Result → Redis → webhook to Fastify
```

---

## Setup (Linux server)

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env — set API_SECRET_KEY, FASTIFY_CALLBACK_URL, etc.

# 2. Run setup (installs deps, downloads models, exports ONNX, builds FAISS)
chmod +x scripts/setup.sh && ./scripts/setup.sh

# 3. Start everything
docker compose up --build -d

# 4. Check status
docker compose ps
curl http://localhost:8001/health
```

---

## Setup (Windows dev machine)

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm --direct
python scripts/export_onnx.py
python scripts/build_faiss_index.py

# Start RabbitMQ + Redis via Docker (Windows)
docker compose up rabbitmq redis -d

# Start FastAPI dev server
uvicorn app.main:app --reload --port 8001

# Start a Celery worker (new terminal)
celery -A app.core.celery_app worker --queues normal,realtime,emergency,batch --concurrency 2 --loglevel info
```

---

## API Reference

### Async flow (all endpoints)

Every AI endpoint returns a `task_id` immediately:

```json
{ "task_id": "uuid-here", "status": "queued" }
```

If result is cached, returns immediately:
```json
{ "task_id": "cached", "status": "complete", "result": { ... } }
```

### Poll for result

```
GET /api/v1/task/{task_id}
X-AI-Secret: your-secret
```

Returns:
```json
{ "task_id": "...", "status": "complete", "result": { ... } }
```

### E-Consultation

| Endpoint | Queue | Typical latency |
|---|---|---|
| `POST /api/v1/consultation/transcribe` | realtime | 2-10s (audio length dependent) |
| `POST /api/v1/consultation/triage` | emergency or normal | 200-500ms |
| `POST /api/v1/consultation/soap-note` | normal | 3-8s |

### Health Social Media

| Endpoint | Queue | Typical latency |
|---|---|---|
| `POST /api/v1/social/moderate` | normal | 100-300ms |
| `POST /api/v1/social/recommend` | batch | 50-200ms |
| `POST /api/v1/social/sentiment` | batch | 50-150ms |

---

## Scaling guide

| Users | Strategy |
|---|---|
| < 10k/day | 1 FastAPI server, 1 Celery worker per queue |
| 100k/day | 2 FastAPI replicas, 2-4 workers per queue |
| 1M/day | Kubernetes, HPA on worker pods, Redis Cluster |
| 50k req/sec | K8s + KEDA autoscaling on RabbitMQ queue depth |

---

## Monitoring

- **Flower** (Celery tasks): http://localhost:5555
- **RabbitMQ**: http://localhost:15672
- **Prometheus metrics**: http://localhost:8001/metrics
- **Audit logs**: `logs/audit.log` 

---

