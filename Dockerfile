# =============================================================================
# Marcusina Health AI Microservice
# =============================================================================
# Multi-stage Docker image for the production API, Celery worker, and optional
# playground server. docker-compose.free.yml reuses this same image with
# different commands for each service.

FROM python:3.11-slim-bookworm AS builder

# Build deps for Python packages with native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements_core.txt requirements_ml.txt requirements_nlp.txt requirements_observability.txt ./

# Install all dependency groups together so pip can resolve shared constraints.
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements_core.txt \
    -r requirements_ml.txt \
    -r requirements_nlp.txt \
    -r requirements_observability.txt


FROM python:3.11-slim-bookworm AS runtime

# Runtime deps:
#   libgomp1 -> OpenMP for faiss-cpu and ONNX Runtime
#   curl     -> Docker health checks
# faster-whisper uses PyAV, so no separate ffmpeg package is required here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local

# Presidio needs the spaCy English model. Installing the pinned model wheel is
# more reliable in Docker than `python -m spacy download en_core_web_sm`.
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

COPY app/ ./app/
COPY playground/ ./playground/
COPY scripts/ ./scripts/

RUN mkdir -p models/whisper models/onnx models/faiss models/embedders logs

RUN useradd -m -u 1001 aiuser && chown -R aiuser:aiuser /app
USER aiuser

EXPOSE 8001 8800

HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command for the API service. Compose overrides this for workers and
# the playground service.
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8001", \
     "--timeout", "60", \
     "--log-level", "info"]
