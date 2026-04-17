# =============================================================================
# Health AI Microservice — Production Dockerfile
# Target: Linux (Ubuntu 22.04 base). Develops on Windows, runs on Linux.
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder) — install Python deps (heavier, not shipped)
#   Stage 2 (runtime) — lean final image with only what's needed at runtime

FROM python:3.11-slim-bookworm AS builder

# System deps needed to BUILD some packages (gcc for hiredis, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# Install to a local dir so we can copy cleanly to runtime stage
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

# Runtime system deps:
#   libgomp1 → OpenMP (needed by faiss-cpu and ONNX Runtime)
#   curl     → health checks
#   Note: NO ffmpeg package needed — faster-whisper uses PyAV (bundled)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Download spaCy English model (needed by presidio)
RUN python -m spacy download en_core_web_sm

# Copy application source
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create dirs for models and logs
RUN mkdir -p models/whisper models/onnx models/faiss models/embedders logs

# Non-root user for security
RUN useradd -m -u 1001 aiuser && chown -R aiuser:aiuser /app
USER aiuser

EXPOSE 8001

HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default: Gunicorn with uvicorn workers
# Override CMD for Celery workers in docker-compose
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8001", \
     "--timeout", "30", \
     "--log-level", "info"]
