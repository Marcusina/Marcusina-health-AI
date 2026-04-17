"""
FastAPI Application — Production Entry Point
=============================================
Run with Gunicorn + uvicorn workers (NOT uvicorn --reload) in production:

    gunicorn app.main:app \\
        --workers 8 \\
        --worker-class uvicorn.workers.UvicornWorker \\
        --bind 0.0.0.0:8001 \\
        --timeout 120 \\
        --keep-alive 5 \\
        --log-level info

This spawns N uvicorn async workers. Each loads AI models once at startup.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger

from app.core.config import get_settings
from app.api.routes.v1 import router as v1_router
from app.utils.exceptions import register_exception_handlers

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models. Shutdown: nothing (GC handles cleanup)."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} [{settings.ENVIRONMENT}]")
    # Note: In Celery worker mode, models are loaded by the ModelTask base class
    # on first task execution, not here. FastAPI workers stay lightweight.
    logger.info("FastAPI worker ready. AI inference handled by Celery workers.")
    yield
    logger.info("FastAPI worker shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## Health AI Microservice v2

High-throughput AI service (50k+ req/sec) bridging Fastify and AI inference.

### How it works
1. Fastify POSTs to an endpoint with `X-AI-Secret` header
2. FastAPI validates, checks cache, enqueues Celery task → returns `task_id` in <5ms
3. Celery worker runs AI inference asynchronously
4. Result delivered via webhook callback OR Fastify polls `GET /api/v1/task/{task_id}`

### Authentication
All endpoints require: `X-AI-Secret: <shared-secret>`
    """,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,  # Disable docs in production
    redoc_url=None,
)

register_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["X-AI-Secret", "Content-Type"],
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

app.include_router(v1_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": settings.APP_VERSION}


@app.get("/", include_in_schema=False)
async def root():
    return {"service": settings.APP_NAME, "docs": "/docs"}
