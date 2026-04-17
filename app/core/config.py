"""
Configuration
=============
All settings are read from environment variables (via .env file in dev,
real env vars in production). Strongly typed with Pydantic.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── App identity ─────────────────────────────────────────────────────────
    APP_NAME: str = "Health AI Service"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    # ── Server (Gunicorn + Uvicorn workers) ──────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    # Number of uvicorn worker processes. Rule of thumb: 2 × CPU cores + 1.
    # Override with WORKERS env var on the server.
    WORKERS: int = 4
    # Max concurrent requests per worker before queuing (uvicorn/asyncio limit)
    WORKER_CONNECTIONS: int = 1000

    # ── Security ─────────────────────────────────────────────────────────────
    API_SECRET_KEY: str = "change-me-in-production-use-32-char-min"
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:4000"]
    # Fastify callback URL for async task results
    FASTIFY_CALLBACK_URL: str = "http://localhost:3000/internal/ai-callback"
    FASTIFY_CALLBACK_SECRET: str = "change-me-callback-secret"

    # ── Redis cluster (cache + Celery result backend) ─────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_RESULT_URL: str = "redis://localhost:6379/1"   # Celery results
    CACHE_TTL_SECONDS: int = 3600
    CACHE_TTL_MODERATION: int = 1800    # Moderation results cached 30 min
    CACHE_TTL_TRIAGE: int = 300         # Triage = 5 min (symptoms may change)
    CACHE_TTL_RECOMMEND: int = 600      # Recommendations = 10 min

    # ── RabbitMQ (Celery broker) ──────────────────────────────────────────────
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672//"

    # ── Celery task queues ────────────────────────────────────────────────────
    CELERY_QUEUE_EMERGENCY: str = "emergency"   # Triage emergencies (highest priority)
    CELERY_QUEUE_REALTIME: str = "realtime"     # Consultation transcription
    CELERY_QUEUE_NORMAL: str = "normal"         # SOAP notes, moderation
    CELERY_QUEUE_BATCH: str = "batch"           # Recommendations, analytics

    # ── Model paths ───────────────────────────────────────────────────────────
    MODELS_DIR: str = "./models"
    ONNX_MODELS_DIR: str = "./models/onnx"

    # ── faster-whisper ────────────────────────────────────────────────────────
    # Model size: tiny|base|small|medium|large-v3
    # "base" for dev, "medium" or "large-v3" for production.
    # compute_type: "int8" (CPU, fastest), "float16" (GPU), "float32" (fallback)
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_COMPUTE_TYPE: Literal["int8", "float16", "float32"] = "int8"
    WHISPER_DEVICE: Literal["cpu", "cuda", "auto"] = "cpu"
    # Number of parallel beam search workers inside whisper (tune per server CPU count)
    WHISPER_NUM_WORKERS: int = 4
    WHISPER_BEAM_SIZE: int = 5
    # Batch audio segments for faster transcription of long recordings
    WHISPER_BATCH_SIZE: int = 16

    # ── ONNX NLP Models ───────────────────────────────────────────────────────
    # HuggingFace model IDs — converted to ONNX at setup time via scripts/export_onnx.py
    # After export, inference uses ONNX Runtime (no PyTorch at runtime).
    HF_NER_MODEL: str = "samrawal/bert-base-uncased_clinical-ner"
    HF_SUMMARIZER_MODEL: str = "facebook/bart-large-cnn"
    HF_TRIAGE_MODEL: str = "distilbert-base-uncased"
    HF_MISINFO_MODEL: str = "hamzab/roberta-fake-news-classification"
    HF_SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── ONNX Runtime settings ─────────────────────────────────────────────────
    # Number of threads for ONNX intra-op parallelism per worker process
    ONNX_INTRA_THREADS: int = 2
    ONNX_INTER_THREADS: int = 1

    # ── FAISS vector index ────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = "./models/faiss/content_index.bin"
    FAISS_METADATA_PATH: str = "./models/faiss/content_metadata.json"

    # ── Inference thresholds ──────────────────────────────────────────────────
    MISINFO_THRESHOLD: float = 0.75
    TRIAGE_EMERGENCY_THRESHOLD: float = 0.85
    TRIAGE_URGENT_THRESHOLD: float = 0.60

    # ── Request batching (dynamic batching for throughput) ────────────────────
    # Collect this many requests before running a batched inference pass.
    # Lower = lower latency. Higher = higher throughput.
    BATCH_MAX_SIZE: int = 32
    BATCH_TIMEOUT_MS: int = 50          # Max wait before flushing partial batch


@lru_cache()
def get_settings() -> Settings:
    return Settings()
