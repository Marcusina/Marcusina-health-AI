

from celery import Celery
from celery.schedules import crontab
from kombu import Queue, Exchange
from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "health_ai",
    broker=settings.RABBITMQ_URL,
    backend=settings.REDIS_RESULT_URL,
    include=[
        "app.tasks.consultation_tasks",
        "app.tasks.social_media_tasks",
        "app.tasks.support_tasks",
    ],
)

celery_app.conf.update(
    # ── Serialisation ──────────────────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # ── Result storage ─────────────────────────────────────────────────────
    result_expires=3600,            # Results expire after 1h
    result_persistent=True,

    # ── Routing: map task names → priority queues ──────────────────────────
    task_routes={
        "app.tasks.consultation_tasks.task_triage_emergency": {"queue": settings.CELERY_QUEUE_EMERGENCY},
        "app.tasks.consultation_tasks.task_transcribe":       {"queue": settings.CELERY_QUEUE_REALTIME},
        "app.tasks.consultation_tasks.task_soap_note":        {"queue": settings.CELERY_QUEUE_NORMAL},
        "app.tasks.consultation_tasks.task_triage_normal":    {"queue": settings.CELERY_QUEUE_NORMAL},
        "app.tasks.social_media_tasks.task_moderate":         {"queue": settings.CELERY_QUEUE_NORMAL},
        "app.tasks.social_media_tasks.task_recommend":        {"queue": settings.CELERY_QUEUE_BATCH},
        "app.tasks.social_media_tasks.task_sentiment":        {"queue": settings.CELERY_QUEUE_BATCH},
        "app.tasks.support_tasks.task_support_assist":        {"queue": settings.CELERY_QUEUE_NORMAL},
    },

    # ── Concurrency ────────────────────────────────────────────────────────
    # worker_concurrency is set via CLI: celery -A app.core.celery_app worker
    # --concurrency=N (set N = CPU cores for CPU-bound inference tasks)
    worker_prefetch_multiplier=1,   # Pull one task at a time (fair for long tasks)
    worker_max_tasks_per_child=1000,  # Recycle workers to prevent memory fragmentation
    broker_connection_retry_on_startup=True,
    task_acks_late=True,            # Ack only after successful completion
    task_reject_on_worker_lost=True,

    # ── Time limits ────────────────────────────────────────────────────────
    task_soft_time_limit=30,        # Warn at 30s
    task_time_limit=60,             # Kill at 60s

    # ── Retry policy ──────────────────────────────────────────────────────
    task_max_retries=3,
    task_default_retry_delay=2,     # seconds

    # ── Queue definitions ──────────────────────────────────────────────────
    task_queues=[
        Queue(settings.CELERY_QUEUE_EMERGENCY, Exchange("emergency"), routing_key="emergency"),
        Queue(settings.CELERY_QUEUE_REALTIME,  Exchange("realtime"),  routing_key="realtime"),
        Queue(settings.CELERY_QUEUE_NORMAL,    Exchange("normal"),    routing_key="normal"),
        Queue(settings.CELERY_QUEUE_BATCH,     Exchange("batch"),     routing_key="batch"),
    ],

    # ── Beat scheduler (periodic tasks) ────────────────────────────────────
    beat_schedule={
        "rebuild-faiss-index-nightly": {
            "task": "app.tasks.social_media_tasks.task_rebuild_faiss_index",
            "schedule": crontab(hour=2, minute=0),  # 2AM daily
            "options": {"queue": settings.CELERY_QUEUE_BATCH},
        },
        "clear-expired-cache": {
            "task": "app.tasks.social_media_tasks.task_clear_expired_cache",
            "schedule": crontab(minute="*/30"),     # Every 30 min
            "options": {"queue": settings.CELERY_QUEUE_BATCH},
        },
    },
)
