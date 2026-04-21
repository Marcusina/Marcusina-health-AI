from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Text, Float, Integer, JSON, DateTime, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class AITask(Base):
    """One row per Celery task — the persistent record of every AI job."""
    __tablename__ = "ai_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="queued", nullable=False)

    # Which entity this task belongs to (patient / content / user / session)
    entity_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    entity_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Task-type-specific shortcut columns (avoid JSONB parse for common filters)
    urgency_level: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    verdict: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    sentiment: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)

    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_ai_tasks_type_created", "task_type", "created_at"),
        Index("ix_ai_tasks_status", "status"),
    )


class AuditEvent(Base):
    """Queryable mirror of the NDJSON audit log in logs/audit.log."""
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audit_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    module: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_audit_events_action_created", "action", "created_at"),
    )


class InferenceMetric(Base):
    """Per-model-call timing and score — enables performance monitoring and drift detection."""
    __tablename__ = "inference_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    top_label: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    top_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_inference_metrics_model_created", "model_name", "created_at"),
    )
