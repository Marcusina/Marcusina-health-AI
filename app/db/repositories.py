from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from sqlalchemy import select, func, desc

from app.db.engine import SyncSessionLocal, AsyncSessionLocal
from app.db.models import AITask, AuditEvent, InferenceMetric


# ============================================================================ #
# Sync writes — called from Celery workers                                      #
# ============================================================================ #

def persist_task_result(
    task_id: str,
    task_type: str,
    entity_id: Optional[str],
    entity_type: Optional[str],
    duration_ms: Optional[int],
    result_summary: Optional[dict],
    urgency_level: Optional[str] = None,
    verdict: Optional[str] = None,
    sentiment: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Upsert a completed task row. Called at the end of every Celery task."""
    try:
        with SyncSessionLocal() as session:
            existing = session.execute(
                select(AITask).where(AITask.task_id == task_id)
            ).scalar_one_or_none()

            now = datetime.now(timezone.utc)
            if existing:
                existing.status = "failed" if error else "complete"
                existing.completed_at = now
                existing.duration_ms = duration_ms
                existing.result_summary = result_summary
                existing.urgency_level = urgency_level
                existing.verdict = verdict
                existing.sentiment = sentiment
                existing.error = error
            else:
                session.add(AITask(
                    task_id=task_id,
                    task_type=task_type,
                    status="failed" if error else "complete",
                    entity_id=entity_id,
                    entity_type=entity_type,
                    duration_ms=duration_ms,
                    result_summary=result_summary,
                    urgency_level=urgency_level,
                    verdict=verdict,
                    sentiment=sentiment,
                    error=error,
                    completed_at=now,
                ))
            session.commit()
    except Exception as e:
        logger.warning(f"[DB] persist_task_result failed for {task_id}: {e}")


def persist_audit_event(
    audit_id: str,
    action: str,
    module: str,
    entity_id: Optional[str],
    data: dict,
) -> None:
    """Mirror an audit event to the database alongside the NDJSON log file."""
    try:
        with SyncSessionLocal() as session:
            session.add(AuditEvent(
                audit_id=audit_id,
                action=action,
                module=module,
                entity_id=entity_id,
                data=data,
            ))
            session.commit()
    except Exception as e:
        logger.warning(f"[DB] persist_audit_event failed ({action}): {e}")


def persist_inference_metric(
    task_id: str,
    model_name: str,
    latency_ms: float,
    top_label: Optional[str] = None,
    top_score: Optional[float] = None,
) -> None:
    """Record a single model inference call with its latency and top prediction."""
    try:
        with SyncSessionLocal() as session:
            session.add(InferenceMetric(
                task_id=task_id,
                model_name=model_name,
                latency_ms=round(latency_ms, 2),
                top_label=top_label,
                top_score=top_score,
            ))
            session.commit()
    except Exception as e:
        logger.warning(f"[DB] persist_inference_metric failed ({model_name}): {e}")


# ============================================================================ #
# Async writes — called from FastAPI routes (fire-and-forget)                   #
# ============================================================================ #

async def register_task_async(
    task_id: str,
    task_type: str,
    entity_id: Optional[str],
    entity_type: Optional[str],
) -> None:
    """Insert a 'queued' row when a task is first dispatched from FastAPI."""
    try:
        async with AsyncSessionLocal() as session:
            session.add(AITask(
                task_id=task_id,
                task_type=task_type,
                status="queued",
                entity_id=entity_id,
                entity_type=entity_type,
            ))
            await session.commit()
    except Exception as e:
        logger.warning(f"[DB] register_task_async failed for {task_id}: {e}")


# ============================================================================ #
# Async reads — called from FastAPI query endpoints                             #
# ============================================================================ #

async def get_task_history(
    task_type: Optional[str],
    entity_id: Optional[str],
    status: Optional[str],
    limit: int,
    offset: int,
) -> list[dict]:
    try:
        async with AsyncSessionLocal() as session:
            q = select(AITask).order_by(desc(AITask.created_at)).limit(limit).offset(offset)
            if task_type:
                q = q.where(AITask.task_type == task_type)
            if entity_id:
                q = q.where(AITask.entity_id == entity_id)
            if status:
                q = q.where(AITask.status == status)
            rows = (await session.execute(q)).scalars().all()
            return [_task_to_dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"[DB] get_task_history failed: {e}")
        return []


async def get_inference_metrics_summary() -> list[dict]:
    try:
        async with AsyncSessionLocal() as session:
            q = (
                select(
                    InferenceMetric.model_name,
                    func.count().label("calls"),
                    func.avg(InferenceMetric.latency_ms).label("avg_latency_ms"),
                    func.min(InferenceMetric.latency_ms).label("min_latency_ms"),
                    func.max(InferenceMetric.latency_ms).label("max_latency_ms"),
                    func.avg(InferenceMetric.top_score).label("avg_top_score"),
                )
                .group_by(InferenceMetric.model_name)
                .order_by(desc(func.count()))
            )
            rows = (await session.execute(q)).all()
            return [
                {
                    "model_name": r.model_name,
                    "total_calls": r.calls,
                    "avg_latency_ms": round(r.avg_latency_ms or 0, 2),
                    "min_latency_ms": round(r.min_latency_ms or 0, 2),
                    "max_latency_ms": round(r.max_latency_ms or 0, 2),
                    "avg_top_score": round(r.avg_top_score or 0, 3),
                }
                for r in rows
            ]
    except Exception as e:
        logger.warning(f"[DB] get_inference_metrics_summary failed: {e}")
        return []


async def get_audit_events(
    action: Optional[str],
    entity_id: Optional[str],
    limit: int,
    offset: int,
) -> list[dict]:
    try:
        async with AsyncSessionLocal() as session:
            q = select(AuditEvent).order_by(desc(AuditEvent.created_at)).limit(limit).offset(offset)
            if action:
                q = q.where(AuditEvent.action == action)
            if entity_id:
                q = q.where(AuditEvent.entity_id == entity_id)
            rows = (await session.execute(q)).scalars().all()
            return [
                {
                    "audit_id": r.audit_id,
                    "action": r.action,
                    "module": r.module,
                    "entity_id": r.entity_id,
                    "data": r.data,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
    except Exception as e:
        logger.warning(f"[DB] get_audit_events failed: {e}")
        return []


# ── Internal helpers ──────────────────────────────────────────────────────────

def _task_to_dict(task: AITask) -> dict:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "status": task.status,
        "entity_id": task.entity_id,
        "entity_type": task.entity_type,
        "urgency_level": task.urgency_level,
        "verdict": task.verdict,
        "sentiment": task.sentiment,
        "duration_ms": task.duration_ms,
        "result_summary": task.result_summary,
        "error": task.error,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }
