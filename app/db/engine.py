from __future__ import annotations
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings

settings = get_settings()

# ── Async engine — used by FastAPI routes ────────────────────────────────────
async_engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    echo=settings.DEBUG,
)
AsyncSessionLocal = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# ── Sync engine — used by Celery workers (no event loop) ─────────────────────
sync_engine = create_engine(
    settings.DATABASE_SYNC_URL,
    pool_size=10,
    max_overflow=5,
    pool_pre_ping=True,
)
SyncSessionLocal = sessionmaker(sync_engine)


async def init_db() -> None:
    """Create all tables if they don't exist. Called once at FastAPI startup."""
    from app.db.models import Base
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
