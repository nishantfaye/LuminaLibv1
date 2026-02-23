"""Database connection and session management."""

from typing import AsyncGenerator

from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.infrastructure.database.models import Base

# Pooled engine — used by FastAPI request handlers (long-lived process, one event loop)
engine = create_async_engine(settings.database_url, echo=False, future=True)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# NullPool engine — used by Celery workers (each asyncio.run() creates a new event loop;
# pooled connections are bound to the previous loop and cause "Future attached to a
# different loop" errors on the second task).  NullPool opens a fresh connection per
# session and closes it immediately, so there is nothing to reuse across loops.
worker_engine = create_async_engine(
    settings.database_url, echo=False, future=True, poolclass=NullPool
)
worker_session_maker = async_sessionmaker(
    worker_engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with async_session_maker() as session:
        yield session


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
