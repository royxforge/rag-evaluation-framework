"""SQLAlchemy async database models and session management."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# Imported for KEY_PREFIX_LENGTH; api.auth only imports api.database lazily
# inside functions, so this is not a circular import.
from api.auth import KEY_PREFIX_LENGTH

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://rag_evaluation_framework_user:rag_evaluation_framework_pass@localhost/rag_evaluation_framework_db",
)

engine = create_async_engine(DATABASE_URL, echo=False, pool_size=10, max_overflow=20)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash = Column(String(60), nullable=False)
    # Deterministic leading characters of the raw key, used for indexed
    # lookup (see api.auth.api_key_prefix). Nullable only for rows created
    # before migration 002; those keys must be rotated.
    key_prefix = Column(String(KEY_PREFIX_LENGTH), nullable=True, index=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    request_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    rate_limit_per_hour = Column(Integer, default=100)


class EvalResultDB(Base):
    __tablename__ = "eval_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    question = Column(Text, nullable=False)
    context = Column(JSONB, nullable=False)
    answer = Column(Text, nullable=False)
    overall_score = Column(Float, nullable=False)

    faithfulness_score = Column(Float, nullable=True)
    faithfulness_details = Column(JSONB, nullable=True)
    hallucination_rate = Column(Float, nullable=True)
    hallucination_details = Column(JSONB, nullable=True)
    retrieval_precision = Column(Float, nullable=True)
    retrieval_details = Column(JSONB, nullable=True)
    answer_relevance = Column(Float, nullable=True)
    relevance_details = Column(JSONB, nullable=True)
    context_coverage = Column(Float, nullable=True)
    coverage_details = Column(JSONB, nullable=True)
    ucm_score = Column(Float, nullable=True)
    ucm_details = Column(JSONB, nullable=True)

    llm_used = Column(String(100), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id"), nullable=True)
    # "metadata" is reserved by SQLAlchemy DeclarativeBase, so use "extra_metadata"
    extra_metadata = Column("metadata", JSONB, default=dict)

    __table_args__ = ({"extend_existing": True},)


class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String(20), default="queued")  # queued, running, completed, failed
    total_items = Column(Integer, nullable=False)
    completed_items = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)
    result_ids = Column(JSONB, default=list)
    webhook_url = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id"), nullable=True)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency: get async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Create all tables."""
    async with engine.begin() as conn:
        # Enable pgvector
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception:
            pass
        await conn.run_sync(Base.metadata.create_all)
