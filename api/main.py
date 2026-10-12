"""FastAPI application factory."""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.database import init_db
from api.routers import api_keys, batch, evaluate, reports

logger = structlog.get_logger()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator:
        """Application lifespan handler."""
        # Startup
        logger.info("Starting RAG Evaluation Framework API")
        try:
            await init_db()
            logger.info("Database initialized")
        except Exception as e:
            logger.warning(
                "Database initialization failed (may not be available yet)", error=str(e)
            )

        yield

        # Shutdown
        logger.info("Shutting down RAG Evaluation Framework API")

    app = FastAPI(
        title="RAG Evaluation Framework API",
        description="Production-grade RAG evaluation API",
        version="0.4.0",
        lifespan=lifespan,
        docs_url="/docs" if os.environ.get("DEBUG", "").lower() == "true" else None,
        redoc_url=None,
    )

    # ── Middleware ──────────────────────────────────────────

    # CORS
    allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # ── Routes ─────────────────────────────────────────────

    app.include_router(evaluate.router, tags=["Evaluation"])
    app.include_router(batch.router, tags=["Batch"])
    app.include_router(api_keys.router, tags=["API Keys"])
    app.include_router(reports.router, tags=["Reports"])

    # ── Health Check ───────────────────────────────────────

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        import redis.asyncio as aioredis

        db_status = "unknown"
        redis_status = "unknown"

        # Check DB
        try:
            from api.database import async_session_factory

            async with async_session_factory() as session:
                from sqlalchemy import text

                await session.execute(text("SELECT 1"))
                db_status = "ok"
        except Exception:
            db_status = "error"

        # Check Redis
        try:
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            r = aioredis.from_url(redis_url, decode_responses=True)
            await r.ping()
            redis_status = "ok"
        except Exception:
            redis_status = "error"

        return {
            "status": "ok" if db_status == "ok" and redis_status == "ok" else "degraded",
            "version": "0.1.0",
            "database": db_status,
            "redis": redis_status,
        }

    return app


app = create_app()
