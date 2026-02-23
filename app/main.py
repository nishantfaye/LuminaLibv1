"""Main FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth_routes import router as auth_router
from app.api.preference_routes import router as preference_router
from app.api.recommendation_routes import router as recommendation_router
from app.api.routes import router as books_router
from app.api.task_routes import router as task_router
from app.infrastructure.database.connection import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting LuminaLib application")
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down LuminaLib application")


app = FastAPI(
    title="LuminaLib",
    description="Intelligent, content-aware library system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(books_router)
app.include_router(recommendation_router)
app.include_router(preference_router)
app.include_router(task_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
