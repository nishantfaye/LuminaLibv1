"""Celery task wrappers for LLM background work.

Each task is a thin synchronous wrapper around the async coroutine defined in
``app.services.background_tasks``.  Celery workers maintain their own event
loop; ``asyncio.run()`` is safe here because each worker process runs
independently of the API server's event loop.

Retry policy (per task):
  - max_retries=3   — up to 3 additional attempts on failure
  - countdown=60    — wait 60 s before each retry (absorbs LLM/network blips)
"""

import asyncio
import logging

from app.infrastructure.tasks.celery_app import celery_app
from app.services.background_tasks import (
    analyze_review_sentiment_task,
    generate_book_summary_task,
    update_rolling_consensus_task,
)

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="llm.generate_book_summary", max_retries=3)
def generate_book_summary(self, book_id: str, mime_type: str) -> None:
    """Celery task: generate summary and embedding for a newly uploaded book."""
    try:
        asyncio.run(generate_book_summary_task(book_id, mime_type))
    except Exception as exc:
        logger.warning(
            "generate_book_summary failed (attempt %d/%d): %s",
            self.request.retries + 1,
            self.max_retries + 1,
            exc,
        )
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, name="llm.analyze_review_sentiment", max_retries=3)
def analyze_review_sentiment(self, review_id: str, review_text: str) -> None:
    """Celery task: run sentiment analysis on a submitted review."""
    try:
        asyncio.run(analyze_review_sentiment_task(review_id, review_text))
    except Exception as exc:
        logger.warning(
            "analyze_review_sentiment failed (attempt %d/%d): %s",
            self.request.retries + 1,
            self.max_retries + 1,
            exc,
        )
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, name="llm.update_rolling_consensus", max_retries=3)
def update_rolling_consensus(self, book_id: str) -> None:
    """Celery task: recompute the rolling review consensus for a book."""
    try:
        asyncio.run(update_rolling_consensus_task(book_id))
    except Exception as exc:
        logger.warning(
            "update_rolling_consensus failed (attempt %d/%d): %s",
            self.request.retries + 1,
            self.max_retries + 1,
            exc,
        )
        raise self.retry(exc=exc, countdown=60)
