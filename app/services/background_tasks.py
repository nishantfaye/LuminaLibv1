"""Async implementations of LLM background work.

These coroutines contain the actual business logic executed by Celery workers.
Each function is fully self-contained:
  - opens its own DB session (independent of any request lifecycle)
  - instantiates the LLM service from config (no FastAPI DI required)
  - reads file content from storage rather than receiving raw bytes

The Celery task wrappers in ``app.infrastructure.tasks.llm_tasks`` call these
with ``asyncio.run()``, which is safe because each Celery worker process runs
its own event loop.
"""

import logging
import uuid as _uuid
from datetime import datetime

from sqlalchemy import select

from app.core.dependencies import get_llm_service, get_storage_service
from app.infrastructure.database.connection import worker_session_maker as async_session_maker
from app.infrastructure.database.models import BookAnalysisModel, ReviewModel
from app.infrastructure.database.repository import BookRepository, ReviewRepository

logger = logging.getLogger(__name__)


async def generate_book_summary_task(book_id: str, mime_type: str) -> None:
    """Generate summary & embedding for a newly uploaded book.

    Reads the file from storage (already persisted by the upload handler),
    extracts its text, calls the LLM, and saves the results to the DB.
    """
    from uuid import UUID

    from app.services.book_service import BookService

    logger.info("BG-TASK: generating summary for book %s", book_id)
    try:
        llm_service = get_llm_service()
        storage_service = get_storage_service()

        async with async_session_maker() as session:
            book_repo = BookRepository(session)
            book = await book_repo.get_by_id(UUID(book_id))
            if book is None:
                logger.error("BG-TASK: book %s not found", book_id)
                return

            # Read file from storage — avoids passing large bytes via Redis
            file_content = await storage_service.get_file(book.file_path)
            text_content = BookService._extract_text(file_content, mime_type)

            summary = await llm_service.generate_summary(text_content)
            embedding = await llm_service.generate_embedding(text_content)

            book.summary = summary
            book.embedding = embedding
            await book_repo.update(book)
            logger.info("BG-TASK: summary & embedding saved for book %s", book_id)
    except Exception as exc:
        logger.error("BG-TASK: summary failed for book %s — %s", book_id, exc, exc_info=True)
        raise


async def analyze_review_sentiment_task(review_id: str, review_text: str) -> None:
    """Run sentiment analysis on a submitted review and persist the result."""
    from uuid import UUID

    logger.info("BG-TASK: analyzing sentiment for review %s", review_id)
    try:
        llm_service = get_llm_service()
        sentiment = await llm_service.analyze_sentiment(review_text)

        async with async_session_maker() as session:
            result = await session.execute(
                select(ReviewModel).where(ReviewModel.id == UUID(review_id))
            )
            db_review = result.scalar_one_or_none()
            if db_review:
                db_review.sentiment = sentiment
                await session.commit()
                logger.info(
                    "BG-TASK: sentiment '%s' saved for review %s", sentiment, review_id
                )
            else:
                logger.error("BG-TASK: review %s not found", review_id)
    except Exception as exc:
        logger.error(
            "BG-TASK: sentiment failed for review %s — %s", review_id, exc, exc_info=True
        )
        raise


async def update_rolling_consensus_task(book_id: str) -> None:
    """Recompute the rolling review consensus for a book and cache it."""
    from uuid import UUID

    logger.info("BG-TASK: updating rolling consensus for book %s", book_id)
    try:
        llm_service = get_llm_service()

        async with async_session_maker() as session:
            book_repo = BookRepository(session)
            review_repo = ReviewRepository(session)

            bid = UUID(book_id)
            book = await book_repo.get_by_id(bid)
            if book is None:
                logger.error("BG-TASK: book %s not found for consensus", book_id)
                return

            reviews = await review_repo.get_by_book(bid)
            if not reviews:
                logger.info("BG-TASK: no reviews for book %s, skipping", book_id)
                return

            review_dicts = [
                {"rating": r.rating, "text": r.text, "sentiment": r.sentiment}
                for r in reviews
            ]
            avg_rating = sum(r.rating for r in reviews) / len(reviews)

            consensus = await llm_service.generate_review_consensus(
                book.title, review_dicts
            )

            result = await session.execute(
                select(BookAnalysisModel).where(BookAnalysisModel.book_id == bid)
            )
            analysis = result.scalar_one_or_none()

            if analysis is None:
                analysis = BookAnalysisModel(
                    id=_uuid.uuid4(),
                    book_id=bid,
                    review_count=len(reviews),
                    average_rating=round(avg_rating, 2),
                    consensus_summary=consensus,
                    updated_at=datetime.utcnow(),
                )
                session.add(analysis)
            else:
                analysis.review_count = len(reviews)
                analysis.average_rating = round(avg_rating, 2)
                analysis.consensus_summary = consensus
                analysis.updated_at = datetime.utcnow()

            await session.commit()
            logger.info(
                "BG-TASK: consensus saved for book %s (%d reviews, avg=%.2f)",
                book_id, len(reviews), avg_rating,
            )
    except Exception as exc:
        logger.error(
            "BG-TASK: consensus failed for book %s — %s", book_id, exc, exc_info=True
        )
        raise
