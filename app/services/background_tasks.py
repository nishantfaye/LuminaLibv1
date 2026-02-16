"""Background tasks for asynchronous LLM processing.

FastAPI's ``BackgroundTasks`` mechanism is used so the HTTP response returns
immediately while heavy LLM work (summary generation, sentiment analysis,
rolling consensus updates) runs in the background.
"""

import logging
import uuid as _uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.repositories import IBookRepository, ILLMService, IReviewRepository
from app.infrastructure.database.connection import async_session_maker
from app.infrastructure.database.models import BookAnalysisModel, ReviewModel
from app.infrastructure.database.repository import BookRepository, ReviewRepository

logger = logging.getLogger(__name__)


async def generate_book_summary_task(
    book_id: str,
    file_content: bytes,
    mime_type: str,
    llm_service: ILLMService,
) -> None:
    """Background task: generate summary & embedding for a newly uploaded book.

    Opens its own DB session so it is independent of the request lifecycle.
    """
    from uuid import UUID

    from app.services.book_service import BookService

    logger.info("BG-TASK: generating summary for book %s", book_id)
    try:
        async with async_session_maker() as session:
            book_repo = BookRepository(session)
            book = await book_repo.get_by_id(UUID(book_id))
            if book is None:
                logger.error("BG-TASK: book %s not found", book_id)
                return

            text_content = BookService._extract_text(file_content, mime_type)
            summary = await llm_service.generate_summary(text_content)
            embedding = await llm_service.generate_embedding(text_content)

            book.summary = summary
            book.embedding = embedding
            await book_repo.update(book)
            logger.info("BG-TASK: summary & embedding saved for book %s", book_id)
    except Exception as exc:
        logger.error("BG-TASK: failed for book %s — %s", book_id, exc, exc_info=True)


async def analyze_review_sentiment_task(
    review_id: str,
    review_text: str,
    llm_service: ILLMService,
) -> None:
    """Background task: run sentiment analysis on a submitted review.

    Opens its own DB session so it is independent of the request lifecycle.
    """
    from uuid import UUID

    logger.info("BG-TASK: analyzing sentiment for review %s", review_id)
    try:
        sentiment = await llm_service.analyze_sentiment(review_text)
        async with async_session_maker() as session:
            review_repo = ReviewRepository(session)
            # Update review sentiment directly via model
            from sqlalchemy import select

            from app.infrastructure.database.models import ReviewModel

            result = await session.execute(
                select(ReviewModel).where(ReviewModel.id == UUID(review_id))
            )
            db_review = result.scalar_one_or_none()
            if db_review:
                db_review.sentiment = sentiment
                await session.commit()
                logger.info("BG-TASK: sentiment '%s' saved for review %s", sentiment, review_id)
            else:
                logger.error("BG-TASK: review %s not found", review_id)
    except Exception as exc:
        logger.error("BG-TASK: failed for review %s — %s", review_id, exc, exc_info=True)


async def update_rolling_consensus_task(
    book_id: str,
    llm_service: ILLMService,
) -> None:
    """Background task: recompute the rolling review consensus for a book.

    Reads all reviews for the book, asks the LLM for a consensus summary,
    and persists the result to the ``book_analyses`` table so that
    ``GET /books/{id}/analysis`` can return the cached result instantly.

    Opens its own DB session so it is independent of the request lifecycle.
    """
    from uuid import UUID

    logger.info("BG-TASK: updating rolling consensus for book %s", book_id)
    try:
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
                logger.info("BG-TASK: no reviews for book %s, skipping consensus", book_id)
                return

            # Build review dicts for the LLM
            review_dicts = [
                {"rating": r.rating, "text": r.text, "sentiment": r.sentiment}
                for r in reviews
            ]
            avg_rating = sum(r.rating for r in reviews) / len(reviews)

            # Ask the LLM for a consensus summary
            consensus = await llm_service.generate_review_consensus(
                book.title, review_dicts
            )

            # Upsert into book_analyses table
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
                "BG-TASK: rolling consensus saved for book %s (%d reviews, avg=%.2f)",
                book_id, len(reviews), avg_rating,
            )
    except Exception as exc:
        logger.error(
            "BG-TASK: consensus update failed for book %s — %s",
            book_id, exc, exc_info=True,
        )
