"""Review service with business logic."""

import logging
from uuid import UUID, uuid4

from app.domain.entities import Review
from app.domain.repositories import (
    IBookRepository,
    IBorrowRepository,
    ILLMService,
    IReviewRepository,
)

logger = logging.getLogger(__name__)


class ReviewService:
    """Handles review creation, sentiment analysis, and consensus generation."""

    def __init__(
        self,
        review_repository: IReviewRepository,
        borrow_repository: IBorrowRepository,
        book_repository: IBookRepository,
        llm_service: ILLMService,
    ):
        self.review_repository = review_repository
        self.borrow_repository = borrow_repository
        self.book_repository = book_repository
        self.llm_service = llm_service

    async def create_review(self, user_id: UUID, book_id: UUID, rating: int, text: str) -> Review:
        """Create a review — user must have borrowed the book.

        The review is persisted immediately.  Sentiment analysis and rolling
        consensus update are handled by **background tasks** dispatched from
        the API layer.
        """
        # Constraint: user must have borrowed (current or past) the book
        has_borrowed = await self.borrow_repository.has_user_borrowed(user_id, book_id)
        if not has_borrowed:
            raise PermissionError("You must borrow the book before reviewing it")

        # Check duplicate
        existing = await self.review_repository.get_by_user_and_book(user_id, book_id)
        if existing:
            raise ValueError("You have already reviewed this book")

        review = Review(
            id=uuid4(),
            user_id=user_id,
            book_id=book_id,
            rating=rating,
            text=text,
            sentiment=None,  # will be filled by background task
        )
        created = await self.review_repository.create(review)
        logger.info(f"Review created: {created.id} for book {book_id}")
        return created

    async def get_book_analysis(self, book_id: UUID) -> dict:
        """Get GenAI-aggregated analysis of all reviews for a book.

        First checks the cached ``book_analyses`` table (populated by the
        rolling consensus background task).  Falls back to on-demand LLM
        generation if no cached result exists.
        """
        book = await self.book_repository.get_by_id(book_id)
        if not book:
            raise ValueError("Book not found")

        reviews = await self.review_repository.get_by_book(book_id)
        if not reviews:
            return {
                "book_id": str(book_id),
                "book_title": book.title,
                "review_count": 0,
                "average_rating": 0.0,
                "consensus_summary": "No reviews yet.",
            }

        avg_rating = round(sum(r.rating for r in reviews) / len(reviews), 2)

        # --- Try cached consensus from background task ---
        cached = await self._get_cached_analysis(book_id)
        if cached and cached.review_count == len(reviews):
            # Cache is up-to-date (same review count)
            logger.info("Returning cached consensus for book %s", book_id)
            return {
                "book_id": str(book_id),
                "book_title": book.title,
                "review_count": cached.review_count,
                "average_rating": cached.average_rating,
                "consensus_summary": cached.consensus_summary,
            }

        # --- Fallback: on-demand LLM generation ---
        logger.info("Cache miss or stale for book %s — generating consensus on-demand", book_id)
        review_dicts = [
            {"rating": r.rating, "text": r.text, "sentiment": r.sentiment} for r in reviews
        ]
        consensus = await self.llm_service.generate_review_consensus(book.title, review_dicts)

        return {
            "book_id": str(book_id),
            "book_title": book.title,
            "review_count": len(reviews),
            "average_rating": avg_rating,
            "consensus_summary": consensus,
        }

    async def _get_cached_analysis(self, book_id: UUID):
        """Read cached BookAnalysis from the DB (if available).

        Uses the same session that the review_repository holds so no new
        connection is needed.
        """
        try:
            from sqlalchemy import select
            from app.infrastructure.database.models import BookAnalysisModel

            session = self.review_repository.session  # type: ignore[attr-defined]
            result = await session.execute(
                select(BookAnalysisModel).where(BookAnalysisModel.book_id == book_id)
            )
            return result.scalar_one_or_none()
        except Exception as exc:
            logger.warning("Failed to read cached analysis for book %s: %s", book_id, exc)
            return None
