"""Book API routes (CRUD, borrow, return, reviews, analysis, recommendations)."""

import logging
from typing import Annotated, Optional
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from app.api.schemas import (
    BookAnalysisResponse,
    BookListResponse,
    BookResponse,
    BookUpdate,
    BorrowResponse,
    RecommendationResponse,
    RecommendedBookResponse,
    ReviewCreateRequest,
    ReviewResponse,
)
from app.core.dependencies import (
    get_book_service,
    get_borrow_repository,
    get_current_user,
    get_preference_service,
    get_recommendation_service,
    get_review_service,
)
from app.domain.entities import BorrowRecord, User
from app.domain.repositories import IBorrowRepository, IRecommendationService
from app.domain.services import IBookService, IPreferenceService, IReviewService
from app.infrastructure.tasks.llm_tasks import (
    analyze_review_sentiment,
    generate_book_summary,
    update_rolling_consensus,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/books", tags=["books"])


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------
@router.post("/", response_model=BookResponse, status_code=status.HTTP_201_CREATED)
async def create_book(
    file: Annotated[UploadFile, File()],
    response: Response,
    book_service: Annotated[IBookService, Depends(get_book_service)],
    current_user: Annotated[User, Depends(get_current_user)],
    title: Annotated[Optional[str], Form()] = None,
    author: Annotated[Optional[str], Form()] = None,
    genre: Annotated[Optional[str], Form()] = None,
) -> BookResponse:
    """Upload a book file — metadata is optional (content-first ingestion).

    The file is the primary artifact.  Title, author, and genre are derived
    automatically from the content when not supplied:

    1. PDF built-in header fields (title / author / subject).
    2. LLM extraction from the opening text of the file.
    3. Fallback: filename stem, "Unknown Author", "general".

    Caller-supplied values always override extracted values.

    The book record is created immediately and the ``X-Task-ID`` header in
    the response contains a Celery task ID that can be polled at
    ``GET /tasks/{task_id}`` to track summary & embedding generation.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File is required")

    file_content = await file.read()
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    mime_type = file.content_type or "application/octet-stream"
    try:
        book = await book_service.create_book(
            file_content=file_content,
            filename=file.filename,
            mime_type=mime_type,
            title=title,
            author=author,
            genre=genre,
        )

        # Dispatch summary + embedding generation to Celery worker
        task = generate_book_summary.delay(str(book.id), mime_type)
        response.headers["X-Task-ID"] = task.id
        logger.info("Celery summary task %s dispatched for book %s", task.id, book.id)

        return BookResponse.model_validate(book)
    except Exception as e:
        logger.error(f"Failed to create book: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create book")


@router.get("/", response_model=BookListResponse)
async def list_books(
    page: int = 1,
    limit: int = 20,
    book_service: Annotated[IBookService, Depends(get_book_service)] = ...,
    current_user: Annotated[User, Depends(get_current_user)] = ...,
) -> BookListResponse:
    """List books with pagination."""
    skip = (page - 1) * limit
    books = await book_service.list_books(skip=skip, limit=limit)
    total = await book_service.count_books()
    return BookListResponse(
        books=[BookResponse.model_validate(b) for b in books],
        total=total,
        page=page,
        limit=limit,
    )


@router.get("/{book_id}", response_model=BookResponse)
async def get_book(
    book_id: UUID,
    book_service: Annotated[IBookService, Depends(get_book_service)],
    current_user: Annotated[User, Depends(get_current_user)] = ...,
) -> BookResponse:
    """Get a book by ID."""
    book = await book_service.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return BookResponse.model_validate(book)


@router.put("/{book_id}", response_model=BookResponse)
async def update_book(
    book_id: UUID,
    body: BookUpdate,
    book_service: Annotated[IBookService, Depends(get_book_service)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> BookResponse:
    """Update book details."""
    updated = await book_service.update_book(book_id, body.title, body.author, body.genre)
    if not updated:
        raise HTTPException(status_code=404, detail="Book not found")
    return BookResponse.model_validate(updated)


@router.delete("/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_book(
    book_id: UUID,
    book_service: Annotated[IBookService, Depends(get_book_service)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> None:
    """Remove book and associated file."""
    deleted = await book_service.delete_book(book_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Book not found")


@router.get("/{book_id}/content")
async def download_book_content(
    book_id: UUID,
    book_service: Annotated[IBookService, Depends(get_book_service)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> Response:
    """Download the stored book file content."""
    result = await book_service.get_book_content(book_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Book not found")
    file_bytes, mime_type, filename = result
    return Response(
        content=file_bytes,
        media_type=mime_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.put("/{book_id}/content", response_model=BookResponse)
async def update_book_content(
    book_id: UUID,
    file: Annotated[UploadFile, File()],
    response: Response,
    book_service: Annotated[IBookService, Depends(get_book_service)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> BookResponse:
    """Replace the book's file content.

    The old file is deleted, the new file is stored, and summary & embedding
    regeneration is dispatched as a **Celery task**.  The task ID is returned
    in the ``X-Task-ID`` response header.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File is required")
    file_content = await file.read()
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    mime_type = file.content_type or "application/octet-stream"
    updated = await book_service.update_book_content(
        book_id=book_id,
        file_content=file_content,
        filename=file.filename,
        mime_type=mime_type,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Book not found")

    # Dispatch summary re-generation to Celery worker
    task = generate_book_summary.delay(str(book_id), mime_type)
    response.headers["X-Task-ID"] = task.id
    logger.info("Celery summary re-generation task %s dispatched for book %s", task.id, book_id)

    return BookResponse.model_validate(updated)


# ---------------------------------------------------------------------------
# Borrow / Return
# ---------------------------------------------------------------------------
@router.post(
    "/{book_id}/borrow", response_model=BorrowResponse, status_code=status.HTTP_201_CREATED
)
async def borrow_book(
    book_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    book_service: Annotated[IBookService, Depends(get_book_service)],
    borrow_repo: Annotated[IBorrowRepository, Depends(get_borrow_repository)],
    pref_service: Annotated[IPreferenceService, Depends(get_preference_service)],
) -> BorrowResponse:
    """User borrows a book."""
    book = await book_service.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    active = await borrow_repo.get_active_borrow(current_user.id, book_id)
    if active:
        raise HTTPException(status_code=400, detail="You already have this book borrowed")

    record = BorrowRecord(id=uuid4(), user_id=current_user.id, book_id=book_id)
    created = await borrow_repo.create(record)

    # Record implicit interaction (Layer 2)
    try:
        await pref_service.record_interaction(
            user_id=current_user.id,
            book_id=book_id,
            interaction_type="borrow",
            interaction_data={"book_title": book.title, "author": book.author},
        )
    except Exception as exc:
        logger.warning("Failed to record borrow interaction: %s", exc)

    return BorrowResponse.model_validate(created)


@router.post("/{book_id}/return", response_model=BorrowResponse)
async def return_book(
    book_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    borrow_repo: Annotated[IBorrowRepository, Depends(get_borrow_repository)],
    pref_service: Annotated[IPreferenceService, Depends(get_preference_service)],
) -> BorrowResponse:
    """User returns a book."""
    active = await borrow_repo.get_active_borrow(current_user.id, book_id)
    if not active:
        raise HTTPException(status_code=400, detail="No active borrow found for this book")

    returned = await borrow_repo.return_book(active.id)

    # Record implicit interaction (Layer 2)
    try:
        await pref_service.record_interaction(
            user_id=current_user.id,
            book_id=book_id,
            interaction_type="return",
        )
    except Exception as exc:
        logger.warning("Failed to record return interaction: %s", exc)

    return BorrowResponse.model_validate(returned)


# ---------------------------------------------------------------------------
# Reviews & Analysis
# ---------------------------------------------------------------------------
@router.post(
    "/{book_id}/reviews", response_model=ReviewResponse, status_code=status.HTTP_201_CREATED
)
async def create_review(
    book_id: UUID,
    body: ReviewCreateRequest,
    response: Response,
    current_user: Annotated[User, Depends(get_current_user)],
    review_service: Annotated[IReviewService, Depends(get_review_service)],
    pref_service: Annotated[IPreferenceService, Depends(get_preference_service)],
) -> ReviewResponse:
    """Submit a review.

    The review is persisted immediately.  Sentiment analysis and rolling
    consensus update are dispatched as **Celery tasks** so the response is
    fast.  Task IDs are returned in ``X-Sentiment-Task-ID`` and
    ``X-Consensus-Task-ID`` response headers.  User must have borrowed the book.
    """
    try:
        review = await review_service.create_review(
            user_id=current_user.id,
            book_id=book_id,
            rating=body.rating,
            text=body.text,
        )

        # Dispatch sentiment analysis to Celery worker
        sentiment_task = analyze_review_sentiment.delay(str(review.id), body.text)
        response.headers["X-Sentiment-Task-ID"] = sentiment_task.id
        logger.info("Celery sentiment task %s dispatched for review %s", sentiment_task.id, review.id)

        # Dispatch rolling consensus update to Celery worker
        consensus_task = update_rolling_consensus.delay(str(book_id))
        response.headers["X-Consensus-Task-ID"] = consensus_task.id
        logger.info("Celery consensus task %s dispatched for book %s", consensus_task.id, book_id)

        # Record implicit interaction (Layer 2)
        try:
            await pref_service.record_interaction(
                user_id=current_user.id,
                book_id=book_id,
                interaction_type="review",
                interaction_data={"rating": body.rating, "text_length": len(body.text)},
            )
        except Exception as exc:
            logger.warning("Failed to record review interaction: %s", exc)

        return ReviewResponse.model_validate(review)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{book_id}/analysis", response_model=BookAnalysisResponse)
async def get_book_analysis(
    book_id: UUID,
    review_service: Annotated[IReviewService, Depends(get_review_service)],
    current_user: Annotated[User, Depends(get_current_user)] = ...,
) -> BookAnalysisResponse:
    """Get GenAI-aggregated summary of all reviews."""
    try:
        analysis = await review_service.get_book_analysis(book_id)
        return BookAnalysisResponse(**analysis)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
@router.get("/{book_id}/recommendations", response_model=RecommendationResponse)
async def get_book_recommendations(
    book_id: UUID,
    limit: int = 5,
    recommendation_service: Annotated[
        IRecommendationService, Depends(get_recommendation_service)
    ] = ...,
    current_user: Annotated[User, Depends(get_current_user)] = ...,
) -> RecommendationResponse:
    """Get similar book recommendations."""
    results = await recommendation_service.get_recommendations(book_id, limit)
    recs = [
        RecommendedBookResponse(**BookResponse.model_validate(book).model_dump(), score=score)
        for book, score in results
    ]
    return RecommendationResponse(
        recommendations=recs,
        total=len(recs),
        strategy="content-based",
    )
