"""Book API routes (CRUD, borrow, return, reviews, analysis, recommendations)."""

import logging
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
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
    get_llm_service,
    get_preference_service,
    get_recommendation_service,
    get_review_service,
)
from app.domain.entities import BorrowRecord, User
from app.domain.repositories import IBorrowRepository
from app.services.background_tasks import (
    analyze_review_sentiment_task,
    generate_book_summary_task,
    update_rolling_consensus_task,
)
from app.services.book_service import BookService
from app.services.preference_service import PreferenceService
from app.services.recommendation import MLRecommendationService
from app.services.review_service import ReviewService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/books", tags=["books"])


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------
@router.post("/", response_model=BookResponse, status_code=status.HTTP_201_CREATED)
async def create_book(
    title: Annotated[str, Form()],
    author: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks,
    book_service: Annotated[BookService, Depends(get_book_service)],
    current_user: Annotated[User, Depends(get_current_user)],
    genre: Annotated[str, Form()] = "general",
) -> BookResponse:
    """Upload book file & metadata.

    The book record is created immediately.  Summary & embedding generation
    is dispatched as a **background task** so the client receives a fast
    ``201 Created`` response while the LLM work runs asynchronously.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File is required")

    file_content = await file.read()
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    mime_type = file.content_type or "application/octet-stream"
    try:
        book = await book_service.create_book(
            title=title,
            author=author,
            file_content=file_content,
            filename=file.filename,
            mime_type=mime_type,
            genre=genre,
        )

        # Dispatch background LLM summary + embedding generation
        llm_service = get_llm_service()
        background_tasks.add_task(
            generate_book_summary_task,
            book_id=str(book.id),
            file_content=file_content,
            mime_type=mime_type,
            llm_service=llm_service,
        )
        logger.info("Background summary task dispatched for book %s", book.id)

        return BookResponse.model_validate(book)
    except Exception as e:
        logger.error(f"Failed to create book: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create book")


@router.get("/", response_model=BookListResponse)
async def list_books(
    page: int = 1,
    limit: int = 20,
    book_service: Annotated[BookService, Depends(get_book_service)] = ...,
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
    book_service: Annotated[BookService, Depends(get_book_service)],
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
    book_service: Annotated[BookService, Depends(get_book_service)],
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
    book_service: Annotated[BookService, Depends(get_book_service)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> None:
    """Remove book and associated file."""
    deleted = await book_service.delete_book(book_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Book not found")


@router.get("/{book_id}/content")
async def download_book_content(
    book_id: UUID,
    book_service: Annotated[BookService, Depends(get_book_service)],
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
    background_tasks: BackgroundTasks,
    book_service: Annotated[BookService, Depends(get_book_service)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> BookResponse:
    """Replace the book's file content.

    The old file is deleted, the new file is stored, and summary & embedding
    regeneration is dispatched as a background task.
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

    # Regenerate summary & embedding for the new content
    llm_service = get_llm_service()
    background_tasks.add_task(
        generate_book_summary_task,
        book_id=str(book_id),
        file_content=file_content,
        mime_type=mime_type,
        llm_service=llm_service,
    )
    logger.info("Background summary re-generation dispatched for book %s", book_id)

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
    book_service: Annotated[BookService, Depends(get_book_service)],
    borrow_repo: Annotated[IBorrowRepository, Depends(get_borrow_repository)],
    pref_service: Annotated[PreferenceService, Depends(get_preference_service)],
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
    pref_service: Annotated[PreferenceService, Depends(get_preference_service)],
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
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_user)],
    review_service: Annotated[ReviewService, Depends(get_review_service)],
    pref_service: Annotated[PreferenceService, Depends(get_preference_service)],
) -> ReviewResponse:
    """Submit a review.

    The review is persisted immediately.  Sentiment analysis is dispatched as
    a **background task** so the response is fast.  User must have borrowed
    the book.
    """
    try:
        review = await review_service.create_review(
            user_id=current_user.id,
            book_id=book_id,
            rating=body.rating,
            text=body.text,
        )

        # Dispatch background sentiment analysis
        llm_service = get_llm_service()
        background_tasks.add_task(
            analyze_review_sentiment_task,
            review_id=str(review.id),
            review_text=body.text,
            llm_service=llm_service,
        )
        logger.info("Background sentiment task dispatched for review %s", review.id)

        # Dispatch background rolling consensus update
        background_tasks.add_task(
            update_rolling_consensus_task,
            book_id=str(book_id),
            llm_service=llm_service,
        )
        logger.info("Background consensus task dispatched for book %s", book_id)

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
    review_service: Annotated[ReviewService, Depends(get_review_service)],
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
        MLRecommendationService, Depends(get_recommendation_service)
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
