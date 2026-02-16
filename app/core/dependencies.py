"""Dependency injection container."""

from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import decode_access_token
from app.domain.entities import User
from app.domain.repositories import (
    IBookRepository,
    IBorrowRepository,
    ILLMService,
    IReviewRepository,
    IStorageService,
    IUserInteractionRepository,
    IUserPreferenceRepository,
    IUserRepository,
    IUserTasteProfileRepository,
)
from app.infrastructure.database.connection import get_db
from app.infrastructure.database.repository import (
    BookRepository,
    BorrowRepository,
    ReviewRepository,
    UserInteractionRepository,
    UserPreferenceRepository,
    UserRepository,
    UserTasteProfileRepository,
)
from app.infrastructure.llm.services import LlamaLLMService, MockLLMService, OpenAILLMService
from app.infrastructure.storage.local import LocalStorageService
from app.infrastructure.storage.s3 import S3StorageService
from app.services.book_service import BookService
from app.services.preference_service import PreferenceService
from app.services.recommendation import MLRecommendationService
from app.services.review_service import ReviewService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ---------------------------------------------------------------------------
# Infrastructure providers
# ---------------------------------------------------------------------------
def get_storage_service() -> IStorageService:
    """Return the configured storage backend."""
    if settings.storage_backend == "local":
        return LocalStorageService(settings.storage_path)
    elif settings.storage_backend == "s3":
        return S3StorageService(
            bucket_name=settings.s3_bucket,
            region=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url or None,
            aws_access_key_id=settings.s3_access_key or None,
            aws_secret_access_key=settings.s3_secret_key or None,
        )
    raise ValueError(f"Unknown storage backend: {settings.storage_backend}")


def get_llm_service() -> ILLMService:
    """Return the configured LLM provider."""
    if settings.llm_provider == "mock":
        return MockLLMService()
    elif settings.llm_provider == "llama":
        return LlamaLLMService(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )
    elif settings.llm_provider == "openai":
        return OpenAILLMService(api_key=settings.llm_api_key)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


# ---------------------------------------------------------------------------
# Repository providers
# ---------------------------------------------------------------------------
async def get_user_repository(session: AsyncSession = Depends(get_db)) -> IUserRepository:
    return UserRepository(session)


async def get_book_repository(session: AsyncSession = Depends(get_db)) -> IBookRepository:
    return BookRepository(session)


async def get_borrow_repository(session: AsyncSession = Depends(get_db)) -> IBorrowRepository:
    return BorrowRepository(session)


async def get_review_repository(session: AsyncSession = Depends(get_db)) -> IReviewRepository:
    return ReviewRepository(session)


async def get_preference_repository(
    session: AsyncSession = Depends(get_db),
) -> IUserPreferenceRepository:
    return UserPreferenceRepository(session)


async def get_interaction_repository(
    session: AsyncSession = Depends(get_db),
) -> IUserInteractionRepository:
    return UserInteractionRepository(session)


async def get_taste_profile_repository(
    session: AsyncSession = Depends(get_db),
) -> IUserTasteProfileRepository:
    return UserTasteProfileRepository(session)


# ---------------------------------------------------------------------------
# Service providers
# ---------------------------------------------------------------------------
async def get_book_service(
    repo: IBookRepository = Depends(get_book_repository),
) -> BookService:
    """Get book service with dependencies."""
    return BookService(
        book_repository=repo,
        storage_service=get_storage_service(),
        llm_service=get_llm_service(),
    )


async def get_review_service(
    review_repo: IReviewRepository = Depends(get_review_repository),
    borrow_repo: IBorrowRepository = Depends(get_borrow_repository),
    book_repo: IBookRepository = Depends(get_book_repository),
) -> ReviewService:
    return ReviewService(
        review_repository=review_repo,
        borrow_repository=borrow_repo,
        book_repository=book_repo,
        llm_service=get_llm_service(),
    )


async def get_recommendation_service(
    book_repo: IBookRepository = Depends(get_book_repository),
    pref_repo: IUserPreferenceRepository = Depends(get_preference_repository),
    borrow_repo: IBorrowRepository = Depends(get_borrow_repository),
    review_repo: IReviewRepository = Depends(get_review_repository),
    taste_repo: IUserTasteProfileRepository = Depends(get_taste_profile_repository),
    interaction_repo: IUserInteractionRepository = Depends(get_interaction_repository),
) -> MLRecommendationService:
    return MLRecommendationService(
        book_repository=book_repo,
        preference_repository=pref_repo,
        borrow_repository=borrow_repo,
        review_repository=review_repo,
        taste_repository=taste_repo,
        interaction_repository=interaction_repo,
    )


async def get_preference_service(
    pref_repo: IUserPreferenceRepository = Depends(get_preference_repository),
    interaction_repo: IUserInteractionRepository = Depends(get_interaction_repository),
    taste_repo: IUserTasteProfileRepository = Depends(get_taste_profile_repository),
    book_repo: IBookRepository = Depends(get_book_repository),
    borrow_repo: IBorrowRepository = Depends(get_borrow_repository),
    review_repo: IReviewRepository = Depends(get_review_repository),
) -> PreferenceService:
    return PreferenceService(
        preference_repo=pref_repo,
        interaction_repo=interaction_repo,
        taste_repo=taste_repo,
        book_repo=book_repo,
        borrow_repo=borrow_repo,
        review_repo=review_repo,
        llm_service=get_llm_service(),
    )


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    user_repo: IUserRepository = Depends(get_user_repository),
) -> User:
    """Decode JWT and return the authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    user_id_str: str = payload.get("sub")
    if user_id_str is None:
        raise credentials_exception
    try:
        user_id = UUID(user_id_str)
    except ValueError:
        raise credentials_exception
    user = await user_repo.get_by_id(user_id)
    if user is None or not user.is_active:
        raise credentials_exception
    return user
