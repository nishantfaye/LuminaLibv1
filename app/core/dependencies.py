"""Dependency injection container."""

from typing import Annotated
from uuid import UUID

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.redis_client import get_redis, is_token_revoked
from app.core.security import decode_access_token
from app.domain.entities import User
from app.domain.repositories import (
    IBookAnalysisRepository,
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
    BookAnalysisRepository,
    BookRepository,
    BorrowRepository,
    ReviewRepository,
    UserInteractionRepository,
    UserPreferenceRepository,
    UserRepository,
    UserTasteProfileRepository,
)
from app.domain.repositories import IRecommendationService
from app.domain.services import IBookService, IPreferenceService, IReviewService
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


async def get_book_analysis_repository(
    session: AsyncSession = Depends(get_db),
) -> IBookAnalysisRepository:
    return BookAnalysisRepository(session)


# ---------------------------------------------------------------------------
# Service providers
# ---------------------------------------------------------------------------
async def get_book_service(
    repo: IBookRepository = Depends(get_book_repository),
    storage: IStorageService = Depends(get_storage_service),
    llm: ILLMService = Depends(get_llm_service),
) -> IBookService:
    """Get book service with dependencies."""
    return BookService(
        book_repository=repo,
        storage_service=storage,
        llm_service=llm,
    )


async def get_review_service(
    review_repo: IReviewRepository = Depends(get_review_repository),
    borrow_repo: IBorrowRepository = Depends(get_borrow_repository),
    book_repo: IBookRepository = Depends(get_book_repository),
    book_analysis_repo: IBookAnalysisRepository = Depends(get_book_analysis_repository),
    llm: ILLMService = Depends(get_llm_service),
) -> IReviewService:
    return ReviewService(
        review_repository=review_repo,
        borrow_repository=borrow_repo,
        book_repository=book_repo,
        book_analysis_repository=book_analysis_repo,
        llm_service=llm,
    )


async def get_recommendation_service(
    book_repo: IBookRepository = Depends(get_book_repository),
    pref_repo: IUserPreferenceRepository = Depends(get_preference_repository),
    borrow_repo: IBorrowRepository = Depends(get_borrow_repository),
    review_repo: IReviewRepository = Depends(get_review_repository),
    taste_repo: IUserTasteProfileRepository = Depends(get_taste_profile_repository),
    interaction_repo: IUserInteractionRepository = Depends(get_interaction_repository),
) -> IRecommendationService:
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
    llm: ILLMService = Depends(get_llm_service),
) -> IPreferenceService:
    return PreferenceService(
        preference_repo=pref_repo,
        interaction_repo=interaction_repo,
        taste_repo=taste_repo,
        book_repo=book_repo,
        borrow_repo=borrow_repo,
        review_repo=review_repo,
        llm_service=llm,
    )


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    user_repo: IUserRepository = Depends(get_user_repository),
    redis_client: aioredis.Redis = Depends(get_redis),
) -> User:
    """Decode JWT and return the authenticated user.

    Rejects tokens whose ``jti`` has been written to the Redis revocation
    blacklist (i.e. the user has signed out).
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    # Check revocation blacklist
    jti: str | None = payload.get("jti")
    if jti and await is_token_revoked(redis_client, jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

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
