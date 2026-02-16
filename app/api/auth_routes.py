"""Authentication API routes."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import (
    LoginRequest,
    ProfileUpdateRequest,
    SignupRequest,
    TokenResponse,
    UserResponse,
)
from app.core.dependencies import get_current_user, get_user_repository
from app.domain.entities import User
from app.domain.repositories import IUserRepository
from app.services.auth_service import AuthService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


def _get_auth_service(
    user_repo: Annotated[IUserRepository, Depends(get_user_repository)],
) -> AuthService:
    return AuthService(user_repository=user_repo)


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    body: SignupRequest,
    auth_service: Annotated[AuthService, Depends(_get_auth_service)],
) -> UserResponse:
    """Register a new user."""
    try:
        user = await auth_service.signup(body.username, body.email, body.password)
        return UserResponse.model_validate(user)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    auth_service: Annotated[AuthService, Depends(_get_auth_service)],
) -> TokenResponse:
    """Authenticate and return JWT."""
    try:
        token = await auth_service.login(body.email, body.password)
        return TokenResponse(access_token=token)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.get("/profile", response_model=UserResponse)
async def get_profile(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserResponse:
    """Get the authenticated user's profile."""
    return UserResponse.model_validate(current_user)


@router.put("/profile", response_model=UserResponse)
async def update_profile(
    body: ProfileUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    auth_service: Annotated[AuthService, Depends(_get_auth_service)],
) -> UserResponse:
    """Update the authenticated user's profile."""
    try:
        updated = await auth_service.update_profile(current_user, body.username, body.email)
        return UserResponse.model_validate(updated)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/signout", status_code=status.HTTP_200_OK)
async def signout(
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict:
    """
    Sign out the current user.

    Note: With stateless JWT the client simply discards the token.
    A production system would maintain a token blacklist in Redis.
    """
    logger.info(f"User signed out: {current_user.id}")
    return {"detail": "Successfully signed out"}
