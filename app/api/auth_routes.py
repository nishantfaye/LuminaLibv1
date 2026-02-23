"""Authentication API routes."""

import logging
import time
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.api.schemas import (
    LoginRequest,
    ProfileUpdateRequest,
    SignupRequest,
    TokenResponse,
    UserResponse,
)
from app.core.dependencies import get_current_user, get_user_repository
from app.core.redis_client import get_redis, revoke_token
from app.core.security import decode_access_token
from app.domain.entities import User
from app.domain.repositories import IUserRepository
from app.services.auth_service import AuthService

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

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
    token: Annotated[str, Depends(_oauth2_scheme)],
    redis_client: Annotated[aioredis.Redis, Depends(get_redis)],
) -> dict:
    """Sign out the current user.

    The token's ``jti`` is written to the Redis revocation blacklist with a
    TTL equal to the token's remaining lifetime.  Subsequent requests carrying
    this token will be rejected by ``get_current_user`` even before the JWT
    expires naturally.
    """
    payload = decode_access_token(token)
    if payload:
        jti: str | None = payload.get("jti")
        exp: int | None = payload.get("exp")
        if jti and exp:
            ttl = max(int(exp - time.time()), 1)
            await revoke_token(redis_client, jti, ttl)
            logger.info("Token jti=%s revoked (TTL=%ds) for user %s", jti, ttl, current_user.id)
    return {"detail": "Successfully signed out"}
