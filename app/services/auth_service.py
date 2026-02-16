"""Authentication service."""

import logging
from datetime import timedelta
from typing import Optional
from uuid import UUID, uuid4

from app.core.config import settings
from app.core.security import create_access_token, hash_password, verify_password
from app.domain.entities import User
from app.domain.repositories import IUserRepository

logger = logging.getLogger(__name__)


class AuthService:
    """Handles signup, login, profile, and token operations."""

    def __init__(self, user_repository: IUserRepository):
        self.user_repository = user_repository

    async def signup(self, username: str, email: str, password: str) -> User:
        """Register a new user."""
        existing = await self.user_repository.get_by_email(email)
        if existing:
            raise ValueError("Email already registered")
        existing = await self.user_repository.get_by_username(username)
        if existing:
            raise ValueError("Username already taken")

        user = User(
            id=uuid4(),
            username=username,
            email=email,
            hashed_password=hash_password(password),
        )
        created = await self.user_repository.create(user)
        logger.info(f"User registered: {created.id}")
        return created

    async def login(self, email: str, password: str) -> str:
        """Authenticate and return a JWT access token."""
        user = await self.user_repository.get_by_email(email)
        if not user or not verify_password(password, user.hashed_password):
            raise ValueError("Invalid email or password")
        if not user.is_active:
            raise ValueError("Account is deactivated")

        token = create_access_token(
            data={"sub": str(user.id)},
            expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
        )
        logger.info(f"User logged in: {user.id}")
        return token

    async def get_profile(self, user_id: UUID) -> Optional[User]:
        return await self.user_repository.get_by_id(user_id)

    async def update_profile(self, user: User, username: str, email: str) -> User:
        user.username = username
        user.email = email
        return await self.user_repository.update(user)
