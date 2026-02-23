"""Domain-level application service interfaces (ports).

These abstract classes define the contracts that the API layer depends on.
Concrete implementations live in ``app/services/`` and are wired together
by the composition root in ``app/core/dependencies.py``.

Keeping these interfaces in the domain layer means:
  - Route handlers import from ``app.domain`` only — no concrete deps.
  - Swapping an implementation requires zero changes in the API layer.
  - Every service can be replaced with a test double via FastAPI's
    ``app.dependency_overrides`` without touching business logic.
"""

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from app.domain.entities import (
    Book,
    Review,
    UserInteraction,
    UserPreference,
    UserTasteProfile,
)


class IBookService(ABC):

    @abstractmethod
    async def create_book(
        self,
        file_content: bytes,
        filename: str,
        mime_type: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> Book:
        """Ingest a book file.

        *Content-first*: the file is the primary artifact.  Title, author, and
        genre are resolved in priority order:

        1. Caller-supplied values (override everything).
        2. Embedded PDF metadata (title / author fields in the PDF header).
        3. LLM extraction from the first few thousand characters of text.
        4. Fallback: filename stem as title, ``"Unknown Author"``, ``"general"``.
        """
        pass

    @abstractmethod
    async def get_book(self, book_id: UUID) -> Optional[Book]:
        pass

    @abstractmethod
    async def list_books(self, skip: int = 0, limit: int = 100) -> list[Book]:
        pass

    @abstractmethod
    async def count_books(self) -> int:
        pass

    @abstractmethod
    async def update_book(
        self, book_id: UUID, title: str, author: str, genre: str = "general"
    ) -> Optional[Book]:
        pass

    @abstractmethod
    async def update_book_content(
        self,
        book_id: UUID,
        file_content: bytes,
        filename: str,
        mime_type: str,
    ) -> Optional[Book]:
        pass

    @abstractmethod
    async def get_book_content(
        self, book_id: UUID
    ) -> Optional[tuple[bytes, str, str]]:
        """Return (file_bytes, mime_type, filename) or None."""
        pass

    @abstractmethod
    async def delete_book(self, book_id: UUID) -> bool:
        pass


class IReviewService(ABC):

    @abstractmethod
    async def create_review(
        self, user_id: UUID, book_id: UUID, rating: int, text: str
    ) -> Review:
        pass

    @abstractmethod
    async def get_book_analysis(self, book_id: UUID) -> dict:
        """Return GenAI-aggregated analysis of all reviews for a book."""
        pass


class IPreferenceService(ABC):

    # --- Layer 1: Explicit ---

    @abstractmethod
    async def get_preferences(self, user_id: UUID) -> UserPreference:
        pass

    @abstractmethod
    async def update_preferences(
        self,
        user_id: UUID,
        *,
        preferred_authors: Optional[list[str]] = None,
        preferred_genres: Optional[list[str]] = None,
        preferred_tags: Optional[list[str]] = None,
        reading_pace: Optional[str] = None,
        discovery_mode: Optional[str] = None,
        content_preferences: Optional[dict] = None,
        notify_new_by_favorite_author: Optional[bool] = None,
    ) -> UserPreference:
        pass

    # --- Layer 2: Implicit ---

    @abstractmethod
    async def record_interaction(
        self,
        user_id: UUID,
        book_id: UUID,
        interaction_type: str,
        interaction_data: Optional[dict] = None,
    ) -> UserInteraction:
        pass

    @abstractmethod
    async def get_interaction_history(
        self,
        user_id: UUID,
        interaction_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[UserInteraction]:
        pass

    @abstractmethod
    async def get_interaction_stats(self, user_id: UUID) -> dict:
        pass

    # --- Layer 3: Computed ---

    @abstractmethod
    async def recompute_taste_profile(self, user_id: UUID) -> UserTasteProfile:
        pass

    @abstractmethod
    async def get_taste_profile(self, user_id: UUID) -> UserTasteProfile:
        pass

    @abstractmethod
    async def get_preference_snapshot(self, user_id: UUID) -> dict:
        pass
