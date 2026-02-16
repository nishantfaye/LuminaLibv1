"""Repository interfaces (ports) for dependency inversion."""

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from app.domain.entities import Book, BorrowRecord, Review, User, UserInteraction, UserPreference, UserTasteProfile


class IUserRepository(ABC):

    @abstractmethod
    async def create(self, user: User) -> User:
        pass

    @abstractmethod
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        pass

    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        pass

    @abstractmethod
    async def get_by_username(self, username: str) -> Optional[User]:
        pass

    @abstractmethod
    async def update(self, user: User) -> User:
        pass


class IBookRepository(ABC):

    @abstractmethod
    async def create(self, book: Book) -> Book:
        pass

    @abstractmethod
    async def get_by_id(self, book_id: UUID) -> Optional[Book]:
        pass

    @abstractmethod
    async def list_all(self, skip: int = 0, limit: int = 100) -> list[Book]:
        pass

    @abstractmethod
    async def count(self) -> int:
        pass

    @abstractmethod
    async def update(self, book: Book) -> Book:
        pass

    @abstractmethod
    async def update_content(
        self, book_id: UUID, file_path: str, file_size: int,
        mime_type: str, content_hash: str,
    ) -> Optional[Book]:
        """Update only the content-related fields (file_path, size, hash, mime).

        Resets summary and embedding to ``None`` so they can be regenerated.
        """
        pass

    @abstractmethod
    async def delete(self, book_id: UUID) -> bool:
        pass


class IBorrowRepository(ABC):

    @abstractmethod
    async def create(self, record: BorrowRecord) -> BorrowRecord:
        pass

    @abstractmethod
    async def get_active_borrow(self, user_id: UUID, book_id: UUID) -> Optional[BorrowRecord]:
        pass

    @abstractmethod
    async def return_book(self, record_id: UUID) -> BorrowRecord:
        pass

    @abstractmethod
    async def get_user_borrows(
        self, user_id: UUID, active_only: bool = False
    ) -> list[BorrowRecord]:
        pass

    @abstractmethod
    async def has_user_borrowed(self, user_id: UUID, book_id: UUID) -> bool:
        """Check if user has ever borrowed (and possibly returned) a book."""
        pass

    @abstractmethod
    async def get_all_borrows(self, limit: int = 5000) -> list[BorrowRecord]:
        """Get all borrow records across all users (for collaborative filtering)."""
        pass

    @abstractmethod
    async def get_book_borrows(self, book_id: UUID) -> list[BorrowRecord]:
        """Get all borrow records for a specific book."""
        pass


class IReviewRepository(ABC):

    @abstractmethod
    async def create(self, review: Review) -> Review:
        pass

    @abstractmethod
    async def get_by_book(self, book_id: UUID) -> list[Review]:
        pass

    @abstractmethod
    async def get_by_user_and_book(self, user_id: UUID, book_id: UUID) -> Optional[Review]:
        pass

    @abstractmethod
    async def get_all_reviews(self, limit: int = 5000) -> list[Review]:
        """Get all reviews across all users (for collaborative filtering)."""
        pass


class IUserPreferenceRepository(ABC):

    @abstractmethod
    async def get_or_create(self, user_id: UUID) -> UserPreference:
        pass

    @abstractmethod
    async def update(self, pref: UserPreference) -> UserPreference:
        pass


class IUserInteractionRepository(ABC):

    @abstractmethod
    async def record(self, interaction: UserInteraction) -> UserInteraction:
        """Record a new interaction event."""
        pass

    @abstractmethod
    async def get_user_interactions(
        self,
        user_id: UUID,
        interaction_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[UserInteraction]:
        """Get interactions for a user, optionally filtered by type."""
        pass

    @abstractmethod
    async def get_interaction_stats(self, user_id: UUID) -> dict:
        """Get aggregated stats: total borrows, reviews, avg rating, etc."""
        pass


class IUserTasteProfileRepository(ABC):

    @abstractmethod
    async def get_or_create(self, user_id: UUID) -> UserTasteProfile:
        pass

    @abstractmethod
    async def update(self, profile: UserTasteProfile) -> UserTasteProfile:
        pass


class IStorageService(ABC):

    @abstractmethod
    async def save_file(self, file_content: bytes, filename: str) -> str:
        pass

    @abstractmethod
    async def get_file(self, file_path: str) -> bytes:
        pass

    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        pass


class ILLMService(ABC):

    @abstractmethod
    async def generate_summary(self, content: str) -> str:
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        pass

    @abstractmethod
    async def generate_review_consensus(self, book_title: str, reviews: list[dict]) -> str:
        """Generate a rolling consensus summary from all reviews."""
        pass

    @abstractmethod
    async def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of a review text."""
        pass


class IRecommendationService(ABC):

    @abstractmethod
    async def get_recommendations(self, book_id: UUID, limit: int = 5) -> list[tuple[Book, float]]:
        pass

    @abstractmethod
    async def get_user_recommendations(
        self, user_id: UUID, limit: int = 10
    ) -> list[tuple[Book, float]]:
        """Get personalized recommendations for a user."""
        pass
