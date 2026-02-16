"""Repository implementations."""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import Book, BorrowRecord, Review, User, UserInteraction, UserPreference, UserTasteProfile
from app.domain.repositories import (
    IBookRepository,
    IBorrowRepository,
    IReviewRepository,
    IUserInteractionRepository,
    IUserPreferenceRepository,
    IUserRepository,
    IUserTasteProfileRepository,
)
from app.infrastructure.database.models import (
    BookAnalysisModel,
    BookModel,
    BorrowModel,
    ReviewModel,
    UserInteractionModel,
    UserModel,
    UserPreferenceModel,
    UserTasteProfileModel,
)


# ---------------------------------------------------------------------------
# User Repository
# ---------------------------------------------------------------------------
class UserRepository(IUserRepository):

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, user: User) -> User:
        db_user = UserModel(
            id=user.id,
            username=user.username,
            email=user.email,
            hashed_password=user.hashed_password,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )
        self.session.add(db_user)
        await self.session.commit()
        await self.session.refresh(db_user)
        return self._to_entity(db_user)

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        result = await self.session.execute(select(UserModel).where(UserModel.id == user_id))
        db_user = result.scalar_one_or_none()
        return self._to_entity(db_user) if db_user else None

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.session.execute(select(UserModel).where(UserModel.email == email))
        db_user = result.scalar_one_or_none()
        return self._to_entity(db_user) if db_user else None

    async def get_by_username(self, username: str) -> Optional[User]:
        result = await self.session.execute(select(UserModel).where(UserModel.username == username))
        db_user = result.scalar_one_or_none()
        return self._to_entity(db_user) if db_user else None

    async def update(self, user: User) -> User:
        result = await self.session.execute(select(UserModel).where(UserModel.id == user.id))
        db_user = result.scalar_one()
        db_user.username = user.username
        db_user.email = user.email
        db_user.hashed_password = user.hashed_password
        db_user.is_active = user.is_active
        db_user.updated_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(db_user)
        return self._to_entity(db_user)

    @staticmethod
    def _to_entity(model: UserModel) -> User:
        return User(
            id=model.id,
            username=model.username,
            email=model.email,
            hashed_password=model.hashed_password,
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


# ---------------------------------------------------------------------------
# Book Repository
# ---------------------------------------------------------------------------
class BookRepository(IBookRepository):

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, book: Book) -> Book:
        db_book = BookModel(
            id=book.id,
            title=book.title,
            author=book.author,
            genre=book.genre,
            file_path=book.file_path,
            file_size=book.file_size,
            mime_type=book.mime_type,
            content_hash=book.content_hash,
            summary=book.summary,
            embedding=book.embedding,
            created_at=book.created_at,
            updated_at=book.updated_at,
        )
        self.session.add(db_book)
        await self.session.commit()
        await self.session.refresh(db_book)
        return self._to_entity(db_book)

    async def get_by_id(self, book_id: UUID) -> Optional[Book]:
        result = await self.session.execute(select(BookModel).where(BookModel.id == book_id))
        db_book = result.scalar_one_or_none()
        return self._to_entity(db_book) if db_book else None

    async def list_all(self, skip: int = 0, limit: int = 100) -> list[Book]:
        result = await self.session.execute(select(BookModel).offset(skip).limit(limit))
        return [self._to_entity(book) for book in result.scalars().all()]

    async def count(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(BookModel))
        return result.scalar_one()

    async def update(self, book: Book) -> Book:
        result = await self.session.execute(select(BookModel).where(BookModel.id == book.id))
        db_book = result.scalar_one()
        db_book.title = book.title
        db_book.author = book.author
        db_book.genre = book.genre
        db_book.summary = book.summary
        db_book.embedding = book.embedding
        db_book.updated_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(db_book)
        return self._to_entity(db_book)

    async def update_content(
        self, book_id: UUID, file_path: str, file_size: int,
        mime_type: str, content_hash: str,
    ) -> Optional[Book]:
        """Update content-related fields and reset summary/embedding."""
        result = await self.session.execute(select(BookModel).where(BookModel.id == book_id))
        db_book = result.scalar_one_or_none()
        if db_book is None:
            return None
        db_book.file_path = file_path
        db_book.file_size = file_size
        db_book.mime_type = mime_type
        db_book.content_hash = content_hash
        db_book.summary = None
        db_book.embedding = None
        db_book.updated_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(db_book)
        return self._to_entity(db_book)

    async def delete(self, book_id: UUID) -> bool:
        result = await self.session.execute(select(BookModel).where(BookModel.id == book_id))
        db_book = result.scalar_one_or_none()
        if db_book:
            await self.session.delete(db_book)
            await self.session.commit()
            return True
        return False

    @staticmethod
    def _to_entity(model: BookModel) -> Book:
        return Book(
            id=model.id,
            title=model.title,
            author=model.author,
            genre=model.genre,
            file_path=model.file_path,
            file_size=model.file_size,
            mime_type=model.mime_type,
            content_hash=model.content_hash,
            summary=model.summary,
            embedding=model.embedding,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


# ---------------------------------------------------------------------------
# Borrow Repository
# ---------------------------------------------------------------------------
class BorrowRepository(IBorrowRepository):

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, record: BorrowRecord) -> BorrowRecord:
        db_record = BorrowModel(
            id=record.id,
            user_id=record.user_id,
            book_id=record.book_id,
            borrowed_at=record.borrowed_at,
            returned_at=record.returned_at,
        )
        self.session.add(db_record)
        await self.session.commit()
        await self.session.refresh(db_record)
        return self._to_entity(db_record)

    async def get_active_borrow(self, user_id: UUID, book_id: UUID) -> Optional[BorrowRecord]:
        result = await self.session.execute(
            select(BorrowModel).where(
                BorrowModel.user_id == user_id,
                BorrowModel.book_id == book_id,
                BorrowModel.returned_at.is_(None),
            )
        )
        db_record = result.scalar_one_or_none()
        return self._to_entity(db_record) if db_record else None

    async def return_book(self, record_id: UUID) -> BorrowRecord:
        result = await self.session.execute(select(BorrowModel).where(BorrowModel.id == record_id))
        db_record = result.scalar_one()
        db_record.returned_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(db_record)
        return self._to_entity(db_record)

    async def get_user_borrows(
        self, user_id: UUID, active_only: bool = False
    ) -> list[BorrowRecord]:
        stmt = select(BorrowModel).where(BorrowModel.user_id == user_id)
        if active_only:
            stmt = stmt.where(BorrowModel.returned_at.is_(None))
        result = await self.session.execute(stmt)
        return [self._to_entity(r) for r in result.scalars().all()]

    async def has_user_borrowed(self, user_id: UUID, book_id: UUID) -> bool:
        result = await self.session.execute(
            select(BorrowModel).where(
                BorrowModel.user_id == user_id,
                BorrowModel.book_id == book_id,
            )
        )
        return result.scalars().first() is not None

    async def get_all_borrows(self, limit: int = 5000) -> list[BorrowRecord]:
        result = await self.session.execute(
            select(BorrowModel).order_by(BorrowModel.borrowed_at.desc()).limit(limit)
        )
        return [self._to_entity(r) for r in result.scalars().all()]

    async def get_book_borrows(self, book_id: UUID) -> list[BorrowRecord]:
        result = await self.session.execute(
            select(BorrowModel).where(BorrowModel.book_id == book_id)
        )
        return [self._to_entity(r) for r in result.scalars().all()]

    @staticmethod
    def _to_entity(model: BorrowModel) -> BorrowRecord:
        return BorrowRecord(
            id=model.id,
            user_id=model.user_id,
            book_id=model.book_id,
            borrowed_at=model.borrowed_at,
            returned_at=model.returned_at,
        )


# ---------------------------------------------------------------------------
# Review Repository
# ---------------------------------------------------------------------------
class ReviewRepository(IReviewRepository):

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, review: Review) -> Review:
        db_review = ReviewModel(
            id=review.id,
            user_id=review.user_id,
            book_id=review.book_id,
            rating=review.rating,
            text=review.text,
            sentiment=review.sentiment,
            created_at=review.created_at,
        )
        self.session.add(db_review)
        await self.session.commit()
        await self.session.refresh(db_review)
        return self._to_entity(db_review)

    async def get_by_book(self, book_id: UUID) -> list[Review]:
        result = await self.session.execute(
            select(ReviewModel).where(ReviewModel.book_id == book_id)
        )
        return [self._to_entity(r) for r in result.scalars().all()]

    async def get_by_user_and_book(self, user_id: UUID, book_id: UUID) -> Optional[Review]:
        result = await self.session.execute(
            select(ReviewModel).where(
                ReviewModel.user_id == user_id,
                ReviewModel.book_id == book_id,
            )
        )
        db_review = result.scalar_one_or_none()
        return self._to_entity(db_review) if db_review else None

    async def get_all_reviews(self, limit: int = 5000) -> list[Review]:
        result = await self.session.execute(
            select(ReviewModel).order_by(ReviewModel.created_at.desc()).limit(limit)
        )
        return [self._to_entity(r) for r in result.scalars().all()]

    @staticmethod
    def _to_entity(model: ReviewModel) -> Review:
        return Review(
            id=model.id,
            user_id=model.user_id,
            book_id=model.book_id,
            rating=model.rating,
            text=model.text,
            sentiment=model.sentiment,
            created_at=model.created_at,
        )


# ---------------------------------------------------------------------------
# User Preference Repository
# ---------------------------------------------------------------------------
class UserPreferenceRepository(IUserPreferenceRepository):

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create(self, user_id: UUID) -> UserPreference:
        result = await self.session.execute(
            select(UserPreferenceModel).where(UserPreferenceModel.user_id == user_id)
        )
        db_pref = result.scalar_one_or_none()
        if not db_pref:
            db_pref = UserPreferenceModel(
                id=uuid4(),
                user_id=user_id,
                preferred_authors=[],
                preferred_genres=[],
                preferred_tags=[],
                reading_pace="moderate",
                discovery_mode="balanced",
                notify_new_by_favorite_author=False,
            )
            self.session.add(db_pref)
            await self.session.commit()
            await self.session.refresh(db_pref)
        return self._to_entity(db_pref)

    async def update(self, pref: UserPreference) -> UserPreference:
        result = await self.session.execute(
            select(UserPreferenceModel).where(UserPreferenceModel.user_id == pref.user_id)
        )
        db_pref = result.scalar_one()
        db_pref.preferred_authors = pref.preferred_authors
        db_pref.preferred_genres = pref.preferred_genres
        db_pref.preferred_tags = pref.preferred_tags
        db_pref.reading_pace = pref.reading_pace
        db_pref.discovery_mode = pref.discovery_mode
        db_pref.content_preferences = pref.content_preferences
        db_pref.notify_new_by_favorite_author = pref.notify_new_by_favorite_author
        db_pref.embedding = pref.embedding
        db_pref.updated_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(db_pref)
        return self._to_entity(db_pref)

    @staticmethod
    def _to_entity(model: UserPreferenceModel) -> UserPreference:
        return UserPreference(
            id=model.id,
            user_id=model.user_id,
            preferred_authors=model.preferred_authors or [],
            preferred_genres=model.preferred_genres or [],
            preferred_tags=model.preferred_tags or [],
            reading_pace=model.reading_pace or "moderate",
            discovery_mode=model.discovery_mode or "balanced",
            content_preferences=model.content_preferences,
            notify_new_by_favorite_author=model.notify_new_by_favorite_author or False,
            embedding=model.embedding,
            updated_at=model.updated_at,
        )


# ---------------------------------------------------------------------------
# User Interaction Repository (Layer 2 — Implicit Signals)
# ---------------------------------------------------------------------------
class UserInteractionRepository(IUserInteractionRepository):

    def __init__(self, session: AsyncSession):
        self.session = session

    async def record(self, interaction: UserInteraction) -> UserInteraction:
        db_interaction = UserInteractionModel(
            id=interaction.id,
            user_id=interaction.user_id,
            book_id=interaction.book_id,
            interaction_type=interaction.interaction_type,
            interaction_data=interaction.interaction_data,
            weight=interaction.weight,
            created_at=interaction.created_at,
        )
        self.session.add(db_interaction)
        await self.session.commit()
        await self.session.refresh(db_interaction)
        return self._to_entity(db_interaction)

    async def get_user_interactions(
        self,
        user_id: UUID,
        interaction_type: str | None = None,
        limit: int = 100,
    ) -> list[UserInteraction]:
        stmt = (
            select(UserInteractionModel)
            .where(UserInteractionModel.user_id == user_id)
            .order_by(UserInteractionModel.created_at.desc())
            .limit(limit)
        )
        if interaction_type:
            stmt = stmt.where(UserInteractionModel.interaction_type == interaction_type)
        result = await self.session.execute(stmt)
        return [self._to_entity(r) for r in result.scalars().all()]

    async def get_interaction_stats(self, user_id: UUID) -> dict:
        """Aggregate stats for a user's interactions."""
        # Count by type
        result = await self.session.execute(
            select(
                UserInteractionModel.interaction_type,
                func.count(UserInteractionModel.id),
            )
            .where(UserInteractionModel.user_id == user_id)
            .group_by(UserInteractionModel.interaction_type)
        )
        type_counts = {row[0]: row[1] for row in result.all()}

        # Average rating from review interactions
        review_result = await self.session.execute(
            select(UserInteractionModel.interaction_data)
            .where(
                UserInteractionModel.user_id == user_id,
                UserInteractionModel.interaction_type == "review",
            )
        )
        ratings = []
        for row in review_result.scalars().all():
            if row and isinstance(row, dict) and "rating" in row:
                ratings.append(row["rating"])
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0

        return {
            "total_borrows": type_counts.get("borrow", 0),
            "total_returns": type_counts.get("return", 0),
            "total_reviews": type_counts.get("review", 0),
            "total_bookmarks": type_counts.get("bookmark", 0),
            "avg_rating_given": round(avg_rating, 2),
            "interaction_counts": type_counts,
        }

    @staticmethod
    def _to_entity(model: UserInteractionModel) -> UserInteraction:
        return UserInteraction(
            id=model.id,
            user_id=model.user_id,
            book_id=model.book_id,
            interaction_type=model.interaction_type,
            interaction_data=model.interaction_data,
            weight=model.weight,
            created_at=model.created_at,
        )


# ---------------------------------------------------------------------------
# User Taste Profile Repository (Layer 3 — Computed)
# ---------------------------------------------------------------------------
class UserTasteProfileRepository(IUserTasteProfileRepository):

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_or_create(self, user_id: UUID) -> UserTasteProfile:
        result = await self.session.execute(
            select(UserTasteProfileModel).where(UserTasteProfileModel.user_id == user_id)
        )
        db_profile = result.scalar_one_or_none()
        if not db_profile:
            db_profile = UserTasteProfileModel(
                id=uuid4(),
                user_id=user_id,
            )
            self.session.add(db_profile)
            await self.session.commit()
            await self.session.refresh(db_profile)
        return self._to_entity(db_profile)

    async def update(self, profile: UserTasteProfile) -> UserTasteProfile:
        result = await self.session.execute(
            select(UserTasteProfileModel).where(
                UserTasteProfileModel.user_id == profile.user_id
            )
        )
        db_profile = result.scalar_one()
        db_profile.genre_affinities = profile.genre_affinities
        db_profile.author_affinities = profile.author_affinities
        db_profile.avg_rating_given = profile.avg_rating_given
        db_profile.total_borrows = profile.total_borrows
        db_profile.total_reviews = profile.total_reviews
        db_profile.taste_embedding = profile.taste_embedding
        db_profile.taste_cluster = profile.taste_cluster
        db_profile.confidence_score = profile.confidence_score
        db_profile.last_computed_at = profile.last_computed_at
        await self.session.commit()
        await self.session.refresh(db_profile)
        return self._to_entity(db_profile)

    @staticmethod
    def _to_entity(model: UserTasteProfileModel) -> UserTasteProfile:
        return UserTasteProfile(
            id=model.id,
            user_id=model.user_id,
            genre_affinities=model.genre_affinities,
            author_affinities=model.author_affinities,
            avg_rating_given=model.avg_rating_given,
            total_borrows=model.total_borrows,
            total_reviews=model.total_reviews,
            taste_embedding=model.taste_embedding,
            taste_cluster=model.taste_cluster,
            confidence_score=model.confidence_score,
            last_computed_at=model.last_computed_at,
        )
