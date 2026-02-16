"""Domain entities for LuminaLib."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID


@dataclass
class User:
    id: UUID
    username: str
    email: str
    hashed_password: str
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Book:
    id: UUID
    title: str
    author: str
    genre: str
    file_path: str
    file_size: int
    mime_type: str
    content_hash: str
    summary: Optional[str] = None
    embedding: Optional[list[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BorrowRecord:
    id: UUID
    user_id: UUID
    book_id: UUID
    borrowed_at: datetime = field(default_factory=datetime.utcnow)
    returned_at: Optional[datetime] = None


@dataclass
class Review:
    id: UUID
    user_id: UUID
    book_id: UUID
    rating: int
    text: str
    sentiment: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserPreference:
    """User preference entity — explicit preferences stated by the user.

    Layer 1 of the three-layer preference model:
      Layer 1 (Explicit)  → this table: what the user *tells* us they like.
      Layer 2 (Implicit)  → UserInteraction: what we *observe* them doing.
      Layer 3 (Computed)  → UserTasteProfile: AI-derived composite of 1 + 2.
    """

    id: UUID
    user_id: UUID
    preferred_authors: list[str] = field(default_factory=list)
    preferred_genres: list[str] = field(default_factory=list)
    preferred_tags: list[str] = field(default_factory=list)
    reading_pace: str = "moderate"  # casual | moderate | avid
    discovery_mode: str = "balanced"  # similar | balanced | exploratory
    content_preferences: Optional[dict] = None  # JSONB for extensible k/v
    notify_new_by_favorite_author: bool = False
    embedding: Optional[list[float]] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserInteraction:
    """Implicit signal — records every meaningful user↔book interaction.

    Layer 2 of the three-layer preference model.
    An event-sourced log of borrows, returns, reviews, bookmarks, etc.
    The preference service aggregates this log to compute taste profiles.
    """

    id: UUID
    user_id: UUID
    book_id: UUID
    interaction_type: str  # borrow | return | review | bookmark | view
    interaction_data: Optional[dict] = None  # {rating, sentiment, duration_days …}
    weight: float = 1.0  # signal strength multiplier
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserTasteProfile:
    """AI-computed taste profile — materialized from explicit + implicit data.

    Layer 3 of the three-layer preference model.
    Periodically recomputed by the PreferenceService.
    """

    id: UUID
    user_id: UUID
    genre_affinities: Optional[dict] = None  # {"fiction": 0.8, "science": 0.6}
    author_affinities: Optional[dict] = None  # {"Orwell": 0.9, "Tolkien": 0.7}
    avg_rating_given: float = 0.0
    total_borrows: int = 0
    total_reviews: int = 0
    taste_embedding: Optional[list[float]] = None  # weighted mean of rated book embeddings
    taste_cluster: Optional[str] = None  # LLM-generated label e.g. "literary fiction enthusiast"
    confidence_score: float = 0.0  # 0–1, how much data backs this profile
    last_computed_at: Optional[datetime] = None


@dataclass
class BookAnalysis:
    """Aggregated GenAI analysis of all reviews for a book."""

    book_id: UUID
    review_count: int
    average_rating: float
    consensus_summary: str
    updated_at: datetime = field(default_factory=datetime.utcnow)
