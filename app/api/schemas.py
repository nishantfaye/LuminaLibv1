"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: UUID
    username: str
    email: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProfileUpdateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    email: EmailStr


# ---------------------------------------------------------------------------
# Books
# ---------------------------------------------------------------------------
class BookCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    author: str = Field(..., min_length=1, max_length=255)


class BookUpdate(BaseModel):
    """Book update request (metadata only)."""

    title: str = Field(..., min_length=1, max_length=255)
    author: str = Field(..., min_length=1, max_length=255)
    genre: str = Field("general", min_length=1, max_length=100)


class BookResponse(BaseModel):
    id: UUID
    title: str
    author: str
    genre: str
    file_path: str
    file_size: int
    mime_type: str
    content_hash: str
    summary: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BookListResponse(BaseModel):
    books: list[BookResponse]
    total: int
    page: int
    limit: int


# ---------------------------------------------------------------------------
# Borrow
# ---------------------------------------------------------------------------
class BorrowResponse(BaseModel):
    id: UUID
    user_id: UUID
    book_id: UUID
    borrowed_at: datetime
    returned_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------
class ReviewCreateRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    text: str = Field(..., min_length=1)


class ReviewResponse(BaseModel):
    id: UUID
    user_id: UUID
    book_id: UUID
    rating: int
    text: str
    sentiment: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
class BookAnalysisResponse(BaseModel):
    book_id: str
    book_title: str
    review_count: int
    average_rating: float
    consensus_summary: str


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
class RecommendedBookResponse(BookResponse):
    """A recommended book with similarity/relevance score and explanation."""

    score: Optional[float] = Field(None, description="Blended relevance score (0.0 to 1.0+)")
    explanation: Optional[str] = Field(
        None, description="Human-readable reason for this recommendation"
    )
    strategies_used: Optional[list[str]] = Field(
        None,
        description="List of strategies that contributed (e.g. content_based, collaborative, popularity, knowledge_graph)",
    )


class RecommendationResponse(BaseModel):
    """Recommendation response with metadata."""

    recommendations: list[RecommendedBookResponse]
    total: int = Field(0, description="Number of recommendations returned")
    strategy: str = Field(
        "hybrid",
        description="Algorithm used: hybrid, content-based, popularity, recency_fallback",
    )


# ---------------------------------------------------------------------------
# User Preferences (Layer 1 — Explicit)
# ---------------------------------------------------------------------------
class UserPreferenceUpdateRequest(BaseModel):
    """Partial-update request for explicit user preferences."""

    preferred_authors: Optional[list[str]] = None
    preferred_genres: Optional[list[str]] = None
    preferred_tags: Optional[list[str]] = None
    reading_pace: Optional[str] = Field(
        None, pattern="^(casual|moderate|avid)$",
        description="casual | moderate | avid",
    )
    discovery_mode: Optional[str] = Field(
        None, pattern="^(similar|balanced|exploratory)$",
        description="similar | balanced | exploratory",
    )
    content_preferences: Optional[dict] = None
    notify_new_by_favorite_author: Optional[bool] = None


class UserPreferenceResponse(BaseModel):
    """Full explicit preference record."""

    user_id: UUID
    preferred_authors: list[str] = []
    preferred_genres: list[str] = []
    preferred_tags: list[str] = []
    reading_pace: str = "moderate"
    discovery_mode: str = "balanced"
    content_preferences: Optional[dict] = None
    notify_new_by_favorite_author: bool = False
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# User Interactions (Layer 2 — Implicit Signals)
# ---------------------------------------------------------------------------
class InteractionCreateRequest(BaseModel):
    """Record an explicit interaction (bookmark/view). Borrow/return/review
    interactions are recorded automatically."""

    book_id: UUID
    interaction_type: str = Field(
        ..., pattern="^(bookmark|view)$",
        description="bookmark | view (borrow/return/review are auto-recorded)",
    )


class InteractionResponse(BaseModel):
    id: UUID
    user_id: UUID
    book_id: UUID
    interaction_type: str
    interaction_data: Optional[dict] = None
    weight: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class InteractionStatsResponse(BaseModel):
    total_borrows: int = 0
    total_returns: int = 0
    total_reviews: int = 0
    total_bookmarks: int = 0
    avg_rating_given: float = 0.0
    interaction_counts: dict = {}


# ---------------------------------------------------------------------------
# Taste Profile (Layer 3 — Computed)
# ---------------------------------------------------------------------------
class TasteProfileResponse(BaseModel):
    user_id: UUID
    genre_affinities: Optional[dict] = None
    author_affinities: Optional[dict] = None
    avg_rating_given: float = 0.0
    total_borrows: int = 0
    total_reviews: int = 0
    taste_cluster: Optional[str] = None
    confidence_score: float = 0.0
    last_computed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Unified Preference Snapshot
# ---------------------------------------------------------------------------
class PreferenceSnapshotResponse(BaseModel):
    """Combines all three preference layers into one view."""

    user_id: str
    explicit: dict
    implicit: dict
    computed: dict
