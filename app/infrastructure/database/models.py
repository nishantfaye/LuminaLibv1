"""SQLAlchemy database models."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class UserModel(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    borrows = relationship("BorrowModel", back_populates="user", lazy="selectin")
    reviews = relationship("ReviewModel", back_populates="user", lazy="selectin")
    preferences = relationship(
        "UserPreferenceModel", back_populates="user", uselist=False, lazy="selectin"
    )
    interactions = relationship("UserInteractionModel", back_populates="user", lazy="dynamic")
    taste_profile = relationship(
        "UserTasteProfileModel", back_populates="user", uselist=False, lazy="selectin"
    )


class BookModel(Base):
    __tablename__ = "books"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False, index=True)
    author = Column(String(255), nullable=False, index=True)
    genre = Column(String(100), nullable=False, default="general", server_default="general", index=True)
    file_path = Column(String(512), nullable=False, unique=True)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    content_hash = Column(String(64), nullable=False, unique=True)
    summary = Column(Text, nullable=True)
    embedding = Column(ARRAY(Float), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    borrows = relationship("BorrowModel", back_populates="book", lazy="selectin", cascade="all, delete-orphan")
    reviews = relationship("ReviewModel", back_populates="book", lazy="selectin", cascade="all, delete-orphan")
    interactions = relationship(
        "UserInteractionModel", back_populates="book", lazy="dynamic", cascade="all, delete-orphan"
    )
    analysis = relationship(
        "BookAnalysisModel", back_populates="book", uselist=False, lazy="selectin", cascade="all, delete-orphan"
    )


class BorrowModel(Base):
    __tablename__ = "borrows"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    book_id = Column(UUID(as_uuid=True), ForeignKey("books.id"), nullable=False, index=True)
    borrowed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    returned_at = Column(DateTime, nullable=True)

    user = relationship("UserModel", back_populates="borrows")
    book = relationship("BookModel", back_populates="borrows")


class ReviewModel(Base):
    __tablename__ = "reviews"
    __table_args__ = (UniqueConstraint("user_id", "book_id", name="uq_user_book_review"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    book_id = Column(UUID(as_uuid=True), ForeignKey("books.id"), nullable=False, index=True)
    rating = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    sentiment = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("UserModel", back_populates="reviews")
    book = relationship("BookModel", back_populates="reviews")


class UserPreferenceModel(Base):
    """User preference database model — Layer 1: Explicit Preferences.

    Stores what the user *explicitly tells us* they prefer.
    """

    __tablename__ = "user_preferences"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True)
    preferred_authors = Column(ARRAY(String), default=list, nullable=False)
    preferred_genres = Column(ARRAY(String), default=list, nullable=False)
    preferred_tags = Column(ARRAY(String), default=list, nullable=False)
    reading_pace = Column(String(20), default="moderate", nullable=False)  # casual|moderate|avid
    discovery_mode = Column(String(20), default="balanced", nullable=False)  # similar|balanced|exploratory
    content_preferences = Column(JSON, nullable=True)  # extensible JSONB key-value
    notify_new_by_favorite_author = Column(Boolean, default=False, nullable=False)
    embedding = Column(ARRAY(Float), nullable=True)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    user = relationship("UserModel", back_populates="preferences")


class UserInteractionModel(Base):
    """User interaction database model — Layer 2: Implicit Signals.

    An event-sourced log of every meaningful user↔book interaction.
    Borrows, returns, reviews, bookmarks, views — each recorded with
    contextual data (rating, sentiment, duration) and a weight multiplier.
    """

    __tablename__ = "user_interactions"
    __table_args__ = (
        Index("ix_interactions_user_type", "user_id", "interaction_type"),
        Index("ix_interactions_book", "book_id"),
        Index("ix_interactions_created", "created_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    book_id = Column(UUID(as_uuid=True), ForeignKey("books.id"), nullable=False)
    interaction_type = Column(String(30), nullable=False)  # borrow|return|review|bookmark|view
    interaction_data = Column(JSON, nullable=True)  # {rating, sentiment, duration_days …}
    weight = Column(Float, default=1.0, nullable=False)  # signal strength multiplier
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("UserModel", back_populates="interactions")
    book = relationship("BookModel")


class UserTasteProfileModel(Base):
    """User taste profile database model — Layer 3: Computed/Materialized.

    A periodically recomputed AI-derived taste profile built from
    explicit preferences (Layer 1) and implicit signals (Layer 2).
    Contains genre/author affinity scores, a taste embedding vector,
    and an LLM-generated cluster label.
    """

    __tablename__ = "user_taste_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True)
    genre_affinities = Column(JSON, nullable=True)  # {"fiction": 0.8, "science": 0.6}
    author_affinities = Column(JSON, nullable=True)  # {"Orwell": 0.9, "Tolkien": 0.7}
    avg_rating_given = Column(Float, default=0.0, nullable=False)
    total_borrows = Column(Integer, default=0, nullable=False)
    total_reviews = Column(Integer, default=0, nullable=False)
    taste_embedding = Column(ARRAY(Float), nullable=True)  # weighted mean of book embeddings
    taste_cluster = Column(String(100), nullable=True)  # LLM label: "literary fiction enthusiast"
    confidence_score = Column(Float, default=0.0, nullable=False)  # 0–1
    last_computed_at = Column(DateTime, nullable=True)

    user = relationship("UserModel", back_populates="taste_profile")


class BookAnalysisModel(Base):
    __tablename__ = "book_analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    book_id = Column(UUID(as_uuid=True), ForeignKey("books.id"), nullable=False, unique=True)
    review_count = Column(Integer, default=0, nullable=False)
    average_rating = Column(Float, default=0.0, nullable=False)
    consensus_summary = Column(Text, default="", nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    book = relationship("BookModel", back_populates="analysis")
