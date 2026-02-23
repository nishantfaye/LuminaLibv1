"""User Preference Service — orchestrates all three preference layers.

Layer 1 (Explicit):  CRUD on user-stated preferences (authors, genres, tags …).
Layer 2 (Implicit):  Records & queries the interaction event log.
Layer 3 (Computed):  Recomputes the materialized taste profile from L1 + L2.

Design note
-----------
The requirement is *intentionally vague* — "store User Preferences."  Our
design choice is a **three-layer model** that separates *what the user says*
from *what we observe* from *what the AI derives*.  This gives the
recommendation engine three complementary signals rather than a single flat
row, and makes the system extensible as new interaction types emerge.
"""

import logging
from collections import Counter
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

import numpy as np

from app.domain.entities import UserInteraction, UserPreference, UserTasteProfile
from app.domain.repositories import (
    IBookRepository,
    IBorrowRepository,
    ILLMService,
    IReviewRepository,
    IUserInteractionRepository,
    IUserPreferenceRepository,
    IUserTasteProfileRepository,
)
from app.domain.services import IPreferenceService

logger = logging.getLogger(__name__)

# Interaction weights — how strongly each event type signals preference
INTERACTION_WEIGHTS = {
    "borrow": 1.0,
    "return": 0.3,  # completing = mild positive signal
    "review": 1.5,  # reviewing = strong signal (especially with rating)
    "bookmark": 0.8,
    "view": 0.2,
}


class PreferenceService(IPreferenceService):
    """Orchestrates all three layers of user preferences.

    Responsibilities:
      • CRUD for explicit preferences (Layer 1)
      • Recording and querying implicit interactions (Layer 2)
      • Recomputing the AI-derived taste profile (Layer 3)
      • Providing a unified "preference snapshot" for the recommendation engine
    """

    def __init__(
        self,
        preference_repo: IUserPreferenceRepository,
        interaction_repo: IUserInteractionRepository,
        taste_repo: IUserTasteProfileRepository,
        book_repo: IBookRepository,
        borrow_repo: IBorrowRepository,
        review_repo: IReviewRepository,
        llm_service: ILLMService,
    ):
        self.preference_repo = preference_repo
        self.interaction_repo = interaction_repo
        self.taste_repo = taste_repo
        self.book_repo = book_repo
        self.borrow_repo = borrow_repo
        self.review_repo = review_repo
        self.llm_service = llm_service

    # -----------------------------------------------------------------------
    # Layer 1 — Explicit Preferences
    # -----------------------------------------------------------------------

    async def get_preferences(self, user_id: UUID) -> UserPreference:
        """Return (or lazily create) the user's explicit preferences."""
        return await self.preference_repo.get_or_create(user_id)

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
        """Merge-update explicit preference fields (only supplied fields change)."""
        pref = await self.preference_repo.get_or_create(user_id)

        if preferred_authors is not None:
            pref.preferred_authors = preferred_authors
        if preferred_genres is not None:
            pref.preferred_genres = preferred_genres
        if preferred_tags is not None:
            pref.preferred_tags = preferred_tags
        if reading_pace is not None:
            if reading_pace not in ("casual", "moderate", "avid"):
                raise ValueError("reading_pace must be one of: casual, moderate, avid")
            pref.reading_pace = reading_pace
        if discovery_mode is not None:
            if discovery_mode not in ("similar", "balanced", "exploratory"):
                raise ValueError("discovery_mode must be one of: similar, balanced, exploratory")
            pref.discovery_mode = discovery_mode
        if content_preferences is not None:
            pref.content_preferences = content_preferences
        if notify_new_by_favorite_author is not None:
            pref.notify_new_by_favorite_author = notify_new_by_favorite_author

        return await self.preference_repo.update(pref)

    # -----------------------------------------------------------------------
    # Layer 2 — Implicit Interactions
    # -----------------------------------------------------------------------

    async def record_interaction(
        self,
        user_id: UUID,
        book_id: UUID,
        interaction_type: str,
        interaction_data: Optional[dict] = None,
    ) -> UserInteraction:
        """Record an implicit interaction event.

        Called automatically by borrow/return/review endpoints (or manually
        for bookmarks/views).
        """
        weight = INTERACTION_WEIGHTS.get(interaction_type, 0.5)

        # Boost weight for high-rating reviews
        if interaction_type == "review" and interaction_data:
            rating = interaction_data.get("rating", 3)
            if rating >= 4:
                weight *= 1.3  # strong positive signal
            elif rating <= 2:
                weight *= 0.5  # negative signal (less influence)

        interaction = UserInteraction(
            id=uuid4(),
            user_id=user_id,
            book_id=book_id,
            interaction_type=interaction_type,
            interaction_data=interaction_data,
            weight=weight,
        )
        recorded = await self.interaction_repo.record(interaction)
        logger.info(
            "Recorded %s interaction for user %s on book %s (weight=%.2f)",
            interaction_type, user_id, book_id, weight,
        )
        return recorded

    async def get_interaction_history(
        self,
        user_id: UUID,
        interaction_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[UserInteraction]:
        """Get a user's interaction history, optionally filtered by type."""
        return await self.interaction_repo.get_user_interactions(
            user_id, interaction_type=interaction_type, limit=limit,
        )

    async def get_interaction_stats(self, user_id: UUID) -> dict:
        """Get aggregated interaction statistics for a user."""
        return await self.interaction_repo.get_interaction_stats(user_id)

    # -----------------------------------------------------------------------
    # Layer 3 — Computed Taste Profile
    # -----------------------------------------------------------------------

    async def recompute_taste_profile(self, user_id: UUID) -> UserTasteProfile:
        """Recompute the user's AI-derived taste profile from all available data.

        This aggregates:
          1. Borrow history → author affinity, genre signals
          2. Review ratings → weighted book embeddings, avg rating
          3. Interaction log → recency-weighted interest signals
          4. Explicit preferences → genre/author boosts
          5. LLM → taste cluster label
        """
        logger.info("Recomputing taste profile for user %s", user_id)

        profile = await self.taste_repo.get_or_create(user_id)
        pref = await self.preference_repo.get_or_create(user_id)

        # --- Gather all data ---
        borrows = await self.borrow_repo.get_user_borrows(user_id)
        borrowed_book_ids = {b.book_id for b in borrows}
        all_books = await self.book_repo.list_all(limit=1000)
        books_by_id = {b.id: b for b in all_books}

        # --- Author affinities ---
        author_scores: dict[str, float] = {}
        for bid in borrowed_book_ids:
            book = books_by_id.get(bid)
            if book:
                author_scores[book.author] = author_scores.get(book.author, 0) + 1.0

        # Boost from explicit preferences
        for author in pref.preferred_authors:
            author_scores[author] = author_scores.get(author, 0) + 2.0

        # Normalise to 0–1
        if author_scores:
            max_score = max(author_scores.values())
            author_scores = {a: round(s / max_score, 3) for a, s in author_scores.items()}

        # --- Genre affinities (from explicit + tags) ---
        genre_scores: dict[str, float] = {}
        for genre in pref.preferred_genres:
            genre_scores[genre] = genre_scores.get(genre, 0) + 2.0
        for tag in pref.preferred_tags:
            genre_scores[tag] = genre_scores.get(tag, 0) + 1.0

        # Derive genre hints from book titles/authors via interactions
        interactions = await self.interaction_repo.get_user_interactions(user_id, limit=500)
        for ix in interactions:
            book = books_by_id.get(ix.book_id)
            if book and ix.interaction_data:
                rating = ix.interaction_data.get("rating")
                if rating and rating >= 4:
                    # Use author as a pseudo-genre signal
                    genre_scores[book.author] = genre_scores.get(book.author, 0) + ix.weight

        if genre_scores:
            max_g = max(genre_scores.values())
            genre_scores = {g: round(s / max_g, 3) for g, s in genre_scores.items()}

        # --- Stats ---
        stats = await self.interaction_repo.get_interaction_stats(user_id)
        total_borrows = stats.get("total_borrows", len(borrows))
        total_reviews = stats.get("total_reviews", 0)
        avg_rating = stats.get("avg_rating_given", 0.0)

        # --- Taste embedding (weighted mean of borrowed book embeddings) ---
        taste_embedding = None
        borrowed_books_with_emb = [
            books_by_id[bid]
            for bid in borrowed_book_ids
            if bid in books_by_id and books_by_id[bid].embedding
        ]
        if borrowed_books_with_emb:
            # Group by dimension, use the most common one
            dim_counts = Counter(len(b.embedding) for b in borrowed_books_with_emb)
            target_dim = dim_counts.most_common(1)[0][0]

            weighted_embs = []
            total_weight = 0.0
            for book in borrowed_books_with_emb:
                if len(book.embedding) != target_dim:
                    continue
                # Weight by review rating if available
                weight = 1.0
                for ix in interactions:
                    if ix.book_id == book.id and ix.interaction_type == "review":
                        r = (ix.interaction_data or {}).get("rating", 3)
                        weight = r / 3.0  # normalize around 1.0
                        break
                weighted_embs.append(np.array(book.embedding) * weight)
                total_weight += weight

            if weighted_embs:
                taste_emb_arr = np.sum(weighted_embs, axis=0) / total_weight
                # Normalise to unit vector
                norm = np.linalg.norm(taste_emb_arr)
                if norm > 0:
                    taste_emb_arr = taste_emb_arr / norm
                taste_embedding = taste_emb_arr.tolist()

        # --- Confidence score (0–1 based on data volume) ---
        # More data = more confidence, logistic-style saturation
        data_points = total_borrows + total_reviews * 2
        confidence = min(1.0, round(data_points / 20.0, 3))  # saturates at ~20 actions

        # --- Taste cluster (LLM-generated label) ---
        taste_cluster = None
        if confidence >= 0.3 and (author_scores or genre_scores):
            try:
                top_authors = sorted(author_scores.items(), key=lambda x: -x[1])[:5]
                top_genres = sorted(genre_scores.items(), key=lambda x: -x[1])[:5]
                taste_cluster = await self.llm_service.generate_taste_cluster_label(
                    top_authors=[a for a, _ in top_authors],
                    top_genres=[g for g, _ in top_genres],
                    avg_rating=avg_rating,
                    total_borrows=total_borrows,
                )
                taste_cluster = taste_cluster.strip().strip('"\'')[:100]
            except Exception as exc:
                logger.warning("Failed to generate taste cluster: %s", exc)

        # --- Persist ---
        profile.genre_affinities = genre_scores or None
        profile.author_affinities = author_scores or None
        profile.avg_rating_given = avg_rating
        profile.total_borrows = total_borrows
        profile.total_reviews = total_reviews
        profile.taste_embedding = taste_embedding
        profile.taste_cluster = taste_cluster
        profile.confidence_score = confidence
        profile.last_computed_at = datetime.utcnow()

        updated = await self.taste_repo.update(profile)
        logger.info(
            "Taste profile recomputed for user %s — confidence=%.2f, cluster=%s",
            user_id, confidence, taste_cluster,
        )
        return updated

    async def get_taste_profile(self, user_id: UUID) -> UserTasteProfile:
        """Return the user's computed taste profile (without recomputing)."""
        return await self.taste_repo.get_or_create(user_id)

    # -----------------------------------------------------------------------
    # Unified Preference Snapshot (for the recommendation engine)
    # -----------------------------------------------------------------------

    async def get_preference_snapshot(self, user_id: UUID) -> dict:
        """Return a unified dict combining all three preference layers.

        The recommendation engine calls this to get a single view of the
        user's preferences without needing to know about the layer internals.
        """
        pref = await self.preference_repo.get_or_create(user_id)
        profile = await self.taste_repo.get_or_create(user_id)
        stats = await self.interaction_repo.get_interaction_stats(user_id)

        return {
            "user_id": str(user_id),
            # Layer 1 — Explicit
            "explicit": {
                "preferred_authors": pref.preferred_authors,
                "preferred_genres": pref.preferred_genres,
                "preferred_tags": pref.preferred_tags,
                "reading_pace": pref.reading_pace,
                "discovery_mode": pref.discovery_mode,
                "content_preferences": pref.content_preferences,
            },
            # Layer 2 — Implicit (aggregated)
            "implicit": stats,
            # Layer 3 — Computed
            "computed": {
                "genre_affinities": profile.genre_affinities,
                "author_affinities": profile.author_affinities,
                "avg_rating_given": profile.avg_rating_given,
                "total_borrows": profile.total_borrows,
                "total_reviews": profile.total_reviews,
                "taste_cluster": profile.taste_cluster,
                "confidence_score": profile.confidence_score,
                "has_taste_embedding": profile.taste_embedding is not None,
                "last_computed_at": (
                    profile.last_computed_at.isoformat() if profile.last_computed_at else None
                ),
            },
        }
