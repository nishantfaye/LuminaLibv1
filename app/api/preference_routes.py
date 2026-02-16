"""User Preference API routes.

Exposes all three layers of the preference system:
  GET/PUT  /preferences             — Layer 1: Explicit preferences
  POST/GET /preferences/interactions — Layer 2: Implicit interactions
  GET/POST /preferences/taste        — Layer 3: Computed taste profile
  GET      /preferences/snapshot     — Unified view of all three layers
"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import (
    InteractionCreateRequest,
    InteractionResponse,
    InteractionStatsResponse,
    PreferenceSnapshotResponse,
    TasteProfileResponse,
    UserPreferenceResponse,
    UserPreferenceUpdateRequest,
)
from app.core.dependencies import get_current_user, get_preference_service
from app.domain.entities import User
from app.services.preference_service import PreferenceService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/preferences", tags=["preferences"])


# ---------------------------------------------------------------------------
# Layer 1 — Explicit Preferences
# ---------------------------------------------------------------------------


@router.get("/", response_model=UserPreferenceResponse)
async def get_preferences(
    current_user: Annotated[User, Depends(get_current_user)],
    pref_service: Annotated[PreferenceService, Depends(get_preference_service)],
) -> UserPreferenceResponse:
    """Get the authenticated user's explicit preferences."""
    pref = await pref_service.get_preferences(current_user.id)
    return UserPreferenceResponse(
        user_id=pref.user_id,
        preferred_authors=pref.preferred_authors,
        preferred_genres=pref.preferred_genres,
        preferred_tags=pref.preferred_tags,
        reading_pace=pref.reading_pace,
        discovery_mode=pref.discovery_mode,
        content_preferences=pref.content_preferences,
        notify_new_by_favorite_author=pref.notify_new_by_favorite_author,
        updated_at=pref.updated_at,
    )


@router.put("/", response_model=UserPreferenceResponse)
async def update_preferences(
    body: UserPreferenceUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    pref_service: Annotated[PreferenceService, Depends(get_preference_service)],
) -> UserPreferenceResponse:
    """Update (merge) the authenticated user's explicit preferences.

    Only fields present in the request body are changed; others are preserved.
    """
    try:
        pref = await pref_service.update_preferences(
            current_user.id,
            preferred_authors=body.preferred_authors,
            preferred_genres=body.preferred_genres,
            preferred_tags=body.preferred_tags,
            reading_pace=body.reading_pace,
            discovery_mode=body.discovery_mode,
            content_preferences=body.content_preferences,
            notify_new_by_favorite_author=body.notify_new_by_favorite_author,
        )
        return UserPreferenceResponse(
            user_id=pref.user_id,
            preferred_authors=pref.preferred_authors,
            preferred_genres=pref.preferred_genres,
            preferred_tags=pref.preferred_tags,
            reading_pace=pref.reading_pace,
            discovery_mode=pref.discovery_mode,
            content_preferences=pref.content_preferences,
            notify_new_by_favorite_author=pref.notify_new_by_favorite_author,
            updated_at=pref.updated_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# ---------------------------------------------------------------------------
# Layer 2 — Implicit Interactions
# ---------------------------------------------------------------------------


@router.post(
    "/interactions",
    response_model=InteractionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def record_interaction(
    body: InteractionCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    pref_service: Annotated[PreferenceService, Depends(get_preference_service)],
) -> InteractionResponse:
    """Manually record a bookmark or view interaction.

    Borrow, return, and review interactions are recorded automatically by
    their respective endpoints.
    """
    interaction = await pref_service.record_interaction(
        user_id=current_user.id,
        book_id=body.book_id,
        interaction_type=body.interaction_type,
    )
    return InteractionResponse.model_validate(interaction)


@router.get("/interactions", response_model=list[InteractionResponse])
async def get_interactions(
    current_user: Annotated[User, Depends(get_current_user)],
    pref_service: Annotated[PreferenceService, Depends(get_preference_service)],
    interaction_type: Optional[str] = None,
    limit: int = 50,
) -> list[InteractionResponse]:
    """Get the authenticated user's interaction history."""
    interactions = await pref_service.get_interaction_history(
        current_user.id,
        interaction_type=interaction_type,
        limit=limit,
    )
    return [InteractionResponse.model_validate(i) for i in interactions]


@router.get("/interactions/stats", response_model=InteractionStatsResponse)
async def get_interaction_stats(
    current_user: Annotated[User, Depends(get_current_user)],
    pref_service: Annotated[PreferenceService, Depends(get_preference_service)],
) -> InteractionStatsResponse:
    """Get aggregated interaction statistics."""
    stats = await pref_service.get_interaction_stats(current_user.id)
    return InteractionStatsResponse(**stats)


# ---------------------------------------------------------------------------
# Layer 3 — Computed Taste Profile
# ---------------------------------------------------------------------------


@router.get("/taste", response_model=TasteProfileResponse)
async def get_taste_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    pref_service: Annotated[PreferenceService, Depends(get_preference_service)],
) -> TasteProfileResponse:
    """Get the user's current computed taste profile (without recomputing)."""
    profile = await pref_service.get_taste_profile(current_user.id)
    return TasteProfileResponse(
        user_id=profile.user_id,
        genre_affinities=profile.genre_affinities,
        author_affinities=profile.author_affinities,
        avg_rating_given=profile.avg_rating_given,
        total_borrows=profile.total_borrows,
        total_reviews=profile.total_reviews,
        taste_cluster=profile.taste_cluster,
        confidence_score=profile.confidence_score,
        last_computed_at=profile.last_computed_at,
    )


@router.post("/taste/recompute", response_model=TasteProfileResponse)
async def recompute_taste_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    pref_service: Annotated[PreferenceService, Depends(get_preference_service)],
) -> TasteProfileResponse:
    """Recompute the user's taste profile from all available data.

    This aggregates explicit preferences, implicit interactions, borrow
    history, review ratings, and uses the LLM to generate a taste cluster
    label.
    """
    profile = await pref_service.recompute_taste_profile(current_user.id)
    return TasteProfileResponse(
        user_id=profile.user_id,
        genre_affinities=profile.genre_affinities,
        author_affinities=profile.author_affinities,
        avg_rating_given=profile.avg_rating_given,
        total_borrows=profile.total_borrows,
        total_reviews=profile.total_reviews,
        taste_cluster=profile.taste_cluster,
        confidence_score=profile.confidence_score,
        last_computed_at=profile.last_computed_at,
    )


# ---------------------------------------------------------------------------
# Unified Snapshot
# ---------------------------------------------------------------------------


@router.get("/snapshot", response_model=PreferenceSnapshotResponse)
async def get_preference_snapshot(
    current_user: Annotated[User, Depends(get_current_user)],
    pref_service: Annotated[PreferenceService, Depends(get_preference_service)],
) -> PreferenceSnapshotResponse:
    """Get a unified view combining all three preference layers.

    Designed for the recommendation engine and for front-end profile pages.
    """
    snapshot = await pref_service.get_preference_snapshot(current_user.id)
    return PreferenceSnapshotResponse(**snapshot)
