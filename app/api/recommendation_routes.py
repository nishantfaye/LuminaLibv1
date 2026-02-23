"""Recommendation API routes."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.schemas import BookResponse, RecommendationResponse, RecommendedBookResponse
from app.core.dependencies import get_current_user, get_recommendation_service
from app.domain.entities import User
from app.domain.repositories import IRecommendationService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["recommendations"])


@router.get("/recommendations", response_model=RecommendationResponse)
async def get_user_recommendations(
    current_user: Annotated[User, Depends(get_current_user)],
    recommendation_service: Annotated[IRecommendationService, Depends(get_recommendation_service)],
    limit: int = 10,
) -> RecommendationResponse:
    """Get ML-based personalised book suggestions for the current user.

    The hybrid engine blends up to four strategies:
      - content_based (embedding cosine similarity + preference boosts)
      - collaborative (user-user Jaccard + item-item co-occurrence)
      - popularity (time-decayed borrows, Wilson-score ratings, trending)
      - knowledge_graph (author co-read graph, tag matching)

    Each recommendation includes a human-readable *explanation* and the list
    of strategies that contributed to it.
    """
    results = await recommendation_service.get_user_recommendations(current_user.id, limit)

    # Determine overall strategy label from per-recommendation metadata
    all_strategies: set[str] = set()
    recs: list[RecommendedBookResponse] = []
    for book, score in results:
        meta = getattr(book, "_rec_meta", None) or {}
        explanation = meta.get("explanation")
        strategies_used = meta.get("strategies_used")
        if strategies_used:
            all_strategies.update(strategies_used)
        rec = RecommendedBookResponse(
            **BookResponse.model_validate(book).model_dump(),
            score=score,
            explanation=explanation,
            strategies_used=strategies_used,
        )
        recs.append(rec)

    if all_strategies:
        strategy = "hybrid:" + "+".join(sorted(all_strategies))
    elif not results:
        strategy = "cold-start"
    else:
        strategy = "hybrid"

    return RecommendationResponse(
        recommendations=recs,
        total=len(recs),
        strategy=strategy,
    )
