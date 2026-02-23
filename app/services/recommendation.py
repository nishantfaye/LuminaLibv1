"""Multi-strategy recommendation engine for LuminaLib.

Implements five complementary algorithms orchestrated by a hybrid engine
that selects and blends strategies based on available data:

  1. Content-Based Filtering
  2. Collaborative Filtering (user-user Jaccard + item-item overlap)
  3. Popularity & Trending (Wilson score + time-decay + trending detection)
  4. Knowledge-Graph Boosting (author co-read graph + tag matching)
  5. Hybrid Orchestrator (weighted ensemble, cold-start cascade, MMR diversity)
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.domain.entities import Book, BorrowRecord, Review
from app.domain.repositories import (
    IBookRepository,
    IBorrowRepository,
    IRecommendationService,
    IReviewRepository,
    IUserInteractionRepository,
    IUserPreferenceRepository,
    IUserTasteProfileRepository,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Data structures
# ======================================================================
@dataclass
class ScoredBook:
    """Internal representation of a scored recommendation candidate."""

    book: Book
    score: float = 0.0
    explanations: list[str] = field(default_factory=list)
    strategy_scores: dict[str, float] = field(default_factory=dict)

    def add(self, strategy: str, score: float, explanation: str) -> None:
        self.strategy_scores[strategy] = score
        if explanation:
            self.explanations.append(explanation)


# ======================================================================
# Strategy 1 -- Content-Based Filtering
# ======================================================================
class ContentBasedEngine:
    """Cosine similarity on book embeddings + preference-layer boosts."""

    async def score(
        self,
        candidates: list[Book],
        user_profile_emb,
        profile_dim,
        pref_authors: set[str],
        pref_genres: set[str],
        taste_author_affinities: dict[str, float],
        taste_genre_affinities: dict[str, float],
    ) -> dict[UUID, ScoredBook]:
        results: dict[UUID, ScoredBook] = {}
        if user_profile_emb is None or profile_dim is None:
            return results

        valid = [b for b in candidates if b.embedding and len(b.embedding) == profile_dim]
        if not valid:
            return results

        emb_matrix = np.array([b.embedding for b in valid])
        sims = cosine_similarity(user_profile_emb, emb_matrix)[0]

        for i, book in enumerate(valid):
            sb = ScoredBook(book=book)
            base_sim = float(sims[i])
            parts = [f"embedding similarity {base_sim:.2f}"]

            # Layer 1 -- explicit author boost
            if book.author.lower() in pref_authors:
                base_sim += 0.15
                parts.append("preferred author (+0.15)")

            # Layer 3 -- computed author affinity
            author_aff = taste_author_affinities.get(book.author, 0)
            if author_aff > 0:
                boost = author_aff * 0.10
                base_sim += boost
                parts.append(f"author affinity {author_aff:.2f} (+{boost:.2f})")

            # Layer 1 -- explicit genre boost
            for genre in pref_genres:
                if genre in book.title.lower() or genre in book.author.lower():
                    base_sim += 0.10
                    parts.append(f"genre match '{genre}' (+0.10)")
                    break

            # Layer 3 -- computed genre affinity
            for genre, aff in taste_genre_affinities.items():
                if genre.lower() in book.title.lower() or genre.lower() in book.author.lower():
                    boost = aff * 0.05
                    base_sim += boost
                    parts.append(f"genre affinity '{genre}' (+{boost:.2f})")
                    break

            sb.add("content_based", base_sim, "Content match: " + ", ".join(parts))
            results[book.id] = sb

        return results


# ======================================================================
# Strategy 2 -- Collaborative Filtering
# ======================================================================
class CollaborativeFilteringEngine:
    """User-user Jaccard + item-item overlap coefficient scoring."""

    async def score(
        self,
        target_user_id: UUID,
        candidates: list[Book],
        all_borrows: list[BorrowRecord],
        all_reviews: list[Review],
        user_borrowed_ids: set[UUID],
    ) -> dict[UUID, ScoredBook]:
        results: dict[UUID, ScoredBook] = {}
        candidate_ids = {b.id for b in candidates}
        candidate_map = {b.id: b for b in candidates}

        # Build user-item matrices
        user_books: dict[UUID, set[UUID]] = defaultdict(set)
        for br in all_borrows:
            user_books[br.user_id].add(br.book_id)

        user_ratings: dict[UUID, dict[UUID, int]] = defaultdict(dict)
        for rv in all_reviews:
            user_ratings[rv.user_id][rv.book_id] = rv.rating

        my_books = user_books.get(target_user_id, set())
        if not my_books:
            return results

        # -- User-user similarity (Jaccard on borrow sets) --
        user_similarities: list[tuple[UUID, float]] = []
        for other_uid, other_books in user_books.items():
            if other_uid == target_user_id or not other_books:
                continue
            intersection = len(my_books & other_books)
            union = len(my_books | other_books)
            if union == 0:
                continue
            jaccard = intersection / union
            if jaccard > 0:
                user_similarities.append((other_uid, jaccard))

        user_similarities.sort(key=lambda x: -x[1])
        top_neighbours = user_similarities[:20]

        cf_user_scores: dict[UUID, float] = defaultdict(float)
        cf_user_explain: dict[UUID, str] = {}
        for neighbour_uid, sim in top_neighbours:
            for bid in user_books[neighbour_uid]:
                if bid in user_borrowed_ids or bid not in candidate_ids:
                    continue
                cf_user_scores[bid] += sim
                rating = user_ratings.get(neighbour_uid, {}).get(bid)
                if rating and rating >= 4:
                    cf_user_scores[bid] += sim * 0.3
                    cf_user_explain[bid] = (
                        f"Users with similar tastes rated this highly (similarity {sim:.2f})"
                    )
                elif not cf_user_explain.get(bid):
                    cf_user_explain[bid] = (
                        f"Users with similar reading patterns borrowed this (similarity {sim:.2f})"
                    )

        # -- Item-item co-occurrence --
        book_users: dict[UUID, set[UUID]] = defaultdict(set)
        for br in all_borrows:
            book_users[br.book_id].add(br.user_id)

        co_scores: dict[UUID, float] = defaultdict(float)
        co_explain: dict[UUID, str] = {}
        for my_bid in my_books:
            my_readers = book_users.get(my_bid, set())
            if not my_readers:
                continue
            for cand_bid in candidate_ids:
                if cand_bid in user_borrowed_ids:
                    continue
                cand_readers = book_users.get(cand_bid, set())
                if not cand_readers:
                    continue
                overlap = len(my_readers & cand_readers)
                if overlap > 0:
                    coeff = overlap / min(len(my_readers), len(cand_readers))
                    co_scores[cand_bid] += coeff
                    co_explain[cand_bid] = (
                        f"Often borrowed alongside books you have read (co-occurrence {coeff:.2f})"
                    )

        # Merge user-user and item-item
        all_cf_ids = set(cf_user_scores.keys()) | set(co_scores.keys())
        for bid in all_cf_ids:
            book = candidate_map.get(bid)
            if not book:
                continue
            uu = cf_user_scores.get(bid, 0)
            ii = co_scores.get(bid, 0)
            combined = uu * 0.6 + ii * 0.4
            if combined <= 0:
                continue
            normalised = min(combined, 1.0)
            sb = results.get(bid, ScoredBook(book=book))
            expl = cf_user_explain.get(bid) or co_explain.get(bid, "")
            sb.add("collaborative", normalised, f"Collaborative: {expl}")
            results[bid] = sb

        return results


# ======================================================================
# Strategy 3 -- Popularity & Trending
# ======================================================================
class PopularityEngine:
    """Time-decayed popularity + Wilson-score rating confidence + trending."""

    @staticmethod
    def _wilson_lower_bound(pos: int, total: int, z: float = 1.96) -> float:
        if total == 0:
            return 0.0
        phat = pos / total
        denom = 1 + z * z / total
        centre = phat + z * z / (2 * total)
        spread = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
        return (centre - spread) / denom

    async def score(
        self,
        candidates: list[Book],
        all_borrows: list[BorrowRecord],
        all_reviews: list[Review],
        user_borrowed_ids: set[UUID],
    ) -> dict[UUID, ScoredBook]:
        results: dict[UUID, ScoredBook] = {}
        candidate_ids = {b.id for b in candidates}
        candidate_map = {b.id: b for b in candidates}
        now = datetime.utcnow()

        borrow_total: dict[UUID, int] = Counter()
        borrow_recent: dict[UUID, float] = defaultdict(float)
        for br in all_borrows:
            if br.book_id not in candidate_ids:
                continue
            borrow_total[br.book_id] += 1
            age_days = max((now - br.borrowed_at).total_seconds() / 86400, 0)
            decay = math.exp(-0.693 * age_days / 14.0)
            borrow_recent[br.book_id] += decay

        book_ratings: dict[UUID, list[int]] = defaultdict(list)
        for rv in all_reviews:
            if rv.book_id in candidate_ids:
                book_ratings[rv.book_id].append(rv.rating)

        wilson_scores: dict[UUID, float] = {}
        for bid, ratings in book_ratings.items():
            pos = sum(1 for r in ratings if r >= 4)
            wilson_scores[bid] = self._wilson_lower_bound(pos, len(ratings))

        trending: dict[UUID, float] = {}
        for bid in candidate_ids:
            recent = borrow_recent.get(bid, 0)
            total = borrow_total.get(bid, 0)
            if total > 0:
                book = candidate_map.get(bid)
                if book and book.created_at:
                    age_days = max((now - book.created_at).total_seconds() / 86400, 1)
                    lifetime_rate = total / (age_days / 30.0)
                    trending[bid] = recent / (lifetime_rate + 0.1)

        max_recent = max(borrow_recent.values()) if borrow_recent else 1.0
        max_total = max(borrow_total.values()) if borrow_total else 1.0

        for cand in candidates:
            if cand.id in user_borrowed_ids:
                continue
            bid = cand.id
            pop_score = 0.0
            parts = []

            total = borrow_total.get(bid, 0)
            if total > 0:
                pop_score += (total / max_total) * 0.35
                parts.append(f"borrowed {total} times")

            recent = borrow_recent.get(bid, 0)
            if recent > 0:
                pop_score += (recent / max_recent) * 0.30
                parts.append("recent borrowing activity")

            wilson = wilson_scores.get(bid, 0)
            if wilson > 0:
                pop_score += wilson * 0.25
                ratings = book_ratings.get(bid, [])
                avg_r = sum(ratings) / len(ratings) if ratings else 0
                parts.append(f"rated {avg_r:.1f}/5 ({len(ratings)} reviews)")

            trend = trending.get(bid, 0)
            if trend > 1.5:
                pop_score += 0.10
                parts.append("trending")

            if pop_score > 0:
                sb = ScoredBook(book=cand)
                explanation = "Popular: " + ", ".join(parts) if parts else ""
                sb.add("popularity", min(pop_score, 1.0), explanation)
                results[bid] = sb

        return results


# ======================================================================
# Strategy 4 -- Knowledge-Graph Boosting
# ======================================================================
class KnowledgeGraphEngine:
    """Author co-read graph + tag overlap + same-author boosting."""

    async def score(
        self,
        target_user_id: UUID,
        candidates: list[Book],
        all_borrows: list[BorrowRecord],
        user_borrowed_ids: set[UUID],
        all_books: list[Book],
        pref_authors: set[str],
        pref_tags: list[str],
    ) -> dict[UUID, ScoredBook]:
        results: dict[UUID, ScoredBook] = {}
        books_by_id = {b.id: b for b in all_books}

        # Build author co-read graph
        user_authors: dict[UUID, set[str]] = defaultdict(set)
        for br in all_borrows:
            book = books_by_id.get(br.book_id)
            if book:
                user_authors[br.user_id].add(book.author)

        author_adj: dict[str, Counter] = defaultdict(Counter)
        for uid, authors in user_authors.items():
            alist = list(authors)
            for i, a in enumerate(alist):
                for b in alist[i + 1:]:
                    author_adj[a][b] += 1
                    author_adj[b][a] += 1

        my_authors = user_authors.get(target_user_id, set()) | pref_authors

        for cand in candidates:
            if cand.id in user_borrowed_ids:
                continue
            kg_score = 0.0
            parts = []

            if cand.author in my_authors or cand.author.lower() in pref_authors:
                kg_score += 0.35
                parts.append(f"by {cand.author} (an author you know)")

            for my_author in my_authors:
                co_count = author_adj.get(my_author, {}).get(cand.author, 0)
                if co_count > 0:
                    boost = min(co_count * 0.08, 0.25)
                    kg_score += boost
                    parts.append(f"readers of {my_author} also read {cand.author}")
                    break

            for tag in pref_tags:
                tag_low = tag.lower()
                if tag_low in cand.title.lower() or tag_low in cand.author.lower():
                    kg_score += 0.10
                    parts.append(f"matches your interest in '{tag}'")
                    break

            if kg_score > 0:
                sb = ScoredBook(book=cand)
                explanation = "Knowledge graph: " + ", ".join(parts) if parts else ""
                sb.add("knowledge_graph", min(kg_score, 1.0), explanation)
                results[cand.id] = sb

        return results


# ======================================================================
# Hybrid Orchestrator (the public IRecommendationService)
# ======================================================================
class MLRecommendationService(IRecommendationService):
    """Hybrid recommendation engine that blends five strategies.

    Strategy weights by discovery mode:
      similar:      CB=0.50  CF=0.25  POP=0.05  KG=0.20
      balanced:     CB=0.35  CF=0.25  POP=0.15  KG=0.25
      exploratory:  CB=0.20  CF=0.20  POP=0.30  KG=0.30

    Cold-start cascade:
      1. User has borrows + embeddings  -> full hybrid
      2. User has borrows or prefs only -> CF + KG + popularity
      3. Nothing                        -> popularity only
    """

    STRATEGY_WEIGHTS = {
        "similar": {
            "content_based": 0.50,
            "collaborative": 0.25,
            "popularity": 0.05,
            "knowledge_graph": 0.20,
        },
        "balanced": {
            "content_based": 0.35,
            "collaborative": 0.25,
            "popularity": 0.15,
            "knowledge_graph": 0.25,
        },
        "exploratory": {
            "content_based": 0.20,
            "collaborative": 0.20,
            "popularity": 0.30,
            "knowledge_graph": 0.30,
        },
    }

    def __init__(
        self,
        book_repository: IBookRepository,
        preference_repository: IUserPreferenceRepository,
        borrow_repository: IBorrowRepository,
        review_repository: IReviewRepository,
        taste_repository: IUserTasteProfileRepository | None = None,
        interaction_repository: IUserInteractionRepository | None = None,
    ):
        self.book_repository = book_repository
        self.preference_repository = preference_repository
        self.borrow_repository = borrow_repository
        self.review_repository = review_repository
        self.taste_repository = taste_repository
        self.interaction_repository = interaction_repository
        self._content = ContentBasedEngine()
        self._collab = CollaborativeFilteringEngine()
        self._popularity = PopularityEngine()
        self._knowledge = KnowledgeGraphEngine()

    # --- Book-to-book recommendations (unchanged) ---
    async def get_recommendations(
        self, book_id: UUID, limit: int = 5
    ) -> list[tuple[Book, float]]:
        target = await self.book_repository.get_by_id(book_id)
        if not target or not target.embedding:
            return []
        target_dim = len(target.embedding)
        all_books = await self.book_repository.list_all(limit=1000)
        candidates = [
            b
            for b in all_books
            if b.embedding and b.id != book_id and len(b.embedding) == target_dim
        ]
        if not candidates:
            return []
        target_emb = np.array(target.embedding).reshape(1, -1)
        emb_matrix = np.array([b.embedding for b in candidates])
        sims = cosine_similarity(target_emb, emb_matrix)[0]
        top_idx = np.argsort(sims)[::-1][:limit]
        return [(candidates[i], round(float(sims[i]), 4)) for i in top_idx]

    # --- Personalised hybrid recommendations ---
    async def get_user_recommendations(
        self, user_id: UUID, limit: int = 10
    ) -> list[tuple[Book, float]]:
        logger.info("Hybrid recommendation engine starting for user %s", user_id)

        # -- Gather data --
        all_books = await self.book_repository.list_all(limit=1000)
        borrows = await self.borrow_repository.get_user_borrows(user_id)
        borrowed_ids = {b.book_id for b in borrows}
        candidates = [b for b in all_books if b.id not in borrowed_ids]
        if not candidates:
            return []

        all_borrows = await self.borrow_repository.get_all_borrows(limit=5000)
        all_reviews = await self.review_repository.get_all_reviews(limit=5000)

        # -- Load preference layers --
        pref = await self.preference_repository.get_or_create(user_id)
        discovery_mode = getattr(pref, "discovery_mode", "balanced") or "balanced"
        pref_authors = {a.lower() for a in (pref.preferred_authors or [])}
        pref_genres = {g.lower() for g in (pref.preferred_genres or [])}
        pref_tags = list(pref.preferred_tags or [])

        taste_author_aff: dict[str, float] = {}
        taste_genre_aff: dict[str, float] = {}
        user_profile_emb = None
        profile_dim = None

        if self.taste_repository:
            try:
                taste_profile = await self.taste_repository.get_or_create(user_id)
                taste_author_aff = taste_profile.author_affinities or {}
                taste_genre_aff = taste_profile.genre_affinities or {}
                if taste_profile.taste_embedding:
                    user_profile_emb = np.array(
                        taste_profile.taste_embedding
                    ).reshape(1, -1)
                    profile_dim = user_profile_emb.shape[1]
                    logger.info(
                        "Using precomputed taste_embedding (%d-dim)", profile_dim
                    )
            except Exception:
                pass

        # Fallback: build profile from borrowed book embeddings
        if user_profile_emb is None and borrowed_ids:
            borrowed_with_emb = [
                b for b in all_books if b.id in borrowed_ids and b.embedding
            ]
            if borrowed_with_emb:
                dim_counts = Counter(len(b.embedding) for b in borrowed_with_emb)
                target_dim = dim_counts.most_common(1)[0][0]
                embs = [
                    np.array(b.embedding)
                    for b in borrowed_with_emb
                    if len(b.embedding) == target_dim
                ]
                if embs:
                    user_profile_emb = np.mean(embs, axis=0).reshape(1, -1)
                    profile_dim = user_profile_emb.shape[1]

        # -- Determine cold-start level --
        has_borrows = len(borrowed_ids) > 0
        has_prefs = bool(
            pref.preferred_authors or pref.preferred_genres or pref.preferred_tags
        )
        has_emb = user_profile_emb is not None

        if has_borrows and has_emb:
            cold_start = "full"
        elif has_borrows or has_prefs:
            cold_start = "warm"
        else:
            cold_start = "cold"

        logger.info(
            "Cold-start: %s, discovery: %s, borrows: %d, candidates: %d",
            cold_start,
            discovery_mode,
            len(borrowed_ids),
            len(candidates),
        )

        # -- Run sub-engines --
        scored: dict[UUID, ScoredBook] = {}

        if cold_start == "full":
            cb = await self._content.score(
                candidates,
                user_profile_emb,
                profile_dim,
                pref_authors,
                pref_genres,
                taste_author_aff,
                taste_genre_aff,
            )
            self._merge(scored, cb)

        if cold_start in ("full", "warm"):
            cf = await self._collab.score(
                user_id, candidates, all_borrows, all_reviews, borrowed_ids
            )
            self._merge(scored, cf)

        pop = await self._popularity.score(
            candidates, all_borrows, all_reviews, borrowed_ids
        )
        self._merge(scored, pop)

        if cold_start in ("full", "warm"):
            kg = await self._knowledge.score(
                user_id,
                candidates,
                all_borrows,
                borrowed_ids,
                all_books,
                pref_authors,
                pref_tags,
            )
            self._merge(scored, kg)

        if not scored:
            logger.info("No strategy produced results; recency fallback")
            candidates.sort(key=lambda b: b.created_at, reverse=True)
            for b in candidates[:limit]:
                b._rec_meta = {
                    "explanation": "New in the library -- check it out!",
                    "strategies_used": ["recency_fallback"],
                }
            return [(b, 0.0) for b in candidates[:limit]]

        # -- Blend scores --
        weights = self.STRATEGY_WEIGHTS.get(
            discovery_mode, self.STRATEGY_WEIGHTS["balanced"]
        )
        active: set[str] = set()
        for sb in scored.values():
            active.update(sb.strategy_scores.keys())

        total_w = sum(weights.get(s, 0) for s in active)
        if total_w > 0:
            adj_w = {s: weights.get(s, 0) / total_w for s in active}
        else:
            adj_w = {s: 1.0 / len(active) for s in active}

        for sb in scored.values():
            sb.score = sum(
                sb.strategy_scores.get(s, 0) * adj_w.get(s, 0) for s in active
            )

        # -- Discovery mode post-processing --
        if discovery_mode == "exploratory":
            rng = random.Random(42)
            for sb in scored.values():
                sb.score += rng.gauss(0, 0.03)
        elif discovery_mode == "similar":
            for sb in scored.values():
                if sb.score > 0:
                    sb.score = sb.score ** 1.2

        # -- Diversity injection (MMR-style author cap) --
        ranked = sorted(scored.values(), key=lambda s: -s.score)
        diverse = self._diversify(ranked, limit)

        # -- Build output --
        output: list[tuple[Book, float]] = []
        for sb in diverse:
            strategies = list(sb.strategy_scores.keys())
            explanation = (
                " | ".join(sb.explanations) if sb.explanations else "Recommended for you"
            )
            sb.book._rec_meta = {
                "explanation": explanation,
                "strategies_used": strategies,
            }
            output.append((sb.book, round(sb.score, 4)))

        logger.info(
            "Hybrid engine: %d recommendations (strategies: %s)",
            len(output),
            "+".join(sorted(active)),
        )
        return output

    # -- Helpers --
    @staticmethod
    def _merge(
        target: dict[UUID, ScoredBook], source: dict[UUID, ScoredBook]
    ) -> None:
        for bid, src in source.items():
            if bid in target:
                for strat, sc in src.strategy_scores.items():
                    target[bid].strategy_scores[strat] = sc
                target[bid].explanations.extend(src.explanations)
            else:
                target[bid] = src

    @staticmethod
    def _diversify(
        ranked: list[ScoredBook], limit: int, author_cap: int = 2
    ) -> list[ScoredBook]:
        """MMR-inspired diversity: cap max books per author in top-N."""
        result: list[ScoredBook] = []
        author_count: Counter = Counter()
        for sb in ranked:
            if len(result) >= limit:
                break
            if author_count[sb.book.author] >= author_cap:
                continue
            result.append(sb)
            author_count[sb.book.author] += 1
        return result
