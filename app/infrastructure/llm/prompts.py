"""Structured, reusable prompt templates for LLM interactions.

This module implements a PromptTemplate pattern so that every LLM call uses
a well-defined, version-controlled prompt.  Prompts are separated from service
logic to allow independent iteration and easy auditing.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptTemplate:
    """A reusable prompt template with named placeholders.

    Usage::

        tpl = PromptTemplate(
            name="book_summary",
            system="You are a literary analyst.",
            user="Summarise the following book content:\n\n{content}",
        )
        messages = tpl.render(content="Once upon a time …")
    """

    name: str
    system: str
    user: str
    description: str = ""
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def render(self, **kwargs: Any) -> list[dict[str, str]]:
        """Return an OpenAI-style messages list with placeholders filled."""
        return [
            {"role": "system", "content": self.system.format(**kwargs)},
            {"role": "user", "content": self.user.format(**kwargs)},
        ]

    def render_flat(self, **kwargs: Any) -> str:
        """Return a single-string prompt (system + user) for simpler APIs."""
        sys_text = self.system.format(**kwargs)
        usr_text = self.user.format(**kwargs)
        return f"{sys_text}\n\n{usr_text}"


# =========================================================================
# Pre-defined prompts used across the Intelligence Layer
# =========================================================================

BOOK_SUMMARY_PROMPT = PromptTemplate(
    name="book_summary",
    description="Generate a concise summary of uploaded book content.",
    version="1.0",
    tags=["ingestion", "summary"],
    system=(
        "You are LuminaLib's book-analysis assistant.  "
        "Produce clear, concise summaries that capture the main themes, "
        "arguments, and narrative arc of the text provided."
    ),
    user=(
        "Please summarise the following book content in 3-5 sentences.  "
        "Focus on the key themes, central argument, and intended audience.\n\n"
        "---\n{content}\n---"
    ),
)

SENTIMENT_ANALYSIS_PROMPT = PromptTemplate(
    name="review_sentiment",
    description="Classify the sentiment of a single book review.",
    version="1.0",
    tags=["review", "sentiment"],
    system=(
        "You are a sentiment-analysis model.  "
        "Classify the sentiment of the user review as exactly one of: "
        "positive, negative, or neutral.  "
        "Respond with a single word."
    ),
    user="Review text:\n\n{review_text}",
)

REVIEW_CONSENSUS_PROMPT = PromptTemplate(
    name="review_consensus",
    description="Generate a rolling consensus summary from multiple reviews.",
    version="1.0",
    tags=["review", "consensus", "analysis"],
    system=(
        "You are LuminaLib's review-aggregation assistant.  "
        "Given a list of reader reviews for a book, produce a concise "
        "consensus paragraph that highlights common praise, common "
        "criticisms, and the overall reader sentiment."
    ),
    user=(
        "Book: {book_title}\n\n"
        "Reviews:\n{reviews_text}\n\n"
        "Write a consensus summary in 3-4 sentences."
    ),
)

EMBEDDING_PROMPT = PromptTemplate(
    name="text_embedding",
    description="Prepare text for embedding generation.",
    version="1.0",
    tags=["embedding"],
    system="You are a text-embedding preprocessor.",
    user="Generate a dense vector representation for:\n\n{text}",
)

BOOK_METADATA_PROMPT = PromptTemplate(
    name="book_metadata_extraction",
    description="Extract title, author, and genre from the opening text of a book.",
    version="1.0",
    tags=["ingestion", "metadata", "extraction"],
    system=(
        "You are a metadata extraction assistant for a digital library. "
        "Given the opening text of a book, extract its title, author, and genre. "
        "Respond ONLY with a valid JSON object containing exactly these three keys: "
        "'title', 'author', 'genre'. "
        "If you cannot determine a value with confidence, use an empty string. "
        "Do not include any explanation or markdown fencing — JSON only."
    ),
    user=(
        "Extract the book metadata from this opening text:\n\n"
        "{text}\n\n"
        'Respond with JSON only, e.g. {{"title": "Dune", "author": "Frank Herbert", "genre": "science fiction"}}'
    ),
)

TASTE_CLUSTER_PROMPT = PromptTemplate(
    name="taste_cluster",
    description="Generate a short reader persona label from a user's taste data.",
    version="1.0",
    tags=["preference", "persona", "taste"],
    system=(
        "You are LuminaLib's reader-profiling assistant. "
        "Generate a concise, evocative reader persona label (3-5 words) that "
        "captures the user's literary taste. "
        "Respond with just the label — no explanation."
    ),
    user=(
        "Favorite authors: {top_authors}\n"
        "Favorite genres/tags: {top_genres}\n"
        "Average rating given: {avg_rating}/5\n"
        "Books read: {total_borrows}\n\n"
        "Generate a short reader persona label, e.g. "
        "'Literary Fiction Enthusiast' or 'Sci-Fi Power Reader'."
    ),
)

# Registry for programmatic access
PROMPT_REGISTRY: dict[str, PromptTemplate] = {
    tpl.name: tpl
    for tpl in [
        BOOK_SUMMARY_PROMPT,
        BOOK_METADATA_PROMPT,
        SENTIMENT_ANALYSIS_PROMPT,
        REVIEW_CONSENSUS_PROMPT,
        EMBEDDING_PROMPT,
        TASTE_CLUSTER_PROMPT,
    ]
}
