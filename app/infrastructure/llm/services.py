"""LLM service implementations.

Each service consumes the structured :class:`PromptTemplate` objects defined in
``app.infrastructure.llm.prompts`` so that prompt engineering is centralised,
reusable, and version-controlled.
"""

import hashlib
import logging
from typing import Optional

import numpy as np

from app.domain.repositories import ILLMService
from app.infrastructure.llm.prompts import (
    BOOK_SUMMARY_PROMPT,
    REVIEW_CONSENSUS_PROMPT,
    SENTIMENT_ANALYSIS_PROMPT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock (development / testing)
# ---------------------------------------------------------------------------
class MockLLMService(ILLMService):
    """Returns deterministic results — useful for tests and offline dev."""

    async def generate_summary(self, content: str) -> str:
        """Generate mock summary using the structured prompt."""
        prompt = BOOK_SUMMARY_PROMPT.render_flat(content=content[:2000])
        logger.debug("MockLLM summary prompt (%d chars)", len(prompt))
        words = content.split()[:50]
        return f"Summary: {' '.join(words)}..."

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding from text."""
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(384).astype(float)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

    async def generate_review_consensus(self, book_title: str, reviews: list[dict]) -> str:
        """Generate mock rolling consensus using the structured prompt."""
        reviews_text = "\n".join(
            f"- Rating {r.get('rating', '?')}/5: {r.get('text', '')}" for r in reviews
        )
        prompt = REVIEW_CONSENSUS_PROMPT.render_flat(
            book_title=book_title, reviews_text=reviews_text
        )
        logger.debug("MockLLM consensus prompt (%d chars)", len(prompt))

        if not reviews:
            return "No reviews yet."
        avg_rating = sum(r.get("rating", 0) for r in reviews) / len(reviews)
        sentiments = [r.get("sentiment", "neutral") for r in reviews]
        return (
            f"Consensus for '{book_title}': {len(reviews)} reviews, "
            f"average rating {avg_rating:.1f}/5. "
            f"Overall sentiment: {max(set(sentiments), key=sentiments.count)}."
        )

    async def analyze_sentiment(self, text: str) -> str:
        """Mock sentiment analysis using the structured prompt."""
        prompt = SENTIMENT_ANALYSIS_PROMPT.render_flat(review_text=text)
        logger.debug("MockLLM sentiment prompt (%d chars)", len(prompt))

        positive_words = {
            "great",
            "good",
            "excellent",
            "amazing",
            "love",
            "wonderful",
            "best",
            "fantastic",
        }
        negative_words = {
            "bad",
            "terrible",
            "awful",
            "hate",
            "worst",
            "boring",
            "poor",
            "disappointing",
        }
        words = set(text.lower().split())
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        if pos > neg:
            return "positive"
        elif neg > pos:
            return "negative"
        return "neutral"


# ---------------------------------------------------------------------------
# Llama 3 (local / Ollama)
# ---------------------------------------------------------------------------
class LlamaLLMService(ILLMService):
    """Local LLM service backed by `Ollama <https://ollama.com>`_.

    Communicates with the Ollama REST API over HTTP using **httpx**.
    Falls back to :class:`MockLLMService` when Ollama is unreachable or
    returns an error, so the application degrades gracefully.

    Constructor args:
        base_url:  Ollama server URL (default ``http://localhost:11434``).
        model:     Model tag pulled into Ollama (default ``llama3``).
        timeout:   Per-request timeout in seconds (default 120).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._mock = MockLLMService()

    # -- internal helpers ---------------------------------------------------

    async def _chat(self, messages: list[dict[str, str]]) -> str:
        """Call ``POST /api/chat`` (non-streaming) and return the response text."""
        import httpx

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.4},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "")
        except Exception as exc:
            logger.warning("Ollama /api/chat failed (%s); falling back to mock", exc)
            return ""

    async def _embed(self, text: str) -> list[float] | None:
        """Call ``POST /api/embed`` and return the embedding vector."""
        import httpx

        payload = {"model": self.model, "input": text}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/embed", json=payload)
                resp.raise_for_status()
                data = resp.json()
                # Ollama returns {"embeddings": [[...]]}
                embeddings = data.get("embeddings", [])
                if embeddings and len(embeddings) > 0:
                    return embeddings[0]
                return None
        except Exception as exc:
            logger.warning("Ollama /api/embed failed (%s); falling back to mock", exc)
            return None

    # -- ILLMService interface ----------------------------------------------

    async def generate_summary(self, content: str) -> str:
        """Generate a book summary via Ollama chat."""
        messages = BOOK_SUMMARY_PROMPT.render(content=content[:4000])
        logger.info("LlamaLLM: requesting summary from %s (model=%s)", self.base_url, self.model)
        result = await self._chat(messages)
        return result or await self._mock.generate_summary(content)

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding via Ollama, falling back to mock."""
        logger.info("LlamaLLM: requesting embedding from %s (model=%s)", self.base_url, self.model)
        embedding = await self._embed(text)
        if embedding:
            return embedding
        return await self._mock.generate_embedding(text)

    async def generate_review_consensus(self, book_title: str, reviews: list[dict]) -> str:
        """Generate a consensus summary from multiple reviews via Ollama."""
        reviews_text = "\n".join(
            f"- Rating {r.get('rating', '?')}/5: {r.get('text', '')}" for r in reviews
        )
        messages = REVIEW_CONSENSUS_PROMPT.render(
            book_title=book_title, reviews_text=reviews_text
        )
        logger.info("LlamaLLM: requesting consensus from %s", self.base_url)
        result = await self._chat(messages)
        return result or await self._mock.generate_review_consensus(book_title, reviews)

    async def analyze_sentiment(self, text: str) -> str:
        """Classify review sentiment via Ollama."""
        messages = SENTIMENT_ANALYSIS_PROMPT.render(review_text=text)
        logger.info("LlamaLLM: requesting sentiment from %s", self.base_url)
        result = await self._chat(messages)
        cleaned = result.strip().lower()
        if cleaned in {"positive", "negative", "neutral"}:
            return cleaned
        # Ollama may return prose — try to extract the label
        for label in ("positive", "negative", "neutral"):
            if label in cleaned:
                return label
        logger.warning("Ollama sentiment response not parseable: %r; falling back", result)
        return await self._mock.analyze_sentiment(text)


# ---------------------------------------------------------------------------
# OpenAI (remote API)
# ---------------------------------------------------------------------------
class OpenAILLMService(ILLMService):
    """OpenAI-backed LLM provider.

    Requires ``LLM_API_KEY`` in env. Falls back to :class:`MockLLMService`
    when the ``openai`` package is missing or the API call fails.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self._mock = MockLLMService()

    async def _chat(self, messages: list[dict[str, str]]) -> str:
        """Call OpenAI chat completions (or fall back to mock)."""
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.4,
                max_tokens=512,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("OpenAI call failed (%s); falling back to mock", exc)
            return ""

    async def generate_summary(self, content: str) -> str:
        messages = BOOK_SUMMARY_PROMPT.render(content=content[:4000])
        result = await self._chat(messages)
        return result or await self._mock.generate_summary(content)

    async def generate_embedding(self, text: str) -> list[float]:
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self.api_key)
            response = await client.embeddings.create(model="text-embedding-3-small", input=text)
            return response.data[0].embedding
        except Exception:
            return await self._mock.generate_embedding(text)

    async def generate_review_consensus(self, book_title: str, reviews: list[dict]) -> str:
        reviews_text = "\n".join(
            f"- Rating {r.get('rating', '?')}/5: {r.get('text', '')}" for r in reviews
        )
        messages = REVIEW_CONSENSUS_PROMPT.render(book_title=book_title, reviews_text=reviews_text)
        result = await self._chat(messages)
        return result or await self._mock.generate_review_consensus(book_title, reviews)

    async def analyze_sentiment(self, text: str) -> str:
        messages = SENTIMENT_ANALYSIS_PROMPT.render(review_text=text)
        result = await self._chat(messages)
        cleaned = result.strip().lower()
        if cleaned in {"positive", "negative", "neutral"}:
            return cleaned
        return await self._mock.analyze_sentiment(text)
