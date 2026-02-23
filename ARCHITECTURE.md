# LuminaLib — Architecture & Design Decisions

> This document explains **why** the system is built the way it is.  Every
> architectural choice is mapped to a design principle or a concrete trade-off.

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Clean Architecture Layers](#clean-architecture-layers)
3. [Dependency Injection & The Composition Root](#dependency-injection--the-composition-root)
4. [Data Flow Diagrams](#data-flow-diagrams)
5. [Database Schema & ER Diagram](#database-schema--er-diagram)
6. [LLM Integration Strategy](#llm-integration-strategy)
7. [ML Recommendation Engine](#ml-recommendation-engine)
8. [User Preference System](#user-preference-system)
9. [Authentication & Authorisation](#authentication--authorisation)
10. [Background Task Pipeline](#background-task-pipeline)
11. [Deployment Topology](#deployment-topology)
12. [SOLID Principles Mapping](#solid-principles-mapping)
13. [Key Trade-Offs & Rationale](#key-trade-offs--rationale)
14. [Code Hygiene](#code-hygiene)

---

## 1. System Architecture Overview

The diagram below shows every runtime component and how they communicate:

```
 +----------------------------------------------------------------------------+
 |                        Docker Compose Network                              |
 |                                                                            |
 |  +-------------------+        +-------------------+                        |
 |  |     Client        |        |      Ollama       |                        |
 |  |  (curl / browser) |        |   LLM Inference   |                        |
 |  |                   |        |  :11434 (internal) |                        |
 |  +--------+----------+        +--------^----------+                        |
 |           | HTTP :5223                  | HTTP /api/chat                    |
 |           v                             |      /api/embed                   |
 |  +------------------------------------------------------------+            |
 |  |                  FastAPI  (api)                             |            |
 |  |                                                            |            |
 |  |   +----------+  +----------+  +----------+  +------+      |            |
 |  |   |  Auth    |  |  Books   |  |  Recs    |  | Pref |      |            |
 |  |   +----+-----+  +----+-----+  +----+-----+  +--+---+     |            |
 |  |        |              |              |           |          |            |
 |  |        v              v              v           v          |            |
 |  |   +---------------------------------------------------------+           |
 |  |   |            Service Layer                                |           |
 |  |   |  AuthService . BookService . ReviewService              |           |
 |  |   |  MLRecommendationService . PreferenceService            |           |
 |  |   |  Celery Tasks  (enqueued to worker via Redis)           |           |
 |  |   +--------------------+------------------------------------+           |
 |  |                        | uses abstract interfaces                       |
 |  |                        v                                                |
 |  |   +---------------------------------------------+                      |
 |  |   |      Infrastructure (Adapters)               |                      |
 |  |   |  7 Repositories . 3 LLMs . 2 Storage        |                      |
 |  |   +--------+-----------+----------+--------------+                      |
 |  +------------|-----------|----------|--------------------+                 |
 |               |           |          |                                      |
 |       asyncpg |    httpx  |    boto3 |                                      |
 |               v           v          v                                      |
 |  +-------------+  +------------+  +--------------+  +----------------+     |
 |  | PostgreSQL  |  |  Ollama    |  |    MinIO     |  |    Redis       |     |
 |  |   :5432     |  |  :11434    |  | :9000 / :9001|  |    :6379       |     |
 |  |  (db)       |  |            |  |   (minio)    |  |   (redis)      |     |
 |  +-------------+  +------------+  +--------------+  +----------------+     |
 |                                                                            |
 +----------------------------------------------------------------------------+
```

**Key points:**
- The FastAPI process is the single entry-point -- every external service is
  accessed through an **abstract interface** defined in the domain layer.
- PostgreSQL stores relational data (8 tables) via SQLAlchemy's async ORM.
- Ollama provides local LLM inference (summaries, embeddings, sentiment,
  consensus, taste-cluster labels).
- MinIO serves as S3-compatible object storage for uploaded book files.
- Redis is the **Celery broker and result backend** (task queue between `api` and
  `worker`) and the **JWT token revocation store** (blacklist for signed-out tokens).

---

## 2. Clean Architecture Layers

LuminaLib follows Robert C. Martin's *Clean Architecture* principle:
**dependencies always point inward**.  The domain layer has zero external
imports; the infrastructure layer depends on the domain, never the reverse.

```
+--------------------------------------------------------------+
|              Presentation Layer                               |
|   routes.py . auth_routes.py . preference_routes.py          |
|   recommendation_routes.py . schemas.py                      |
+--------------------+--------- -------------------------------+
                     |  depends on
+--------------------v-----------------------------------------+
|           Application / Service Layer                         |
|   BookService . AuthService . ReviewService                   |
|   MLRecommendationService . PreferenceService                 |
|   Celery Worker Tasks (summary, sentiment, consensus)         |
+--------------------+-----------------------------------------+
                     |  depends on
+--------------------v-----------------------------------------+
|                Domain Layer                                   |
|   entities.py  (User, Book, BorrowRecord, Review,            |
|      UserPreference, UserInteraction, UserTasteProfile,       |
|      BookAnalysis)                                            |
|   repositories.py  (IUserRepo, IBookRepo, IBorrowRepo,       |
|      IReviewRepo, IUserPreferenceRepo,                        |
|      IUserInteractionRepo, IUserTasteProfileRepo,             |
|      IStorageService, ILLMService,                            |
|      IRecommendationService)                                  |
+--------------------+-----------------------------------------+
                     |  implemented by
+--------------------v-----------------------------------------+
|           Infrastructure Layer (Adapters)                     |
|   database/repository.py  (7 concrete repositories)          |
|   llm/services.py         (Mock . Llama . OpenAI)            |
|   llm/prompts.py          (PromptTemplate + registry)        |
|   storage/local.py . storage/s3.py                           |
+--------------------------------------------------------------+
```

### Why this matters

| Benefit                     | How it is achieved                                                                     |
|-----------------------------|----------------------------------------------------------------------------------------|
| **Testability**             | Service tests inject `AsyncMock` objects for every repository & infrastructure service |
| **Swappability**            | Change `LLM_PROVIDER=llama` to `openai` -- zero code changes, one env var              |
| **Framework independence**  | Domain entities are pure Python `@dataclass` -- no FastAPI or SQLAlchemy imports        |
| **DB migration freedom**    | Repository abstraction means switching from PostgreSQL to another store touches only `infrastructure/database/` |
| **Extensibility**           | Adding a new preference layer, recommendation strategy, or LLM provider requires zero changes to existing services |

### Design Decision: Pure Dataclass Entities

Entities are plain `@dataclass` objects, **not** SQLAlchemy models.  This keeps
the domain layer free from any ORM dependency.  The repository implementations
in `infrastructure/database/repository.py` translate between ORM models and
domain entities at the persistence boundary.

**8 domain entities:**

| Entity | Fields | Purpose |
|---|---|---|
| `User` | id, username, email, hashed_password, is_active, timestamps | Identity & auth |
| `Book` | id, title, author, **genre**, file_path, file_size, mime_type, content_hash, summary, embedding, timestamps | Core library item |
| `BorrowRecord` | id, user_id, book_id, borrowed_at, returned_at | Borrow lifecycle |
| `Review` | id, user_id, book_id, rating, text, sentiment, created_at | Reader feedback |
| `UserPreference` | id, user_id, preferred_authors/genres/tags, reading_pace, discovery_mode, content_preferences, notify flag, embedding | Layer 1: Explicit preferences |
| `UserInteraction` | id, user_id, book_id, interaction_type, interaction_data, weight, created_at | Layer 2: Implicit signals |
| `UserTasteProfile` | id, user_id, genre/author affinities, avg_rating, totals, taste_embedding, taste_cluster, confidence | Layer 3: AI-derived profile |
| `BookAnalysis` | book_id, review_count, average_rating, consensus_summary | Cached GenAI consensus |

**Trade-off:** This means there is a mapping step (ORM <-> entity).  We
accepted this cost for cleaner separation, easier unit testing, and the ability
to swap databases without touching business logic.

---

## 3. Dependency Injection & The Composition Root

FastAPI's `Depends()` system is the DI container.  All wiring lives in a
**single file** -- `app/core/dependencies.py` -- the *Composition Root*.

```
                   dependencies.py  (Composition Root)
                   ------------------------------------
                          |
      +-------------------+----------------------------------+
      |                   |                                   |
      v                   v                                   v
 get_storage_service  get_llm_service              get_*_repository (x7)
      |                   |                                   |
 +----+----+        +-----+----+                       +------+-----+
 | local   |        | mock     |                       | UserRepo   |
 | s3      |        | llama    |                       | BookRepo   |
 +---------+        | openai   |                       | BorrowRepo |
                    +----------+                       | ReviewRepo |
                                                       | PrefRepo   |
      v                                                | InterRepo  |
 get_*_service (x5)                                    | TasteRepo  |
      |                                                +------------+
 +----+-----------+
 | BookService    |
 | ReviewService  |
 | RecommService  |
 | PrefService    |
 | get_current_user|
 +----------------+
```

### How it works

1. **Infrastructure providers** read `settings.*` (typed `Literal` values) to
   decide which concrete class to return:

   ```python
   def get_storage_service() -> IStorageService:
       if settings.storage_backend == "local":
           return LocalStorageService(settings.storage_path)
       elif settings.storage_backend == "s3":
           return S3StorageService(...)

   def get_llm_service() -> ILLMService:
       if settings.llm_provider == "mock":
           return MockLLMService()
       elif settings.llm_provider == "llama":
           return LlamaLLMService(...)
       elif settings.llm_provider == "openai":
           return OpenAILLMService(...)
   ```

2. **Repository providers** (x7) receive an `AsyncSession` via `Depends(get_db)`
   and wrap it in the concrete repository:

   ```python
   async def get_book_repository(session = Depends(get_db)) -> IBookRepository:
       return BookRepository(session)

   async def get_preference_repository(session = ...) -> IUserPreferenceRepository:
       return UserPreferenceRepository(session)

   async def get_interaction_repository(session = ...) -> IUserInteractionRepository:
       return UserInteractionRepository(session)

   async def get_taste_profile_repository(session = ...) -> IUserTasteProfileRepository:
       return UserTasteProfileRepository(session)
   ```

3. **Service providers** (x5) compose repositories + infrastructure into
   application services:

   ```python
   async def get_book_service(repo=Depends(get_book_repository)) -> BookService:
       return BookService(
           book_repository=repo,
           storage_service=get_storage_service(),
           llm_service=get_llm_service(),
       )

   async def get_preference_service(
       pref_repo, interaction_repo, taste_repo,
       book_repo, borrow_repo, review_repo
   ) -> PreferenceService:
       return PreferenceService(
           preference_repo=pref_repo,
           interaction_repo=interaction_repo,
           taste_repo=taste_repo,
           book_repo=book_repo,
           borrow_repo=borrow_repo,
           review_repo=review_repo,
           llm_service=get_llm_service(),
       )
   ```

### Design Decision: Why Not a DI Framework?

We evaluated libraries like `dependency-injector` and `lagom`, but FastAPI's
built-in `Depends()` already provides:
- Lazy instantiation per request
- Automatic session lifecycle management
- Clear, readable provider functions

A third-party container would add complexity without proportional benefit for
this project's scope.

---

## 4. Data Flow Diagrams

### 4.1 Book Upload Pipeline

```
  Client              API               BookService         Storage        LLM (background)
    |                  |                     |                 |                    |
    |  POST /books/    |                     |                 |                    |
    |  (multipart file)|                     |                 |                    |
    |----------------->|                     |                 |                    |
    |                  |  create_book(file)   |                 |                    |
    |                  |-------------------->|                 |                    |
    |                  |                     |  save_file()    |                    |
    |                  |                     |---------------->|                    |
    |                  |                     |  <-- file_path --|                    |
    |                  |                     |                 |                    |
    |                  |                     |  SHA-256 hash   |                    |
    |                  |                     |  detect MIME    |                    |
    |                  |                     |  extract text   |                    |
    |                  |                     |                 |                    |
    |                  |                     |  INSERT book    |                    |
    |                  |                     |  (summary=null) |                    |
    |                  |                     |                 |                    |
    |                  |                     |  generate_book_summary.delay() ---->| Redis
    |                  |                     |  (X-Task-ID header)     |           |  (broker)
    |  <-- 201 --------|  <-- Book ----------|                 |       |           |
    |  (summary=null,  |                     |                 |       |  Celery   |
    |   X-Task-ID)     |                     |                 |       |  Worker   |
    |                  |                     |                 |  <----|  consumes |
    |                  |                     |                 |       |  task     |
    |                  |                     |                 |   generate_summary|
    |                  |                     |                 |   generate_embedd.|
    |                  |                     |                 |   UPDATE book     |
    |                  |                     |                 |   SET summary,    |
    |                  |                     |                 |       embedding   |
```

**Key design choice:** The HTTP response returns **immediately** (201) with
`summary: null`.  The expensive LLM work is dispatched via `generate_book_summary.delay()`
to a **Celery worker** running in a separate `worker` container.  The caller receives an
`X-Task-ID` header and can poll `GET /tasks/{task_id}` to track completion.
This keeps p99 latency low and ensures tasks survive API process restarts.

### 4.2 Review Submission & Consensus Pipeline

When a review is submitted via `POST /books/{id}/reviews`, the following happens:

1. **ReviewService.create_review()** validates the user has borrowed the book
   and has not already reviewed it, then INSERTs the review.
2. The API layer dispatches **3 parallel actions**:
   - `analyze_review_sentiment.delay(review_id, text)` — Celery task, LLM classifies sentiment; task ID returned in `X-Sentiment-Task-ID` header
   - `update_rolling_consensus.delay(book_id)` — Celery task, LLM generates consensus; task ID returned in `X-Consensus-Task-ID` header
   - `record_interaction` (inline) — records a Layer 2 implicit interaction

**Key design choices:**
- The rolling consensus is regenerated on every review submission --
  `GET /books/{id}/analysis` can serve the cached result instantly.
- Cache-first logic in `ReviewService.get_book_analysis()`: reads from
  `book_analyses` table first, only falls back to on-demand LLM if no cached
  result exists.

### 4.3 ML Recommendation Pipeline (5-Strategy Hybrid)

```
  Client              API              MLRecommendationService              DB
    |                  |                        |                            |
    | GET /recs?       |                        |                            |
    |----------------->|                        |                            |
    |                  |  get_user_recs()        |                            |
    |                  |----------------------->|                            |
    |                  |                        |  Load all data:            |
    |                  |                        |  - user borrows ---------->|
    |                  |                        |  - all books ------------->|
    |                  |                        |  - all borrows (collab.) ->|
    |                  |                        |  - all reviews (pop.) ---->|
    |                  |                        |  - user preferences (L1) ->|
    |                  |                        |  - taste profile (L3) ---->|
    |                  |                        |                            |
    |                  |                        |  Determine cold-start:     |
    |                  |                        |  full / warm / cold        |
    |                  |                        |                            |
    |                  |                        |  Run sub-engines:          |
    |                  |                        |  1. Content-Based (CB)     |
    |                  |                        |  2. Collaborative (CF)     |
    |                  |                        |  3. Popularity (POP)       |
    |                  |                        |  4. Knowledge Graph (KG)   |
    |                  |                        |                            |
    |                  |                        |  Weighted ensemble blend   |
    |                  |                        |  (weights by discovery_mode)|
    |                  |                        |                            |
    |                  |                        |  MMR diversity (author cap)|
    |                  |                        |                            |
    |  <-- top-N ------|  <-- [(Book, score)] --|                            |
    |  + explanations  |  + strategy metadata   |                            |
```

### 4.4 Preference System Pipeline

The preference system operates across three layers:

1. **Layer 1 (Explicit):** User calls `PUT /preferences/` with authors, genres, tags
2. **Layer 2 (Implicit):** Borrow/return/review actions auto-record interactions
3. **Layer 3 (Computed):** `POST /preferences/taste/recompute` triggers:
   - Aggregate borrows -> author frequency -> normalised affinities
   - Aggregate reviews -> rating patterns
   - Aggregate interactions -> weighted interest signals
   - Compute taste embedding (rating-weighted mean of book embeddings)
   - LLM generates taste-cluster label (e.g. "Sci-Fi Power Reader")
   - Confidence score = `min(1.0, data_points / 20.0)`

---

## 5. Database Schema & ER Diagram

### Table Summary

| Table | Rows | Purpose | Key Constraints |
|---|---|---|---|
| `users` | N users | Identity & auth | UQ(email), UQ(username) |
| `books` | N books | Library catalogue + AI data | UQ(file_path), UQ(content_hash) |
| `borrows` | N borrows | Borrow lifecycle | FK(user_id, book_id), cascade delete |
| `reviews` | N reviews | Reader feedback + sentiment | UQ(user_id, book_id), cascade delete |
| `user_preferences` | 1 per user | Layer 1: explicit prefs | UQ(user_id) |
| `user_interactions` | event log | Layer 2: implicit signals | 3 indexes for fast aggregation |
| `user_taste_profiles` | 1 per user | Layer 3: AI-derived profile | UQ(user_id) |
| `book_analyses` | 1 per book | Cached GenAI consensus | UQ(book_id), cascade delete |

**Total: 8 tables, 9 ORM models** (including `Base`).

### Table Columns

**users:** id (UUID PK), username (STR), email (STR UQ), hashed_password (STR),
is_active (BOOL), created_at (DT), updated_at (DT)

**books:** id (UUID PK), title (STR), author (STR), genre (STR), file_path (STR UQ),
file_size (INT), mime_type (STR), content_hash (STR UQ), summary (TEXT),
embedding (FLOAT[]), created_at (DT), updated_at (DT)

**borrows:** id (UUID PK), user_id (UUID FK), book_id (UUID FK), borrowed_at (DT),
returned_at (DT nullable)

**reviews:** id (UUID PK), user_id (UUID FK), book_id (UUID FK), rating (INT 1-5),
text (TEXT), sentiment (STR nullable), created_at (DT), UQ(user_id, book_id)

**user_preferences (L1):** id (UUID PK), user_id (UUID FK UQ), preferred_authors (STR[]),
preferred_genres (STR[]), preferred_tags (STR[]), reading_pace (STR),
discovery_mode (STR), content_preferences (JSONB), notify_new_by_favorite_author (BOOL),
embedding (FLOAT[]), updated_at (DT)

**user_interactions (L2):** id (UUID PK), user_id (UUID FK), book_id (UUID FK),
interaction_type (STR), interaction_data (JSONB), weight (FLOAT), created_at (DT).
Indexes: IX(user_id, interaction_type), IX(book_id), IX(created_at)

**user_taste_profiles (L3):** id (UUID PK), user_id (UUID FK UQ),
genre_affinities (JSONB), author_affinities (JSONB), avg_rating_given (FLOAT),
total_borrows (INT), total_reviews (INT), taste_embedding (FLOAT[]),
taste_cluster (STR), confidence_score (FLOAT), last_computed_at (DT)

**book_analyses:** id (UUID PK), book_id (UUID FK UQ), review_count (INT),
average_rating (FLOAT), consensus_summary (TEXT), updated_at (DT)

### Relationship Summary

| Relationship                        | Type    | Constraint                                        |
|-------------------------------------|---------|---------------------------------------------------|
| `users` -> `borrows`               | 1 : N   | FK `borrows.user_id` -> `users.id`               |
| `books` -> `borrows`               | 1 : N   | FK, **cascade delete**                            |
| `users` -> `reviews`               | 1 : N   | FK `reviews.user_id` -> `users.id`               |
| `books` -> `reviews`               | 1 : N   | FK, **cascade delete**                            |
| `users` <-> `reviews` (per book)   | 1 : 1   | `UniqueConstraint("user_id", "book_id")`          |
| `books` <-> `book_analyses`        | 1 : 1   | `book_analyses.book_id` is `unique=True`, **cascade delete** |
| `users` <-> `user_preferences`     | 1 : 1   | `user_preferences.user_id` is `unique=True`       |
| `users` -> `user_interactions`     | 1 : N   | FK, `lazy="dynamic"` for large event logs         |
| `books` -> `user_interactions`     | 1 : N   | FK, **cascade delete**                            |
| `users` <-> `user_taste_profiles`  | 1 : 1   | `user_taste_profiles.user_id` is `unique=True`    |

### Design Decision: Cascade Delete on Books

Deleting a book cascades to its `borrows`, `reviews`, `user_interactions`, and
`book_analyses`.  This ensures referential integrity without orphan rows.
Configured via `cascade="all, delete-orphan"` on all `BookModel` relationships.

### Design Decision: UUID Primary Keys

All primary keys are **UUID v4** instead of auto-incrementing integers.

| For UUID                                      | Against UUID                              |
|-----------------------------------------------|-------------------------------------------|
| No sequential guessing (security)             | 16 bytes vs 4 bytes per FK                |
| Safe for distributed ID generation            | Slightly slower B-tree indexing            |
| Merge-friendly (no PK collisions)             | Harder to read in logs                    |
| Client-side ID generation possible            |                                           |

We chose UUIDs because the system may be horizontally scaled and because
the security benefit (no enumerable IDs) outweighs the marginal storage cost.

### Design Decision: Eager Loading (selectin) & Dynamic for Event Logs

Most relationships use `lazy="selectin"` to avoid N+1 queries.  The exception
is `user_interactions`, which uses `lazy="dynamic"` because the event log can
grow very large -- dynamic loading returns a query object that can be filtered
and paginated rather than eagerly loading all rows.

---

## 6. LLM Integration Strategy

### Three-Provider Architecture

```
          ILLMService (interface -- 4 methods)
          +---------------------------+
          | generate_summary()        |
          | generate_embedding()      |
          | generate_review_consensus()|
          | analyze_sentiment()       |
          +----------+----------------+
                     |
         +-----------+-----------+
         |           |           |
    +----v---+  +----v----+  +---v-----+
    | Mock   |  | Llama   |  | OpenAI  |
    | LLM    |  | LLM     |  | LLM     |
    |        |  | (httpx)  |  | (openai)|
    +--------+  +----+----+  +---+-----+
                     |            |
               +-----v----+  +---v--------+
               |  Ollama   |  | OpenAI    |
               |  :11434   |  | Cloud     |
               +----------+  +-----------+
```

| Provider      | When to Use                       | Embedding Dim | Config                                | Fallback |
|---------------|-----------------------------------|:---:|---------------------------------------|---|
| **MockLLM**   | Tests, CI, local dev without GPU  | 384 | `LLM_PROVIDER=mock` | -- (is the fallback) |
| **LlamaLLM**  | On-prem / air-gapped deployment   | model-dependent | `LLM_PROVIDER=llama`, Ollama running | -> MockLLM |
| **OpenAILLM** | Cloud production with GPT-4       | 1536 | `LLM_PROVIDER=openai`, API key set   | -> MockLLM |

### Graceful Fallback Chain

Both `LlamaLLMService` and `OpenAILLMService` instantiate an internal
`MockLLMService` and fall back to it on any exception (network timeout,
bad API key, model not loaded, etc.).  The application **never crashes**
due to an LLM failure -- it degrades to deterministic mock output.

```python
# In LlamaLLMService:
async def generate_summary(self, content: str) -> str:
    messages = BOOK_SUMMARY_PROMPT.render(content=content[:4000])
    result = await self._chat(messages)
    return result or await self._mock.generate_summary(content)  # fallback
```

### Prompt Engineering

Prompts are centralised in `app/infrastructure/llm/prompts.py` using an
**immutable, versioned `PromptTemplate` dataclass**:

```python
@dataclass(frozen=True)
class PromptTemplate:
    name: str           # unique identifier
    system: str         # system message (role/persona)
    user: str           # user message with {placeholders}
    description: str    # what this prompt does
    version: str        # version tracking ("1.0")
    tags: list[str]     # categorisation tags

    def render(self, **kwargs) -> list[dict[str, str]]:   # OpenAI-style messages
    def render_flat(self, **kwargs) -> str:                # single-string prompt
```

**4 registered prompts:**

| Constant | Name | Used by | Tags |
|---|---|---|---|
| `BOOK_SUMMARY_PROMPT` | `book_summary` | `generate_summary()` in all 3 LLM services | `ingestion, summary` |
| `SENTIMENT_ANALYSIS_PROMPT` | `review_sentiment` | `analyze_sentiment()` in all 3 LLM services | `review, sentiment` |
| `REVIEW_CONSENSUS_PROMPT` | `review_consensus` | `generate_review_consensus()` in all 3 LLM services | `review, consensus, analysis` |
| `EMBEDDING_PROMPT` | `text_embedding` | embedding preprocessing | `embedding` |

**`PROMPT_REGISTRY`** -- a dict keyed by name for programmatic lookup.

**Why this design:**

| Principle | Evidence |
|---|---|
| **Separation of concerns** | Prompts live in `prompts.py`, not scattered across service methods |
| **Single source of truth** | Changing a prompt in one place updates all three providers |
| **Version tracking** | Each prompt has a `version` field for A/B testing or rollback |
| **Auditable** | `tags` and `description` make it easy to audit which prompts power which feature |
| **Immutable** | `@dataclass(frozen=True)` prevents accidental mutation at runtime |
| **Provider-agnostic** | `render()` returns OpenAI-style messages that work with both Ollama and OpenAI |

### Design Decision: Ollama over vLLM / TGI

We chose Ollama as the default local provider because:

1. **Zero-config Docker** -- single container, no CUDA setup required for CPU mode
2. **REST-compatible** -- simple `/api/chat` and `/api/embed` endpoints
3. **Model management** -- `ollama pull llama3` handles download + quantisation
4. **Swap cost** -- switching to vLLM or TGI only requires a new `ILLMService`
   implementation; the rest of the app is unchanged

---

## 7. ML Recommendation Engine

### Architecture: Five-Strategy Hybrid Engine (697 lines)

LuminaLib implements a **five-strategy hybrid recommendation engine** that
selects and blends strategies based on available user data:

```
+----------------------------------------------------------------------+
|                     MLRecommendationService                          |
|                     (implements IRecommendationService)               |
|                                                                      |
|  +--------------+ +--------------+ +--------------+ +--------------+ |
|  | 1. Content-  | | 2. Collab.   | | 3. Popularity| | 4. Knowledge | |
|  |    Based     | |    Filtering | |    & Trending| |    Graph     | |
|  |  (cosine sim | | (Jaccard +   | | (Wilson +    | | (author co-  | |
|  |   + pref     | |  item-item   | |  time-decay  | |  read graph  | |
|  |   boosts)    | |  overlap)    | |  + trending) | |  + tags)     | |
|  +------+-------+ +------+-------+ +------+-------+ +------+-------+ |
|         |                |                |                |          |
|         +----------------+----------------+----------------+          |
|                                   |                                   |
|                          5. Hybrid Orchestrator                       |
|                    +--------------+--------------+                    |
|                    |  Weighted ensemble blend     |                   |
|                    |  Cold-start cascade          |                   |
|                    |  MMR diversity (author cap)  |                   |
|                    +-----------------------------+                    |
+----------------------------------------------------------------------+
```

### Strategy Details

| # | Strategy | Algorithm | What it uses | Strength |
|---|---|---|---|---|
| 1 | **Content-Based** | Cosine similarity on LLM embeddings + preference boosts (L1 author/genre + L3 affinities) | Book embeddings, user taste profile | Best when user has borrows + embeddings |
| 2 | **Collaborative** | User-user Jaccard similarity (top-20 neighbours) + item-item co-occurrence coefficient | All borrow records, all reviews | Best when many users have overlapping behaviour |
| 3 | **Popularity** | Wilson lower-bound confidence + exponential time-decay (14-day half-life) + trending detection | All borrows (time-stamped), all reviews (ratings) | Best for cold-start users |
| 4 | **Knowledge Graph** | Author co-read adjacency graph + preferred author/tag matching | Borrow network, explicit preferences | Best for genre/author discovery |
| 5 | **Hybrid Orchestrator** | Weighted ensemble with discovery-mode-adaptive weights, MMR diversity | Outputs from strategies 1--4 | Combines all signals |

### Cold-Start Cascade

The engine adapts to how much data is available for the user:

| Level | Condition | Strategies activated |
|---|---|---|
| **full** | User has borrows + embeddings | CB + CF + POP + KG (all 4) |
| **warm** | User has borrows or explicit preferences | CF + KG + POP (3) |
| **cold** | No borrows, no preferences | POP only -> recency fallback |

### Discovery-Mode Weights

The user's `discovery_mode` preference (from Layer 1) controls how strategies
are blended:

| Mode | CB | CF | POP | KG | Behaviour |
|---|:---:|:---:|:---:|:---:|---|
| `similar` | 0.50 | 0.25 | 0.05 | 0.20 | Recommend more of the same |
| `balanced` | 0.35 | 0.25 | 0.15 | 0.25 | Mix of familiar + new |
| `exploratory` | 0.20 | 0.20 | 0.30 | 0.30 | Push new genres/authors + random noise |

### Diversity Injection (MMR-Style)

After scoring, an author-cap rule ensures variety: at most 2 books by the same
author appear in the top-N.  In `exploratory` mode, a small Gaussian noise term
is added to scores to introduce serendipity.

### Embedding Space

Every book gets a dense vector when its summary is generated:

```
book file -> text extraction -> LLM.generate_embedding(text) -> Float[]
```

These vectors live in `books.embedding` (PostgreSQL `ARRAY(Float)`).
NumPy + scikit-learn's `cosine_similarity` does the scoring at query time.

**Trade-off:** Storing embeddings in a PostgreSQL array column means we do a
full table scan + in-memory similarity.  For <10k books this is fine.  At
larger scale, we'd migrate to pgvector or a dedicated vector DB -- the
`IBookRepository.list_all()` call is the only touch point.

### Why Hybrid Instead of Single-Strategy?

| Single strategy | Problem |
|---|---|
| Content-based only | Filter bubble -- only recommends what you already like |
| Collaborative only | Needs many users with overlapping behaviour; fails for new users |
| Popularity only | Ignores personal taste entirely |

The hybrid approach combines the strengths: content-based provides relevance,
collaborative adds social proof, popularity handles cold starts, and the
knowledge graph enables genre/author discovery.

---

## 8. User Preference System

### Three-Layer Architecture

The preference system uses a **three-layer model** that separates what the user
*says*, what we *observe*, and what the AI *derives*:

**Layer 1 -- Explicit (user_preferences table):**
What the user tells us directly.
- preferred_authors, preferred_genres, preferred_tags
- reading_pace (casual | moderate | avid)
- discovery_mode (similar | balanced | exploratory)
- content_preferences (JSONB -- extensible key/value)
- notify_new_by_favorite_author (boolean)

**Layer 2 -- Implicit (user_interactions table):**
What we observe them doing. Event-sourced log.
- interaction_type: borrow | return | review | bookmark | view
- interaction_data: {rating, sentiment, duration_days, ...}
- weight: signal strength multiplier
  - borrow=1.0, return=0.3, review=1.5, bookmark=0.8, view=0.2
  - high-rating review: weight x 1.3
  - low-rating review: weight x 0.5

**Layer 3 -- Computed (user_taste_profiles table):**
AI-derived composite of Layer 1 + Layer 2.
- genre_affinities: {"fiction": 0.8, "science": 0.6}
- author_affinities: {"Orwell": 0.9, "Tolkien": 0.7}
- taste_embedding: weighted mean of rated book embeddings
- taste_cluster: LLM-generated label ("Literary Fiction Fan")
- confidence_score: 0--1 (saturates at ~20 data points)
- avg_rating_given, total_borrows, total_reviews

### Preference API Endpoints (8)

| Method | Endpoint | Layer | Description |
|---|---|---|---|
| GET | `/preferences/` | L1 | Get explicit preferences |
| PUT | `/preferences/` | L1 | Merge-update explicit preferences |
| POST | `/preferences/interactions` | L2 | Record a manual interaction |
| GET | `/preferences/interactions` | L2 | Get interaction history (filterable) |
| GET | `/preferences/interactions/stats` | L2 | Get aggregated stats |
| GET | `/preferences/taste` | L3 | Get current taste profile |
| POST | `/preferences/taste/recompute` | L3 | Trigger AI recomputation |
| GET | `/preferences/snapshot` | All | Unified view of all 3 layers |

### How Preferences Feed the Recommendation Engine

1. **Layer 1 (Explicit)** -> Content-Based Engine adds +0.15 for preferred
   authors, +0.10 for preferred genres; Knowledge Graph Engine matches tags;
   `discovery_mode` controls strategy weight distribution
2. **Layer 2 (Implicit)** -> Automatically recorded on borrow, return, and
   review; aggregated stats feed confidence scoring
3. **Layer 3 (Computed)** -> Content-Based Engine uses `taste_embedding` as the
   user profile vector and `author_affinities` / `genre_affinities` for
   secondary boosts

### Design Decision: Three Layers vs Flat Preferences

| Approach | Pros | Cons |
|---|---|---|
| **Flat row** (one table) | Simple schema | Mixes explicit and implicit; no AI enrichment; hard to extend |
| **Three-layer** (our choice) | Clean separation; extensible; AI-derived insights; recommendation engine gets 3 complementary signals | More tables; recomputation cost |

The requirement was intentionally vague -- "store User Preferences."  We chose
three layers because it provides the recommendation engine with richer signals
and makes the system extensible as new interaction types emerge.

---

## 9. Authentication & Authorisation

### JWT Flow

1. `POST /auth/signup` -- bcrypt hashes the password, INSERTs user
2. `POST /auth/login` -- bcrypt verifies password, returns JWT (HS256, 60 min TTL)
3. All other routes require `Authorization: Bearer <JWT>` header
4. `get_current_user` dependency decodes the token and loads the user from DB

### Security Choices

| Choice              | Rationale                                                              |
|---------------------|------------------------------------------------------------------------|
| **HS256** JWT       | Symmetric signing -- simple, fast, one secret to manage               |
| **bcrypt** hashing  | Industry standard, adaptive cost factor, resistant to rainbow tables  |
| **60 min** expiry   | Balances UX (not too short) with security (not open-ended)            |
| **OAuth2 bearer**   | FastAPI-native `OAuth2PasswordBearer` integrates into Swagger UI      |
| **Every route auth'd** | Defence in depth -- even read routes require a valid token         |

### Design Decision: No Refresh Tokens (Yet)

Tokens have a 60-minute TTL.  On `POST /auth/signout` the token's `jti` (JWT ID) is
written to Redis with a TTL equal to the token's remaining lifetime — this is the
**token revocation blacklist**.  Every authenticated request checks the blacklist
before granting access, so signed-out tokens are immediately invalidated server-side.

Further hardening that could be added:
- Refresh token rotation (sliding sessions)
- Role-based access control (RBAC)

---

## 10. Background Task Pipeline

LuminaLib uses **Celery + Redis** for all LLM-heavy background work.
Tasks run in a dedicated `worker` container, completely isolated from the API process.

### Architecture

```
  FastAPI (api)                Redis (broker)         Celery (worker)
  ─────────────                ──────────────         ───────────────
  task.delay(args)  ──push──►  task queue     ──pop──►  execute task
                               result backend ◄─store── update book/review
  GET /tasks/{id}  ◄──read───  result backend
```

The API **never waits** for LLM work to complete.  Every task ID is returned in a
response header so the caller can poll `GET /tasks/{task_id}` for status.

### Three Celery Tasks

| Celery task name | Triggered by | What it does | LLM calls |
|---|---|---|---|
| `llm.generate_book_summary` | `POST /books/` (upload), `PUT /books/{id}/content` (re-upload) | Extracts text → AI summary + embedding → `UPDATE book` | `generate_summary()` + `generate_embedding()` |
| `llm.analyze_review_sentiment` | `POST /books/{id}/reviews` | Classifies review sentiment → `UPDATE review.sentiment` | `analyze_sentiment()` |
| `llm.update_rolling_consensus` | `POST /books/{id}/reviews` | Reads all reviews → generates consensus → upserts `book_analyses` | `generate_review_consensus()` |

### Review Submission Dispatches 3 Actions

When a review is submitted, the API layer dispatches:
1. `analyze_review_sentiment.delay(review_id, text)` — Celery task; `X-Sentiment-Task-ID` header returned
2. `update_rolling_consensus.delay(book_id)` — Celery task; `X-Consensus-Task-ID` header returned
3. `record_interaction(...)` — inline call to `PreferenceService.record_interaction()`
   (Layer 2 implicit signal)

### Why Celery + Redis?

| Option              | Pros                                      | Cons                                   | Chosen? |
|---------------------|-------------------------------------------|----------------------------------------|:---:|
| FastAPI BackgroundTasks | Zero infra, simple                  | Lost on crash, no retry, no state      | No  |
| **Celery + Redis**  | Retry, task state, distributed workers, monitoring | Extra container, broker       | **Yes** |
| asyncio.create_task | Lighter                                   | Unstructured, not durable              | No  |

We chose Celery because:
1. **Durability** — tasks survive API process restarts (`task_acks_late=True`,
   `task_reject_on_worker_lost=True`)
2. **Retry logic** — each task retries up to 3 times with 60-second back-off on failure
3. **Observability** — `GET /tasks/{task_id}` exposes `PENDING → STARTED → SUCCESS / FAILURE`
   state transitions backed by the Redis result backend
4. **Isolation** — LLM calls (which can take 10–30 s) never block the API event loop

### Task Implementation Pattern

Each Celery task is a thin synchronous wrapper that calls an async coroutine via
`asyncio.run()`, which is safe because each Celery worker process has its own event loop:

```python
@celery_app.task(bind=True, name="llm.generate_book_summary", max_retries=3,
                 task_acks_late=True, task_reject_on_worker_lost=True)
def generate_book_summary(self, book_id: str, mime_type: str) -> None:
    try:
        asyncio.run(generate_book_summary_task(book_id, mime_type))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)
```

The async coroutine opens its own `AsyncSession` — independent of the request-scoped
session — so there is no "session already closed" risk.

### NullPool Engine for Celery Workers

Each `asyncio.run()` call inside a Celery task creates a **new event loop** and
destroys it when it exits.  asyncpg's default connection pool holds connections
that are bound to the event loop that created them.  On the second task executed
by the same worker process, the pool tries to reuse those connections — but they
are attached to the now-closed loop, causing:

```
Task got Future <Future pending cb=[Protocol._on_waiter_completed()]>
attached to a different loop
```

**Fix:** A dedicated `worker_engine` with `NullPool` is created in
`app/infrastructure/database/connection.py`:

```python
from sqlalchemy.pool import NullPool

# NullPool: opens a fresh DB connection per session, closes it immediately.
# No connections are held between asyncio.run() calls → no loop-mismatch error.
worker_engine = create_async_engine(
    settings.database_url, echo=False, future=True, poolclass=NullPool
)
worker_session_maker = async_sessionmaker(
    worker_engine, class_=AsyncSession, expire_on_commit=False
)
```

`app/services/background_tasks.py` imports `worker_session_maker` (aliased as
`async_session_maker`) so that every Celery task coroutine gets a clean
connection regardless of which event loop is currently active.

The FastAPI process continues to use the pooled `engine` / `async_session_maker`
unaffected — only the worker-side code uses `NullPool`.

### Task Status Endpoint

```
GET /tasks/{task_id}
→ { "task_id": "...", "status": "SUCCESS", "result": null, "error": null }
```

States: `PENDING → STARTED → SUCCESS | FAILURE | RETRY`

### Cache-First Analysis Reads

`ReviewService.get_book_analysis()` uses a **cache-first** strategy:

1. Read from `book_analyses` table (populated by `llm.update_rolling_consensus`)
2. If cached result exists → return immediately (zero LLM calls)
3. If no cached result → fall back to on-demand LLM consensus generation

This means `GET /books/{id}/analysis` is instant for books that have had reviews
submitted, because the Celery task pre-populates the cache.

---

## 11. Deployment Topology

### Docker Compose Services

| Service | Image | Internal Port | Host Port | Healthcheck |
|---|---|---|---|---|
| **api** | Built from Dockerfile | 5223 | 5223 | -- |
| **worker** | Built from Dockerfile | -- | -- | -- (Celery worker process) |
| **db** | postgres:15-alpine | 5432 | 5433 | `pg_isready -U lumina -d luminalib` |
| **redis** | redis:7-alpine | 6379 | 6380 | `redis-cli ping` |
| **minio** | minio/minio | 9000, 9001 | 9000, 9002 | `mc ready local` |
| **ollama** | ollama/ollama | 11434 | 11434 | `ollama list` (start_period: 30s) |
| **ollama-init** | ollama/ollama | -- | -- | one-shot: pulls `llama3.2:1b` into shared volume, exits 0 |

Both `api` and `worker` gate on health conditions before starting:

```yaml
depends_on:
  db:           { condition: service_healthy }                  # waits for pg_isready
  redis:        { condition: service_healthy }                  # waits for redis-cli ping
  minio:        { condition: service_healthy }                  # waits for mc ready local
  ollama-init:
    condition: service_completed_successfully
    required: false   # soft dep — api/worker start even if pull fails
```

`ollama-init` is a one-shot container: it waits for the `ollama` service healthcheck
to pass, then runs `ollama pull llama3.2:1b` against `OLLAMA_HOST=http://ollama:11434`.
The pulled model is stored in the shared `ollama_data` named volume so subsequent
`docker compose up` runs skip the download.

**Single-command startup — no flags needed:**

```bash
docker compose up -d --build   # starts everything: api, worker, db, redis, minio, ollama, ollama-init
```

**Auto-fallback chain (three layers):**
1. `required: false` — `api`/`worker` start even if `ollama-init` never ran
2. `|| echo` in `ollama-init` entrypoint — a failed model pull exits 0 and never blocks the stack
3. `LlamaLLMService` code — every Ollama HTTP call is wrapped in `try/except`; any failure
   returns `""` and the public method falls back to `self._mock.<method>()` automatically

The stack **never goes down** due to an LLM failure.

### Design Decision: Separate Host Ports

All services bind to **non-standard host ports** (5223, 5433, 6380) to avoid
conflicts with any locally-running PostgreSQL (5432), Redis (6379), or other
services.  Internal container ports remain standard.

### Resilience

- `restart: on-failure` on `api`, `worker`, and `ollama` for automatic recovery
- Celery worker uses `task_acks_late=True` + `task_reject_on_worker_lost=True` —
  tasks are re-queued if the worker crashes mid-execution
- GPU-ready: commented-out `deploy.resources.reservations.devices` block for
  NVIDIA GPU pass-through on the `ollama` service
- Named volumes (`postgres_data`, `minio_data`, `ollama_data`) survive
  `docker compose down`; only wiped with `-v`

---

## 12. SOLID Principles Mapping

### S -- Single Responsibility

| Class / Module               | Single Responsibility                                        |
|------------------------------|--------------------------------------------------------------|
| `BookService`                | Book CRUD + file ingestion orchestration                     |
| `ReviewService`              | Review lifecycle + consensus generation + cache-first reads  |
| `AuthService`                | Signup, login, token management                              |
| `MLRecommendationService`    | 5-strategy hybrid scoring + diversity + cold-start           |
| `PreferenceService`          | 3-layer preference orchestration + taste recomputation       |
| `ContentBasedEngine`         | Cosine similarity + preference boosting                      |
| `CollaborativeFilteringEngine` | Jaccard + item-item co-occurrence                         |
| `PopularityEngine`           | Wilson score + time-decay + trending                         |
| `KnowledgeGraphEngine`       | Author co-read graph + tag matching                          |
| `llm_tasks.py`               | Celery task wrappers — dispatch LLM work to worker process   |
| `prompts.py`                 | Prompt template definition + versioning                      |
| `dependencies.py`            | Object graph wiring (Composition Root)                       |

### O -- Open/Closed

New providers and strategies are added **without modifying existing code**:

- To add a new LLM provider (e.g., Anthropic):
  1. Create `AnthropicLLMService(ILLMService)` in `llm/services.py`
  2. Add one `elif` branch in `dependencies.py`
  3. All existing code remains untouched.

- To add a new recommendation strategy (e.g., trending-topics):
  1. Create `TrendingTopicsEngine` class
  2. Add it to `MLRecommendationService.__init__()` and the run pipeline
  3. All existing strategies are unchanged.

### L -- Liskov Substitution

Every `ILLMService` implementation (`Mock`, `Llama`, `OpenAI`) is fully
substitutable -- callers receive `ILLMService` and never know which concrete
class is behind it.  The same applies to `IStorageService` and all repository
interfaces.

**Live-tested proof:** 11/11 swap tests pass -- `STORAGE_BACKEND` and
`LLM_PROVIDER` swapped at runtime with zero code changes.

### I -- Interface Segregation

Instead of a single "god interface", the domain defines **10 focused
interfaces**:

| Interface | Methods | Used by |
|---|---|---|
| `IUserRepository` | create, get_by_id, get_by_email, get_by_username, update | AuthService |
| `IBookRepository` | create, get_by_id, list_all, count, update, update_content, delete | BookService, RecommService |
| `IBorrowRepository` | create, get_active_borrow, return_book, get_user_borrows, has_user_borrowed, get_all_borrows, get_book_borrows | Routes, RecommService |
| `IReviewRepository` | create, get_by_book, get_by_user_and_book, get_all_reviews | ReviewService, RecommService |
| `IUserPreferenceRepository` | get_or_create, update | PreferenceService |
| `IUserInteractionRepository` | record, get_user_interactions, get_interaction_stats | PreferenceService |
| `IUserTasteProfileRepository` | get_or_create, update | PreferenceService, RecommService |
| `IStorageService` | save_file, get_file, delete_file | BookService |
| `ILLMService` | generate_summary, generate_embedding, generate_review_consensus, analyze_sentiment | BookService, ReviewService, PreferenceService, BG tasks |
| `IRecommendationService` | get_recommendations, get_user_recommendations | Routes |

No class is forced to implement methods it doesn't use.

### D -- Dependency Inversion

High-level modules (`BookService`, `ReviewService`, `PreferenceService`,
`MLRecommendationService`) depend on **abstractions** (`IBookRepository`,
`ILLMService`, `IUserPreferenceRepository`), never on concrete implementations.
The Composition Root in `dependencies.py` wires concrete classes at runtime.

**Evidence:** `book_service.py` imports only `IBookRepository, ILLMService,
IStorageService`.  It never imports `LocalStorageService`, `S3StorageService`,
`LlamaLLMService`, or any other concrete class.

---

## 13. Key Trade-Offs & Rationale

| Decision                               | Alternative Considered         | Why We Chose This                                                                                      |
|----------------------------------------|--------------------------------|--------------------------------------------------------------------------------------------------------|
| **FastAPI** over Django/Flask          | Django REST, Flask             | Native async, Pydantic-first validation, auto OpenAPI docs, built-in DI                               |
| **SQLAlchemy 2.x async** over Tortoise | Tortoise ORM, raw asyncpg     | Mature ecosystem, migration support, 2.x async is production-stable                                    |
| **Celery + Redis** over BackgroundTasks | FastAPI BackgroundTasks       | Tasks survive crashes (acks_late), support retries, expose state (`PENDING→SUCCESS`), and run in an isolated `worker` container — critical for 10–30 s LLM calls |
| **ARRAY(Float)** over pgvector         | pgvector, Pinecone, Qdrant     | No extra extension; sufficient for <10k books; pgvector migration requires only IBookRepository change |
| **5-strategy hybrid** over single      | Content-based only, CF only    | Hybrid avoids filter bubble; cold-start cascade handles new users; discovery mode gives user control   |
| **3-layer preferences** over flat row  | Single user_preferences table  | Separates explicit/implicit/computed signals; richer recommendation input; extensible                  |
| **Dataclass entities** over ORM models | Use ORM models as entities     | Keeps domain layer ORM-free; unit tests don't need a database; clear persistence boundary             |
| **bcrypt + HS256** JWT                 | Argon2 + RS256                 | bcrypt is battle-tested; HS256 is simpler (one secret vs key pair) for a single-service deployment    |
| **Ollama** over vLLM / TGI            | vLLM, TGI, llama.cpp           | Simplest Docker setup; REST API; model management built-in; swap is one new class                      |
| **MinIO** for S3 compat               | LocalStack, Ceph               | Purpose-built S3 clone; tiny Docker image; production MinIO works at scale                            |
| **Non-standard host ports**            | Standard ports                 | Avoids conflicts with local PostgreSQL/Redis; safe for development machines                            |
| **Cascade delete on books**            | Soft delete, manual cleanup    | Ensures referential integrity; no orphan rows; appropriate for this scale                              |
| **Dynamic lazy-load for interactions** | Eager selectin (like others)   | Event log can grow unbounded; dynamic returns a query, not all rows                                    |
| **Immutable prompt templates**         | String constants, f-strings    | Versioned, auditable, render to both message-list and flat-string formats; frozen dataclass prevents mutation |

---

## 14. Code Hygiene

| Practice                     | Tool / Convention                                    |
|------------------------------|------------------------------------------------------|
| Formatting                   | `black` (line length 100) + `isort` (profile=black)  |
| Type checking                | `mypy` (strict: `disallow_untyped_defs = true`)      |
| Linting                      | `flake8`                                              |
| Test runner                  | `pytest` + `pytest-asyncio` (`asyncio_mode = auto`)  |
| Dependency management        | `requirements.txt` (prod, 20 packages) + `requirements-dev.txt` (8 packages) |
| Config                       | `pydantic-settings` -- validated, typed `Literal` enums, from `.env` |
| Async everywhere             | No blocking I/O in any handler or service             |
| One-way dependencies         | Domain <- Service <- Infrastructure (never reversed)  |
| Explicit DI                  | All wiring in `dependencies.py`, no hidden globals    |
| Module docstrings            | Every `.py` file has a module-level docstring          |
| Type annotations             | Every function signature is annotated                  |
| Pydantic v2                  | `model_config = ConfigDict(...)` -- no deprecated `class Config` |

### Production Dependencies (requirements.txt -- 20 packages)

| Category | Packages |
|---|---|
| **Web framework** | `fastapi`, `uvicorn[standard]`, `python-multipart` |
| **Database** | `sqlalchemy`, `asyncpg`, `greenlet` |
| **Validation & Config** | `pydantic[email]`, `pydantic-settings`, `python-dotenv` |
| **Auth** | `python-jose[cryptography]`, `passlib[bcrypt]`, `bcrypt` |
| **LLM / GenAI** | `httpx`, `openai` |
| **ML** | `numpy`, `scikit-learn` |
| **Storage** | `boto3`, `aiofiles` |
| **PDF** | `PyMuPDF` |
| **HTTP client** | `requests` (integration tests) |

### Dev Dependencies (requirements-dev.txt -- 8 packages)

`pytest`, `pytest-asyncio`, `pytest-cov`, `httpx`, `black`, `isort`, `mypy`, `flake8`

### Makefile Targets

| Target | Command | Purpose |
|---|---|---|
| `make up` | `docker compose up -d --build` | Start all containers (api, worker, db, redis, minio, ollama, ollama-init) |
| `make down` | `docker compose down -v` | Stop + wipe data |
| `make test` | `pytest tests/ -v` | Run unit tests |
| `make lint` | `mypy app/` + `flake8 app/` | Static analysis |
| `make format` | `black app/ tests/` + `isort app/ tests/` | Auto-format |
| `make clean` | Remove `__pycache__`, `.pyc`, caches | Cleanup |

---

*Last updated: February 2026*
