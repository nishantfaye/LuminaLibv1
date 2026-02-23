# LuminaLib — Intelligent, Content-Aware Library System

---

## Application Context

Modern libraries manage **thousands of digital books** but still rely on
manual cataloguing, hand-written summaries, and simple keyword search.
Readers have no easy way to discover new titles that match their taste, and
librarians have no automated tool to gauge how a book is being received.

**LuminaLib** solves this by building an intelligent backend that:

1. **Ingests real book files** (TXT, PDF) — not just metadata titles.
2. **Generates AI-powered summaries and semantic embeddings** automatically
   in the background the moment a book is uploaded.
3. **Drives personalised, ML-based recommendations** — "readers who liked
   this also liked …" — powered by cosine similarity over LLM-generated
   embeddings.
4. **Synthesises reader sentiment** by running GenAI across all reviews to
   produce a rolling consensus analysis ("What do readers really think about
   this book?").
5. **Enforces a complete borrow-before-review lifecycle** — only users who
   have actually borrowed a book can review it, ensuring review authenticity.

Everything is wrapped in a **Clean Architecture** so that the LLM provider
(Mock → Ollama → OpenAI), the storage backend (local disk → S3 / MinIO), and
the database can each be swapped with a single environment variable — zero
code changes.

> **In short:** LuminaLib turns a basic CRUD library API into a smart,
> content-aware platform that reads your books, understands them, and helps
> readers find what they'll love.

---

## Table of Contents

1. [Application Context](#application-context)
2. [Features](#features)
3. [Run the Application — One Command](#run-the-application--one-command)
4. [Prerequisites](#prerequisites)
5. [What Gets Started](#what-gets-started)
6. [API Endpoints](#api-endpoints)
7. [Usage Walkthrough (curl)](#usage-walkthrough-curl)
8. [Configuration & Environment Variables](#configuration--environment-variables)
9. [Swappable Components — Architecture Deep-Dive](#swappable-components--architecture-deep-dive)
    - [Tutorial A: Switch Storage Local → S3 / MinIO](#tutorial-a-switch-file-storage-from-local-disk--s3--minio)
    - [Tutorial B: Swap Llama 3 → OpenAI](#tutorial-b-swap-llama-3-for-openai-without-rewriting-business-logic)
10. [Project Structure](#project-structure)
11. [Running Tests](#running-tests)
12. [Code Quality](#code-quality)
13. [Technology Stack](#technology-stack)
14. [Architecture Documentation](#architecture-documentation)

---

## Features

### Core Library Management
- **Book upload with file ingestion** — upload TXT or PDF files; the system
  stores the file, computes a SHA-256 content hash (deduplication), detects
  MIME type, and extracts raw text (PyMuPDF for PDFs).
- **Full CRUD** — create, read (with pagination), update, and delete books.
- **Borrow / Return lifecycle** — users borrow books and return them; the
  system tracks active borrows and history per user.

### AI & Machine Learning
- **Automatic AI summaries** — when a book is uploaded, a background task
  sends the extracted text to the configured LLM and saves a generated
  summary back to the database asynchronously.
- **Semantic embeddings** — the LLM generates a dense vector embedding for
  every book, stored in PostgreSQL and used for similarity search.
- **Book-to-book recommendations** — given a book, find the most similar
  books using cosine similarity over embeddings (scikit-learn).
- **Personalised user recommendations** — builds a user profile embedding
  from all borrowed books, ranks unseen books by similarity, and boosts
  books from the user's preferred authors.
- **Cold-start handling** — users with no borrow history receive recent
  books with an honest score of 0.0.

### GenAI Review Analysis
- **Authenticated reviews** — only users who have borrowed a book can submit
  a review (1-5 star rating + text).  One review per user per book enforced
  via unique constraint.
- **Async sentiment analysis** — each review's sentiment (positive / neutral /
  negative) is analysed by the LLM in a background task.
- **Rolling consensus summary** — every new review triggers regeneration of
  an AI-written consensus that aggregates all reviews into a single insight
  (e.g. *"Readers praise the practical examples but note the dated Java syntax"*).

### Authentication & Security
- **JWT-based auth** — signup → login → receive Bearer token.  HS256 signing
  with configurable expiry (default 60 min).
- **bcrypt password hashing** — industry-standard, adaptive-cost hashing.
- **Every route protected** — all endpoints except `/health`, `/auth/signup`,
  and `/auth/login` require a valid JWT.  Unauthenticated requests receive
  `401 Unauthorized`.

### Infrastructure Flexibility (Swappable Backends)
- **3 LLM providers** — `MockLLMService` (tests / CI), `LlamaLLMService`
  (Ollama, on-prem), `OpenAILLMService` (cloud).  Switch via one env var.
- **2 storage backends** — `LocalStorageService` (filesystem, SHA-256
  sub-directories) and `S3StorageService` (MinIO or AWS).  Switch via one
  env var.
- **Interface-driven DI** — every infrastructure component implements an
  abstract interface; the Composition Root in `dependencies.py` wires the
  concrete class at runtime based on config.

### Production-Ready
- **Fully async** — `async/await` everywhere, from HTTP handlers to DB
  queries (`asyncpg`) to LLM calls (`httpx`).
- **Background tasks** — heavy LLM work runs in a dedicated **Celery + Redis**
  worker, completely isolated from the API process, with retry and task-state
  tracking (`GET /tasks/{task_id}`).
- **Docker Compose** — one command spins up 7 health-checked containers.
- **Auto-generated API docs** — Swagger UI + ReDoc available instantly.
- **10 unit tests** — pytest + pytest-asyncio with `AsyncMock` for every
  infrastructure dependency.

---

## Run the Application — One Command

### Prerequisites

| Requirement       | Version    | Notes                                            |
|-------------------|------------|--------------------------------------------------|
| **Docker Desktop** | ≥ 24.x   | Includes Docker Compose v2                       |
| *Optional:* Python | ≥ 3.10   | Only needed to run tests locally outside Docker  |
| *Optional:* Make   | any       | For `make format`, `make lint`, `make test`      |

> **No Python install is needed to run the application** — everything runs
> inside Docker containers.

### Step 1 — Clone & Start

```bash
git clone <repo-url>
cd luminaLib
docker compose up --build
```

That's it. Docker will:

1. Build the Python 3.11 API image from the `Dockerfile`.
2. Pull `postgres:15-alpine`, `redis:7-alpine`, `minio/minio:latest`, and
   `ollama/ollama:latest`.
3. Start all 7 containers with healthchecks.
4. Automatically create the database tables on first boot.
5. Begin serving the API on **http://localhost:5223**.

> **Tip:** Add `-d` to run in the background: `docker compose up --build -d`

### Step 2 — Verify It's Running

```bash
curl http://localhost:5223/health
# → {"status": "healthy"}
```

### Step 3 — Open the Docs

| What                   | URL                                                    |
|------------------------|--------------------------------------------------------|
| **Swagger UI**         | [http://localhost:5223/docs](http://localhost:5223/docs) |
| **ReDoc**              | [http://localhost:5223/redoc](http://localhost:5223/redoc) |
| **MinIO Console**      | [http://localhost:9002](http://localhost:9002) *(minioadmin / minioadmin)* |

### Step 4 — Create Your First User

```bash
# Sign up
curl -X POST http://localhost:5223/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","email":"alice@example.com","password":"secret123"}'

# Login — save the JWT token
TOKEN=$(curl -s -X POST http://localhost:5223/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"alice@example.com","password":"secret123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "Your token: $TOKEN"
```

You're ready to use every endpoint.  See the [Usage Walkthrough](#usage-walkthrough-curl) below.

### Alternative: `start.sh` helper script

```bash
chmod +x start.sh
./start.sh
```

This script checks Docker is running, creates `.env` if missing, builds all
containers, waits for the healthcheck, and prints the status.

### Stopping the Application

```bash
# Stop and keep data (volumes preserved)
docker compose down

# Stop and wipe all data (clean slate)
docker compose down -v --remove-orphans
```

---

## What Gets Started

`docker compose up` launches **7 containers** that form the complete stack:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Docker Compose Stack                               │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐                   │
│  │   api        │  │   db         │  │   redis           │                  │
│  │  FastAPI     │  │  PostgreSQL  │  │  Redis 7          │                  │
│  │  :5223       │  │  :5433→5432  │  │  :6380→6379       │                  │
│  │  Python 3.11 │  │  15-alpine   │  │  Celery broker    │                  │
│  └──────┬───────┘  └──────────────┘  └──────────────────┘                  │
│         │                                                                   │
│  ┌──────┴─────────────────────────────────────────────────┐                │
│  │                                                         │                │
│  ▼               ▼               ▼              ▼          │                │
│  ┌─────────────┐ ┌─────────────┐ ┌───────────┐ ┌────────┐ │                │
│  │   minio      │ │   ollama     │ │  worker   │ │ollama  │ │                │
│  │  S3 Storage  │ │  Local LLM   │ │  Celery   │ │ -init  │ │                │
│  │  :9000 API   │ │  :11434      │ │  worker   │ │ pulls  │ │                │
│  │  :9002 UI    │ │  llama3.2:1b │ │  process  │ │ model  │ │                │
│  └─────────────┘ └─────────────┘ └───────────┘ └────────┘ │                │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Service          | Image                  | Host Port       | Purpose                                    | Healthcheck          |
|------------------|------------------------|-----------------|--------------------------------------------|----------------------|
| **api**          | `python:3.11-slim`     | **5223**        | FastAPI application server                 | `GET /health`        |
| **db**           | `postgres:15-alpine`   | **5433**        | PostgreSQL relational database             | `pg_isready`         |
| **redis**        | `redis:7-alpine`       | **6380**        | Celery broker + JWT revocation store       | `redis-cli ping`     |
| **minio**        | `minio/minio:latest`   | **9000 / 9002** | S3-compatible object storage               | `mc ready local`     |
| **ollama**       | `ollama/ollama:latest` | **11434**       | Local LLM inference (llama3.2:1b)          | `ollama list`        |
| **worker**       | `python:3.11-slim`     | —               | Celery worker for LLM background tasks     | —                    |
| **ollama-init**  | `ollama/ollama:latest` | —               | One-shot: pulls llama3.2:1b on first boot  | —                    |

> Host ports are intentionally **non-standard** (5223, 5433, 6380) to avoid
> conflicts with any locally-running PostgreSQL, Redis, or other services.

---

## API Endpoints

Every endpoint except `/health`, `/auth/signup`, and `/auth/login` requires
a JWT token in the `Authorization: Bearer <token>` header.

### Authentication (5 endpoints)

| Method | Endpoint          | Description               | Auth |
|--------|-------------------|---------------------------|:----:|
| POST   | `/auth/signup`    | Register a new user       | ✗    |
| POST   | `/auth/login`     | Login → JWT access token  | ✗    |
| GET    | `/auth/profile`   | Get current user profile  | ✓    |
| PUT    | `/auth/profile`   | Update profile            | ✓    |
| POST   | `/auth/signout`   | Sign out                  | ✓    |

### Books — CRUD & File Upload (5 endpoints)

| Method | Endpoint           | Description                                | Auth |
|--------|--------------------|------------------------------------------  |:----:|
| POST   | `/books/`          | Upload book file + metadata                | ✓    |
| GET    | `/books/`          | List all books (paginated: `?page=&limit=`)| ✓    |
| GET    | `/books/{id}`      | Get single book by ID                      | ✓    |
| PUT    | `/books/{id}`      | Update book title / author                 | ✓    |
| DELETE | `/books/{id}`      | Delete book and its stored file            | ✓    |

### Borrow / Return (2 endpoints)

| Method | Endpoint                | Description                  | Auth |
|--------|-------------------------|------------------------------|:----:|
| POST   | `/books/{id}/borrow`    | Borrow a book                | ✓    |
| POST   | `/books/{id}/return`    | Return a borrowed book       | ✓    |

### Reviews & GenAI Analysis (2 endpoints)

| Method | Endpoint                  | Description                                        | Auth |
|--------|---------------------------|----------------------------------------------------|:----:|
| POST   | `/books/{id}/reviews`     | Submit a review *(must have borrowed the book)*    | ✓    |
| GET    | `/books/{id}/analysis`    | GenAI-aggregated review consensus for the book     | ✓    |

### ML Recommendations (2 endpoints)

| Method | Endpoint                          | Description                                   | Auth |
|--------|-----------------------------------|-----------------------------------------------|:----:|
| GET    | `/books/{id}/recommendations`     | Find similar books (cosine similarity)        | ✓    |
| GET    | `/recommendations`                | Personalised recommendations for current user | ✓    |

### System (1 endpoint)

| Method | Endpoint   | Description          | Auth |
|--------|------------|----------------------|:----:|
| GET    | `/health`  | Healthcheck          | ✗    |

**Total: 17 endpoints**

---

## Usage Walkthrough (curl)

A complete end-to-end flow using `curl`.  Replace `localhost:5223` with your
host if different.

### 1. Register & Login

```bash
# Sign up
curl -s -X POST http://localhost:5223/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","email":"alice@example.com","password":"secret123"}'

# Login — capture the JWT token
TOKEN=$(curl -s -X POST http://localhost:5223/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"alice@example.com","password":"secret123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

### 2. Upload a Book

```bash
curl -s -X POST http://localhost:5223/books/ \
  -H "Authorization: Bearer $TOKEN" \
  -F "title=Clean Code" \
  -F "author=Robert C. Martin" \
  -F "file=@mybook.txt;type=text/plain"
```

The response returns **immediately** with `"summary": null`.  A background task
generates the AI summary and embedding asynchronously — check again in a few
seconds and the `summary` field will be populated.

### 3. List Books (paginated)

```bash
curl -s "http://localhost:5223/books/?page=1&limit=10" \
  -H "Authorization: Bearer $TOKEN"
```

### 4. Borrow → Review → Analysis

```bash
BOOK_ID="<uuid-from-step-2>"

# Borrow the book
curl -s -X POST "http://localhost:5223/books/$BOOK_ID/borrow" \
  -H "Authorization: Bearer $TOKEN"

# Submit a review (only works after borrowing)
curl -s -X POST "http://localhost:5223/books/$BOOK_ID/reviews" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"rating":5,"text":"Excellent guide to software craftsmanship!"}'

# Get the GenAI analysis of all reviews
curl -s "http://localhost:5223/books/$BOOK_ID/analysis" \
  -H "Authorization: Bearer $TOKEN"
```

### 5. Get Personalised Recommendations

```bash
# Recommendations for the current user (based on borrow history)
curl -s "http://localhost:5223/recommendations" \
  -H "Authorization: Bearer $TOKEN"

# Recommendations similar to a specific book
curl -s "http://localhost:5223/books/$BOOK_ID/recommendations" \
  -H "Authorization: Bearer $TOKEN"
```

### 6. Return a Book

```bash
curl -s -X POST "http://localhost:5223/books/$BOOK_ID/return" \
  -H "Authorization: Bearer $TOKEN"
```

### 7. Profile & Signout

```bash
# View your profile
curl -s "http://localhost:5223/auth/profile" \
  -H "Authorization: Bearer $TOKEN"

# Sign out
curl -s -X POST "http://localhost:5223/auth/signout" \
  -H "Authorization: Bearer $TOKEN"
```

---

## Configuration & Environment Variables

All configuration is via environment variables (loaded from `.env` or
overridden in `docker-compose.yml`).  Docker Compose pre-configures
everything — you only need to touch these if customising.

| Variable                      | Default                                                       | Description                             |
|-------------------------------|---------------------------------------------------------------|-----------------------------------------|
| `DATABASE_URL`                | `postgresql+asyncpg://lumina:lumina@localhost:5432/luminalib`  | Async PostgreSQL DSN                    |
| `STORAGE_BACKEND`             | `local`                                                       | `local` or `s3`                         |
| `STORAGE_PATH`                | `./storage`                                                   | Root directory for local file storage   |
| `S3_BUCKET`                   | `luminalib-books`                                             | S3 / MinIO bucket name                  |
| `S3_ENDPOINT_URL`             | *(empty)*                                                     | Custom S3 endpoint (e.g. MinIO URL)     |
| `S3_ACCESS_KEY`               | *(empty)*                                                     | S3 / MinIO access key                   |
| `S3_SECRET_KEY`               | *(empty)*                                                     | S3 / MinIO secret key                   |
| `LLM_PROVIDER`                | `mock`                                                        | `mock`, `llama`, or `openai`            |
| `LLM_BASE_URL`                | `http://localhost:11434`                                      | Ollama server URL                       |
| `LLM_MODEL`                   | `llama3.2:1b`                                                 | Model tag for Ollama                    |
| `LLM_API_KEY`                 | *(empty)*                                                     | API key for OpenAI provider             |
| `REDIS_URL`                   | `redis://localhost:6379`                                      | Redis connection URL                    |
| `SECRET_KEY`                  | `dev-secret-key`                                              | JWT signing secret (**change in prod**) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60`                                                          | JWT token lifetime in minutes           |

> Inside Docker Compose, `DATABASE_URL`, `REDIS_URL`, `S3_ENDPOINT_URL`,
> `S3_ACCESS_KEY`, and `S3_SECRET_KEY` are automatically overridden to use
> container hostnames and credentials.

---

## Swappable Components — Architecture Deep-Dive

LuminaLib's Clean Architecture enforces **strict dependency inversion**:
every service and route depends only on abstract interfaces — never on
concrete classes.  Swapping an infrastructure component requires **changing
one environment variable** and restarting the container.  Zero business
logic changes, zero code edits, zero redeployments of other services.

### Why It Works: The Three-Layer Pattern

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Business Logic Layer                              │
│                                                                      │
│  BookService ─────────► IStorageService    (abstract interface)      │
│  ReviewService ───────► ILLMService        (abstract interface)      │
│  RecommendationService  IRecommendationService                      │
│  PreferenceService                                                   │
│                                                                      │
│          ▲  depends on abstractions only — never on concrete classes │
├──────────┼───────────────────────────────────────────────────────────┤
│          │          Composition Root (dependencies.py)                │
│          │                                                           │
│          │   def get_storage_service() -> IStorageService:           │
│          │       if settings.storage_backend == "local":             │
│          │           return LocalStorageService(...)                  │
│          │       elif settings.storage_backend == "s3":              │
│          │           return S3StorageService(...)                     │
│          │                                                           │
│          │   def get_llm_service() -> ILLMService:                   │
│          │       if settings.llm_provider == "mock":                 │
│          │           return MockLLMService()                          │
│          │       elif settings.llm_provider == "llama":              │
│          │           return LlamaLLMService(...)                      │
│          │       elif settings.llm_provider == "openai":             │
│          │           return OpenAILLMService(...)                     │
│          │                                                           │
├──────────┼───────────────────────────────────────────────────────────┤
│          │          Infrastructure Layer (adapters)                   │
│          │                                                           │
│          ├──► LocalStorageService    implements  IStorageService      │
│          ├──► S3StorageService       implements  IStorageService      │
│          ├──► MockLLMService         implements  ILLMService          │
│          ├──► LlamaLLMService        implements  ILLMService          │
│          └──► OpenAILLMService       implements  ILLMService          │
└──────────────────────────────────────────────────────────────────────┘
```

**Key design points:**

| Principle | How LuminaLib applies it |
|---|---|
| **Dependency Inversion** | `BookService` depends on `IStorageService`, never on `LocalStorageService` or `S3StorageService` |
| **Interface Segregation** | `IStorageService` has exactly 3 methods (`save_file`, `get_file`, `delete_file`); `ILLMService` has exactly 4 (`generate_summary`, `generate_embedding`, `generate_review_consensus`, `analyze_sentiment`) |
| **Single Composition Root** | All wiring happens in one file: `app/core/dependencies.py` |
| **Config-driven factory** | `pydantic-settings` reads `STORAGE_BACKEND` and `LLM_PROVIDER` from env vars with typed `Literal` validation |
| **Graceful fallback** | `LlamaLLMService` and `OpenAILLMService` both fall back to `MockLLMService` on failure |

---

### Swap Matrix

| Component | Env Var | Options | Lines to Change |
|---|---|---|:---:|
| **File Storage** | `STORAGE_BACKEND` | `local` (filesystem) · `s3` (MinIO or AWS) | **1** |
| **LLM Provider** | `LLM_PROVIDER` | `mock` · `llama` (Ollama) · `openai` | **1** |
| **LLM Model** | `LLM_MODEL` | Any Ollama-compatible model tag | **1** |
| **Database** | `DATABASE_URL` | Any asyncpg-compatible PostgreSQL | **1** |

---

### Swap 1 — File Storage: Local Disk → AWS S3 / MinIO

**What changes:** one line in `docker-compose.yml` (or `.env`).

```yaml
# Before:
- STORAGE_BACKEND=local

# After:
- STORAGE_BACKEND=s3
```

**Restart:**
```bash
docker compose up -d --build api
```

**What happens internally:**

1. `config.py` reads `STORAGE_BACKEND=s3` into `settings.storage_backend`
2. `dependencies.py → get_storage_service()` returns `S3StorageService` instead of `LocalStorageService`
3. `BookService.create_book()` calls `self.storage_service.save_file()` — same interface, different backend
4. The file is uploaded to MinIO at `http://minio:9000/luminalib-books/`
5. **No change** to `BookService`, `routes.py`, `background_tasks.py`, or any other file

**Browse uploaded files:** [http://localhost:9002](http://localhost:9002) (login: `minioadmin` / `minioadmin`)

**Live-tested proof:**
```
STORAGE_BACKEND=s3 → Book uploaded to MinIO
   File stored at S3 key: 9a/9af6f29e...swap_s3-storage.txt
STORAGE_BACKEND=local → Restored, book uploaded to ./storage/
   Zero code changes between the two configurations.
```

---

### Swap 2 — LLM: Llama 3 (Ollama) → OpenAI

**What changes:** two lines (provider + API key).

```yaml
# Before:
- LLM_PROVIDER=llama
- LLM_BASE_URL=http://ollama:11434
- LLM_MODEL=llama3.2:1b

# After:
- LLM_PROVIDER=openai
- LLM_API_KEY=sk-your-openai-api-key-here
```

**Restart:**
```bash
docker compose up -d --build api
```

**What happens internally:**

1. `config.py` reads `LLM_PROVIDER=openai`
2. `dependencies.py → get_llm_service()` returns `OpenAILLMService` instead of `LlamaLLMService`
3. All downstream consumers (`BookService`, `ReviewService`, `PreferenceService`, background tasks) call the exact same interface methods: `generate_summary()`, `generate_embedding()`, `analyze_sentiment()`, `generate_review_consensus()`
4. Under the hood, `OpenAILLMService` uses `openai.AsyncOpenAI` with `gpt-4o-mini` for chat and `text-embedding-3-small` for embeddings
5. If the API key is missing or invalid, it **gracefully falls back** to `MockLLMService` — the app never crashes

**Affected code: none.**  The business logic files (`book_service.py`, `review_service.py`, `background_tasks.py`, `recommendation.py`, `preference_service.py`) have **zero awareness** of which LLM provider is active.

---

### Swap 3 — LLM: Llama 3 → Mock (for CI / Testing)

**What changes:** one line.

```yaml
# Before:
- LLM_PROVIDER=llama

# After:
- LLM_PROVIDER=mock
```

**Live-tested proof:**
```
LLM_PROVIDER=mock → Summary: "Summary: [uid:53dc9d40] This is a test book..."
   (MockLLMService returns deterministic, word-based summary — instant, no GPU)
LLM_PROVIDER=llama → Summary: "Here is a summary of the book content..."
   (LlamaLLMService calls Ollama API — real LLM inference)
```

The Mock provider is useful for:
- **CI/CD pipelines** — no Ollama container needed, tests run in milliseconds
- **Development** — instant responses, deterministic output for debugging
- **Demo environments** — works anywhere without GPU or API key

---

### Swap 4 — LLM Model: Llama 3.2 1B → Any Ollama Model

**What changes:** one line.

```yaml
# Before:
- LLM_MODEL=llama3.2:1b

# After (examples):
- LLM_MODEL=llama3.2:3b       # Larger Llama model
- LLM_MODEL=mistral:7b        # Mistral
- LLM_MODEL=phi3:mini         # Microsoft Phi-3
- LLM_MODEL=gemma2:2b         # Google Gemma 2
```

**Pull the new model first:**
```bash
docker compose exec ollama ollama pull mistral:7b
```

Then restart:
```bash
docker compose up -d --build api
```

---

### Code-Path Walkthrough: What Exactly Stays Unchanged

When you swap `STORAGE_BACKEND=local` → `STORAGE_BACKEND=s3`, here are
the files that **are not modified** (they only know about `IStorageService`):

| File | What it does | Touches storage? |
|---|---|---|
| `app/services/book_service.py` | Book CRUD + file pipeline | Calls `self.storage_service.save_file()` via interface |
| `app/api/routes.py` | HTTP handlers | Delegates to `BookService` |
| `app/services/background_tasks.py` | Async summary generation | Doesn't touch storage at all |
| `app/services/review_service.py` | Review + consensus | Doesn't touch storage at all |
| `app/services/recommendation.py` | ML recommendations | Doesn't touch storage at all |
| `app/services/preference_service.py` | User preferences | Doesn't touch storage at all |

The **only** file that knows about `LocalStorageService` vs `S3StorageService`
is `app/core/dependencies.py` — the Composition Root.

The same isolation applies to the LLM swap:

| File | What it does | Touches LLM? |
|---|---|---|
| `app/services/book_service.py` | Book CRUD | Via `self.llm_service` interface |
| `app/services/review_service.py` | Reviews + consensus | Via `self.llm_service` interface |
| `app/services/background_tasks.py` | Async BG tasks | Receives `llm_service: ILLMService` param |
| `app/services/preference_service.py` | Taste profiles | Via `self.llm_service` interface |
| `app/api/routes.py` | HTTP handlers | Calls `get_llm_service()` for BG task injection |

---

### Live Swap Test Results

All swap tests pass with **zero code changes** between configurations:

```
======================================================================
COMPONENT SWAP VERIFICATION — 11/11 passed
======================================================================
  STORAGE_BACKEND=s3     → Book uploaded to MinIO S3 bucket
  LLM_PROVIDER=mock      → MockLLM summary detected (instant)
  LLM_PROVIDER=openai    → Graceful fallback to mock (no API key)
  Original config restored → API healthy, upload succeeded
======================================================================
  Storage local→s3 :  1 config line change   PASS
  LLM llama→mock   :  1 config line change   PASS
  LLM llama→openai :  2 config lines         PASS  (provider + API key)
  Business logic changes required: ZERO
======================================================================
```

---

### Try It Yourself — Step-by-Step Swap Tutorials

The following two tutorials let you **reproduce** the swaps on your own
running stack.  Each tutorial is fully self-contained: open a terminal,
paste the commands, and watch the swap happen.

---

#### Tutorial A: Switch File Storage from Local Disk → S3 / MinIO

> **Goal:** Upload a book while storage is `local`, then swap to `s3` with
> a single config edit, upload another book, and verify it landed in MinIO —
> all without touching any Python file.

**Step 1 — Confirm current config is `local`**

```bash
grep "STORAGE_BACKEND" docker-compose.yml
# Expected output:
#   - STORAGE_BACKEND=local
```

**Step 2 — Login and upload a book with local storage**

```bash
# Get a JWT token
TOKEN=$(curl -s -X POST http://localhost:5223/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"alice@example.com","password":"secret123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Upload a book
echo "This is a book stored on LOCAL disk." > /tmp/local_test.txt
curl -s -X POST http://localhost:5223/books/ \
  -H "Authorization: Bearer $TOKEN" \
  -F "title=Local Storage Book" \
  -F "author=Test Author" \
  -F "file=@/tmp/local_test.txt;type=text/plain" | python3 -m json.tool
```

Notice the `file_path` in the response — it points to a SHA-256 subdirectory
on the local filesystem (e.g. `a1/a1b2c3d4…_local_test.txt`).

**Step 3 — Swap to S3 (one line)**

Open `docker-compose.yml` and change **exactly one value**:

```yaml
# ─── BEFORE ───
      - STORAGE_BACKEND=local

# ─── AFTER ────
      - STORAGE_BACKEND=s3
```

Or do it with sed (no editor needed):

```bash
sed -i '' 's/STORAGE_BACKEND=local/STORAGE_BACKEND=s3/' docker-compose.yml
```

**Step 4 — Rebuild and restart (only the API container)**

```bash
docker compose up -d --build api
# Wait for healthy
sleep 5
curl -s http://localhost:5223/health
# → {"status":"healthy"}
```

**Step 5 — Upload another book — now it goes to MinIO**

```bash
TOKEN=$(curl -s -X POST http://localhost:5223/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"alice@example.com","password":"secret123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "This is a book stored on S3 / MinIO." > /tmp/s3_test.txt
curl -s -X POST http://localhost:5223/books/ \
  -H "Authorization: Bearer $TOKEN" \
  -F "title=S3 Storage Book" \
  -F "author=Test Author" \
  -F "file=@/tmp/s3_test.txt;type=text/plain" | python3 -m json.tool
```

The `file_path` in the response is an S3 object key (same format, but now
the file physically lives in MinIO).

**Step 6 — Verify in MinIO console**

Open **http://localhost:9002** in your browser (login: `minioadmin` /
`minioadmin`).  Navigate to the **luminalib-books** bucket — you'll see
the uploaded file.

Or verify via CLI:

```bash
docker logs --tail 10 projectluminalib-api-1 2>&1 | grep "S3"
# Expected: "S3: uploaded a1/a1b2c3…_s3_test.txt (38 bytes)"
```

**Step 7 — Swap back to local (same one-line edit)**

```bash
sed -i '' 's/STORAGE_BACKEND=s3/STORAGE_BACKEND=local/' docker-compose.yml
docker compose up -d --build api
```

**What files were modified?**  Only `docker-compose.yml` (one value).
No `.py` file was opened, edited, or redeployed.

**Result:** Storage backend swapped with a single config line.

---

#### Tutorial B: Swap Llama 3 for OpenAI Without Rewriting Business Logic

> **Goal:** Upload a book with Llama 3 generating the summary, then swap to
> OpenAI with one config change, upload another book, and see that OpenAI now
> generates the summary — all while `book_service.py`, `review_service.py`,
> `background_tasks.py`, etc. remain **completely untouched**.

**Step 1 — Confirm current config is `llama`**

```bash
grep "LLM_PROVIDER" docker-compose.yml
# Expected output:
#   - LLM_PROVIDER=llama
```

**Step 2 — Upload a book and verify Llama-generated summary**

```bash
TOKEN=$(curl -s -X POST http://localhost:5223/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"alice@example.com","password":"secret123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "Artificial intelligence is transforming every industry." > /tmp/llama_test.txt
curl -s -X POST http://localhost:5223/books/ \
  -H "Authorization: Bearer $TOKEN" \
  -F "title=Llama Summary Book" \
  -F "author=Test Author" \
  -F "file=@/tmp/llama_test.txt;type=text/plain" | python3 -m json.tool
```

Wait ~10 seconds for the background task, then fetch the book:

```bash
BOOK_ID="<id-from-response-above>"
curl -s "http://localhost:5223/books/$BOOK_ID" \
  -H "Authorization: Bearer $TOKEN" | python3 -c "
import sys, json
b = json.load(sys.stdin)
print(f'Provider: Llama 3')
print(f'Summary:  {b[\"summary\"][:120]}...')
"
```

You'll see a natural-language summary generated by the Llama 3 model.

**Step 3 — Swap to OpenAI (change one or two lines)**

Open `docker-compose.yml` and change:

```yaml
# ─── BEFORE ───
      - LLM_PROVIDER=llama

# ─── AFTER ────
      - LLM_PROVIDER=openai
      - LLM_API_KEY=sk-your-openai-api-key-here   # ← add this line
```

Or via sed:

```bash
sed -i '' 's/LLM_PROVIDER=llama/LLM_PROVIDER=openai/' docker-compose.yml
```

> **No API key?** That's fine — `OpenAILLMService` gracefully falls back to
> `MockLLMService`, so the app still works.  You'll get mock summaries instead.

**Step 4 — Rebuild and restart**

```bash
docker compose up -d --build api
sleep 5
curl -s http://localhost:5223/health
# → {"status":"healthy"}
```

**Step 5 — Upload another book — now OpenAI generates the summary**

```bash
TOKEN=$(curl -s -X POST http://localhost:5223/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"alice@example.com","password":"secret123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "Deep learning enables remarkable breakthroughs in NLP." > /tmp/openai_test.txt
curl -s -X POST http://localhost:5223/books/ \
  -H "Authorization: Bearer $TOKEN" \
  -F "title=OpenAI Summary Book" \
  -F "author=Test Author" \
  -F "file=@/tmp/openai_test.txt;type=text/plain" | python3 -m json.tool
```

Wait ~5 seconds, then verify:

```bash
BOOK_ID="<id-from-response-above>"
curl -s "http://localhost:5223/books/$BOOK_ID" \
  -H "Authorization: Bearer $TOKEN" | python3 -c "
import sys, json
b = json.load(sys.stdin)
print(f'Provider: OpenAI (or mock fallback)')
print(f'Summary:  {b[\"summary\"][:120]}...')
"
```

**Step 6 — Confirm zero business logic files were touched**

```bash
# These files were NOT modified during the swap:
cat -n app/services/book_service.py | head -5    # same as before
cat -n app/services/review_service.py | head -5  # same as before
cat -n app/services/background_tasks.py | head -5 # same as before
cat -n app/services/recommendation.py | head -5   # same as before
```

The only file that changed was `docker-compose.yml` (the `LLM_PROVIDER`
value).  Every service file still calls the same `ILLMService` interface:

```python
# book_service.py — this code is IDENTICAL regardless of provider:
summary = await self.llm_service.generate_summary(text_content)
embedding = await self.llm_service.generate_embedding(text_content)

# review_service.py — same interface, different backend:
consensus = await self.llm_service.generate_review_consensus(book.title, review_dicts)

# background_tasks.py — receives ILLMService, never imports a concrete class:
async def generate_book_summary_task(book_id, file_content, mime_type, llm_service: ILLMService):
```

**Step 7 — Swap back to Llama**

```bash
sed -i '' 's/LLM_PROVIDER=openai/LLM_PROVIDER=llama/' docker-compose.yml
docker compose up -d --build api
```

**Result:** LLM provider swapped without rewriting a single line of
business logic.  The `BookService`, `ReviewService`, `BackgroundTasks`,
`PreferenceService`, and `RecommendationService` are completely decoupled
from the concrete LLM implementation.

---

## Project Structure

```
app/
├── main.py                         # FastAPI app, lifespan hook, CORS, routers
│
├── api/                            # ── Presentation Layer ──
│   ├── auth_routes.py              #   signup, login, profile, signout
│   ├── routes.py                   #   books CRUD, borrow, reviews, analysis
│   ├── recommendation_routes.py    #   /recommendations (user-level)
│   ├── preference_routes.py        #   user taste-preference endpoints
│   └── schemas.py                  #   Pydantic v2 request / response models
│
├── core/                           # ── Cross-Cutting Concerns ──
│   ├── config.py                   #   pydantic-settings (.env loading)
│   ├── security.py                 #   JWT creation / validation + bcrypt
│   ├── redis_client.py             #   shared Redis client (JWT blacklist + Celery)
│   └── dependencies.py             #   FastAPI DI container (Composition Root)
│
├── domain/                         # ── Domain Layer (zero external deps) ──
│   ├── entities.py                 #   Pure Python @dataclass entities
│   └── repositories.py            #   7 abstract interfaces (ports)
│
├── infrastructure/                 # ── Infrastructure Layer (adapters) ──
│   ├── database/
│   │   ├── connection.py           #   async engine + session factory (+ NullPool worker engine)
│   │   ├── models.py              #   6 SQLAlchemy 2.x ORM models
│   │   └── repository.py          #   5 concrete repository implementations
│   ├── llm/
│   │   ├── prompts.py             #   PromptTemplate + versioned registry
│   │   └── services.py            #   MockLLM / LlamaLLM / OpenAILLM
│   ├── storage/
│   │   ├── local.py               #   LocalStorageService (SHA-256 dirs)
│   │   └── s3.py                  #   S3StorageService (AWS / MinIO via boto3)
│   └── tasks/
│       ├── celery_app.py          #   Celery application instance + config
│       └── llm_tasks.py           #   Celery task wrappers (generate_book_summary, etc.)
│
└── services/                       # ── Application / Use-Case Layer ──
    ├── auth_service.py             #   signup, login, profile management
    ├── book_service.py             #   book CRUD + file ingestion pipeline
    ├── review_service.py           #   reviews + GenAI consensus
    ├── background_tasks.py         #   async coroutines executed by Celery workers
    ├── preference_service.py       #   user taste-profile management
    └── recommendation.py           #   ML cosine-similarity engine

tests/
    └── test_book_service.py        # 10 unit tests (pytest + AsyncMock)

docker-compose.yml                  # 7-service stack with healthchecks
Dockerfile                          # python:3.11-slim, port 5223
start.sh                            # Helper script to launch everything
requirements.txt                    # Production dependencies
requirements-dev.txt                # Dev/test dependencies (pytest, black, etc.)
ARCHITECTURE.md                     # Detailed architecture & design decisions
```

---

## Running Tests

```bash
# Local (requires Python 3.10+ and dev dependencies)
pip install -r requirements-dev.txt
pytest tests/ -v

# Via Makefile
make test
```

All 10 tests use `AsyncMock` for infrastructure dependencies (database, LLM,
storage) so they run **instantly** without Docker or any external service.

---

## Code Quality

```bash
make format   # black + isort
make lint     # mypy + flake8
```

**Standards applied:**

| Practice                        | Detail                                                        |
|---------------------------------|---------------------------------------------------------------|
| Type hints                      | Every function signature is annotated                         |
| Import ordering                 | `isort` with `profile=black` (stdlib → third-party → local)  |
| Formatting                      | `black` with line length 100                                  |
| Docstrings                      | All public classes and methods                                |
| Fully async                     | No blocking I/O anywhere — `async/await` end-to-end          |
| Pydantic v2                     | `model_config = ConfigDict(...)` — no deprecated `class Config` |
| Interface-driven design         | Every infra component has an abstract interface               |
| Dependency inversion            | Services depend on abstractions, never on concrete classes    |

---

## Technology Stack

| Layer            | Technology                                        |
|------------------|---------------------------------------------------|
| **Runtime**      | Python 3.11 · FastAPI · Uvicorn                   |
| **Database**     | PostgreSQL 15 · SQLAlchemy 2.x (async) · asyncpg  |
| **Validation**   | Pydantic v2 · pydantic-settings                   |
| **Auth**         | JWT (python-jose, HS256) · bcrypt (passlib)        |
| **LLM / GenAI**  | Ollama (Llama 3) · OpenAI · httpx                 |
| **ML**           | NumPy · scikit-learn (cosine similarity)           |
| **Storage**      | Local filesystem · MinIO (S3-compatible) · boto3   |
| **Task queue**   | Celery 5 · Redis broker · task-state result backend |
| **Containers**   | Docker Compose — 7 services with healthchecks      |
| **Cache/Broker** | Redis 7 — Celery broker + JWT revocation store     |
| **PDF**          | PyMuPDF (fitz)                                     |
| **Testing**      | pytest · pytest-asyncio · AsyncMock                |
| **Formatting**   | black · isort · flake8 · mypy                      |

---

## Architecture Documentation

For detailed design decisions, system diagrams, ER diagram, data flow
diagrams, SOLID principle mapping, and trade-off analysis, see:

**[ARCHITECTURE.md](ARCHITECTURE.md)**

Contents include:
- System architecture overview diagram (all 7 services)
- Clean Architecture layers diagram with class mapping
- Book upload pipeline (sequence diagram)
- Review submission & consensus pipeline
- ML recommendation pipeline
- Database ER diagram (6 tables with relationships)
- Deployment topology with healthcheck strategy
- SOLID principles — concrete examples from the codebase
- 10 key trade-offs with rationale

---

## License
 — educational assessment project.

# LuminaLib-v1