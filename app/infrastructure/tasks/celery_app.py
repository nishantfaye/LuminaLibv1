"""Celery application — broker and result backend both backed by Redis.

Workers run as a completely separate process from the API server, so LLM
calls never block request handlers and are not lost on API restart.

Task lifecycle states stored in Redis:
  PENDING  → task dispatched, not yet picked up by a worker
  STARTED  → worker has begun execution  (task_track_started=True)
  SUCCESS  → task finished without error
  FAILURE  → task raised an unhandled exception
  RETRY    → task failed and is waiting for its next retry attempt
"""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "luminalib",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.infrastructure.tasks.llm_tasks"],
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # State tracking
    task_track_started=True,        # emit STARTED state when a worker picks up a task
    result_expires=86400,           # keep results in Redis for 24 h
    # Reliability
    task_acks_late=True,            # ack only after the task finishes, not before
    worker_prefetch_multiplier=1,   # one task at a time per worker process
    task_reject_on_worker_lost=True,  # re-queue if worker crashes mid-task
)
