"""Task status API route.

Exposes a single endpoint so API clients can poll the state of any
background Celery task dispatched by the books or reviews endpoints.

  GET /tasks/{task_id}

Possible ``status`` values mirror Celery's task state machine:
  PENDING  — task queued, not yet picked up by a worker
  STARTED  — worker has begun execution
  SUCCESS  — task completed without error (``result`` field populated)
  FAILURE  — task raised an unhandled exception (``error`` field populated)
  RETRY    — task failed and is scheduled for a retry attempt
"""

import logging

from celery.result import AsyncResult
from fastapi import APIRouter

from app.api.schemas import TaskStatusResponse
from app.infrastructure.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Get the current state of a background task.

    Poll this endpoint after receiving an ``X-Task-ID`` (or similar) header
    from a mutating request (book upload, content update, review submission).
    """
    result = AsyncResult(task_id, app=celery_app)

    error: str | None = None
    if result.state == "FAILURE":
        error = str(result.result)

    logger.debug("Task %s state: %s", task_id, result.state)

    return TaskStatusResponse(
        task_id=task_id,
        status=result.state,
        result=None,   # LLM tasks return None on success — state is enough
        error=error,
    )
