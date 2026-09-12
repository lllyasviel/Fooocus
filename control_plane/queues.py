from __future__ import annotations

from .models import Job, JobType, QueueName


def select_queue(job: Job) -> QueueName:
    if job.type == JobType.TRACE:
        return QueueName.TRACE
    if job.type == JobType.UPSCALE:
        return QueueName.HEAVY
    if job.type == JobType.BATCH:
        return QueueName.BATCH
    if job.payload.get("priority"):
        return QueueName.PRIORITY

    width = int(job.payload.get("width") or 0)
    height = int(job.payload.get("height") or 0)
    steps = int(job.payload.get("steps") or 0)

    if width * height >= 1024 * 1024 or steps >= 40:
        return QueueName.HEAVY
    return QueueName.FAST
