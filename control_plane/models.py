from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class JobType(str, Enum):
    TRACE = "trace"
    GENERATE = "generate"
    UPSCALE = "upscale"
    BATCH = "batch"
    ADMIN = "admin"


class ComputeMode(str, Enum):
    LOCAL = "local"
    RUNPOD = "runpod"
    AUTO = "auto"


class QueueName(str, Enum):
    TRACE = "trace_queue"
    FAST = "fast_queue"
    HEAVY = "heavy_queue"
    PRIORITY = "priority_queue"
    BATCH = "batch_queue"


@dataclass
class Job:
    type: JobType
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: f"job_{uuid4().hex}")
    status: JobStatus = JobStatus.QUEUED
    queue: QueueName = QueueName.FAST
    compute_mode: ComputeMode = ComputeMode.AUTO
    compute_target: str | None = None
    worker_id: str | None = None
    error: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        data["status"] = self.status.value
        data["queue"] = self.queue.value
        data["compute_mode"] = self.compute_mode.value
        return data


@dataclass
class TraceResult:
    image_path: str
    caption: str = ""
    subjects: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    lighting: str = ""
    composition: str = ""
    quality_tags: list[str] = field(default_factory=list)
    suggested_prompt: str = ""
    suggested_negative_prompt: str = ""
    id: str = field(default_factory=lambda: f"trace_{uuid4().hex}")
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GalleryItem:
    job_id: str
    output_path: str
    prompt: str
    id: str = field(default_factory=lambda: f"img_{uuid4().hex}")
    negative_prompt: str = ""
    seed: int | None = None
    model: str | None = None
    loras: list[str] = field(default_factory=list)
    width: int | None = None
    height: int | None = None
    source_image_path: str | None = None
    trace_id: str | None = None
    compute_target: str | None = None
    worker_id: str | None = None
    generation_seconds: float | None = None
    estimated_cost: float | None = None
    favorite: bool = False
    created_at: str = field(default_factory=utc_now)
    deleted_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
