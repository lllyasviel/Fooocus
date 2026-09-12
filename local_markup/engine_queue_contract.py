from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Dict, List, Optional
from uuid import uuid4


class EngineJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class EngineJobKind(str, Enum):
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_PROMPT = "image_prompt"
    INPAINT = "inpaint"
    UPSCALE = "upscale"
    ENHANCE = "enhance"


@dataclass(frozen=True)
class EngineJobRequest:
    kind: EngineJobKind
    prompt: str
    negative_prompt: str = ""
    seed: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineJobProgress:
    job_id: str
    status: EngineJobStatus
    message: str
    step: int = 0
    total_steps: int = 0
    updated_at: float = field(default_factory=time)

    @property
    def percent(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return max(0.0, min(100.0, (float(self.step) / float(self.total_steps)) * 100.0))


@dataclass(frozen=True)
class EngineJobRecord:
    job_id: str
    request: EngineJobRequest
    status: EngineJobStatus
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)
    outputs: List[str] = field(default_factory=list)
    error: Optional[str] = None


def new_job_id() -> str:
    return uuid4().hex


def create_queued_record(request: EngineJobRequest, job_id: Optional[str] = None) -> EngineJobRecord:
    return EngineJobRecord(
        job_id=job_id or new_job_id(),
        request=request,
        status=EngineJobStatus.QUEUED,
    )


def transition_job(record: EngineJobRecord, status: EngineJobStatus, error: Optional[str] = None, outputs: Optional[List[str]] = None) -> EngineJobRecord:
    return EngineJobRecord(
        job_id=record.job_id,
        request=record.request,
        status=status,
        created_at=record.created_at,
        updated_at=time(),
        outputs=record.outputs if outputs is None else outputs,
        error=error,
    )
