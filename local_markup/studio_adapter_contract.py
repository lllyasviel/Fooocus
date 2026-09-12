from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol

from local_markup.engine_queue_contract import EngineJobKind, EngineJobRequest


class AdapterJobStatus(str, Enum):
    ACCEPTED = "accepted"
    MANUAL_HANDOFF = "manual_handoff"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ReferenceImage:
    name: str
    path: Optional[str] = None
    role: str = "reference"


@dataclass(frozen=True)
class ImageStudioJob:
    goal: str
    prompt: str
    negative_prompt: str
    kind: EngineJobKind
    references: List[ReferenceImage] = field(default_factory=list)
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_engine_request(self) -> EngineJobRequest:
        return EngineJobRequest(
            kind=self.kind,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            seed=self.seed,
            width=self.width,
            height=self.height,
            references=[item.path or item.name for item in self.references],
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class AdapterResult:
    status: AdapterJobStatus
    message: str
    job_id: Optional[str] = None
    handoff_steps: List[str] = field(default_factory=list)


class ProviderAdapter(Protocol):
    name: str

    def submit(self, job: ImageStudioJob) -> AdapterResult:
        ...


class ManualHandoffAdapter:
    name = "manual_handoff"

    def submit(self, job: ImageStudioJob) -> AdapterResult:
        steps = [
            "Open Fooocus in the browser.",
            f"Choose workflow: {job.kind.value}.",
            "Copy the prompt into the prompt box.",
            "Copy the negative prompt into the negative prompt box.",
            "Upload the listed reference images if any.",
            "Generate one candidate first, then review before continuing.",
        ]
        return AdapterResult(
            status=AdapterJobStatus.MANUAL_HANDOFF,
            message="Manual Fooocus handoff created. No automatic generation was started.",
            handoff_steps=steps,
        )
