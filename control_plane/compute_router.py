from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import Settings, get_settings
from .models import ComputeMode, Job


@dataclass
class WorkerHealth:
    name: str
    enabled: bool
    online: bool
    busy: bool = False
    gpu_name: str | None = None
    free_vram_mb: int | None = None
    url: str = ""


@dataclass
class RouteDecision:
    target: Literal["local", "runpod", "provision_required"]
    reason: str
    worker_url: str = ""
    requires_approval: bool = False


class ComputeRouter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def route(
        self,
        job: Job,
        local: WorkerHealth | None = None,
        runpod: WorkerHealth | None = None,
    ) -> RouteDecision:
        mode = ComputeMode(job.compute_mode or self.settings.compute_mode)
        local = local or self.local_health()
        runpod = runpod or self.runpod_health()

        if mode == ComputeMode.LOCAL:
            if local.enabled and local.online:
                return RouteDecision("local", "compute mode forced to local", local.url)
            return RouteDecision("provision_required", "local requested but local worker is offline", requires_approval=True)

        if mode == ComputeMode.RUNPOD:
            if runpod.enabled and runpod.online:
                return RouteDecision("runpod", "compute mode forced to RunPod", runpod.url)
            return self._runpod_unavailable_decision()

        if self._job_should_use_runpod(job):
            if runpod.enabled and runpod.online:
                return RouteDecision("runpod", "heavy job routed to RunPod", runpod.url)
            if local.enabled and local.online and not local.busy:
                return RouteDecision("local", "RunPod unavailable; using healthy local fallback", local.url)
            return self._runpod_unavailable_decision()

        if local.enabled and local.online and not local.busy:
            return RouteDecision("local", "local worker healthy and job is light", local.url)
        if runpod.enabled and runpod.online:
            return RouteDecision("runpod", "local busy/offline; RunPod healthy", runpod.url)
        if local.enabled and local.online:
            return RouteDecision("local", "RunPod offline; local available", local.url)
        return self._runpod_unavailable_decision()

    def local_health(self) -> WorkerHealth:
        return WorkerHealth(
            name="local",
            enabled=self.settings.local_enabled,
            online=self.settings.local_enabled,
            url=self.settings.local_worker_url,
        )

    def runpod_health(self) -> WorkerHealth:
        return WorkerHealth(
            name="runpod",
            enabled=self.settings.runpod_enabled,
            online=bool(self.settings.runpod_worker_url) and self.settings.runpod_enabled,
            url=self.settings.runpod_worker_url,
        )

    def _runpod_unavailable_decision(self) -> RouteDecision:
        if self.settings.runpod_auto_provision:
            return RouteDecision(
                "provision_required",
                "RunPod unavailable; auto-provision policy is enabled",
                requires_approval=self.settings.runpod_require_approval,
            )
        return RouteDecision(
            "provision_required",
            "RunPod unavailable and auto-provisioning is disabled",
            requires_approval=True,
        )

    @staticmethod
    def _job_should_use_runpod(job: Job) -> bool:
        payload = job.payload
        if payload.get("force_runpod"):
            return True
        width = int(payload.get("width") or 0)
        height = int(payload.get("height") or 0)
        steps = int(payload.get("steps") or 0)
        batch_size = int(payload.get("batch_size") or 1)
        operation = str(payload.get("operation") or job.type.value).lower()
        return (
            width * height >= 1024 * 1024
            or steps >= 40
            or batch_size > 2
            or operation in {"upscale", "inpaint", "batch"}
        )
