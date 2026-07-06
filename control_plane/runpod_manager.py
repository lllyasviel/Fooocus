from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, get_settings


@dataclass
class ProvisionRequest:
    gpu_type: str
    max_hourly_cost: float
    max_session_cost: float
    require_approval: bool = True
    approved: bool = False


@dataclass
class ProvisionResult:
    started: bool
    message: str
    worker_url: str = ""
    pod_id: str = ""


class RunPodManager:
    """Policy-first RunPod manager.

    This scaffold intentionally refuses to spend money unless approval and a real
    RunPod provider adapter are added. It creates a safe contract for the UI and
    backend while preventing silent spend.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def build_request(self, gpu_type: str | None = None) -> ProvisionRequest:
        selected_gpu = gpu_type or (self.settings.runpod_allowed_gpus[0] if self.settings.runpod_allowed_gpus else "RTX_4090")
        if selected_gpu not in self.settings.runpod_allowed_gpus:
            raise ValueError(f"GPU {selected_gpu} is not allowed by RUNPOD_ALLOWED_GPUS")
        return ProvisionRequest(
            gpu_type=selected_gpu,
            max_hourly_cost=self.settings.runpod_max_hourly_cost,
            max_session_cost=self.settings.runpod_max_session_cost,
            require_approval=self.settings.runpod_require_approval,
        )

    def provision(self, request: ProvisionRequest) -> ProvisionResult:
        if request.require_approval and not request.approved:
            return ProvisionResult(False, "Approval required before starting a paid RunPod pod.")
        if not self.settings.runpod_auto_provision:
            return ProvisionResult(False, "RUNPOD_AUTO_PROVISION is disabled.")
        if not self.settings.runpod_api_key:
            return ProvisionResult(False, "RUNPOD_API_KEY is missing.")
        return ProvisionResult(False, "Provider adapter not implemented yet. This prevents silent spend.")
