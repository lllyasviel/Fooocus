from __future__ import annotations

from typing import Any

from .compute_router import ComputeRouter
from .models import ComputeMode, Job, JobType
from .queues import select_queue
from .runpod_manager import RunPodManager
from .storage import ControlPlaneStore
from .trace_service import TraceService

try:
    from fastapi import FastAPI, HTTPException
except Exception:  # keeps existing Fooocus installs working when FastAPI is absent
    FastAPI = None
    HTTPException = Exception


if FastAPI:
    app = FastAPI(title="Fooocus Image Command Center")
    store = ControlPlaneStore()
    router = ComputeRouter()
    tracer = TraceService()
    runpod = RunPodManager()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "fooocus-control-plane"}

    @app.get("/compute/status")
    def compute_status() -> dict[str, Any]:
        return {
            "local": router.local_health().__dict__,
            "runpod": router.runpod_health().__dict__,
        }

    @app.post("/jobs")
    def create_job(payload: dict[str, Any]) -> dict[str, Any]:
        job_type = JobType(payload.get("type", "generate"))
        compute_mode = ComputeMode(payload.get("compute_mode", "auto"))
        job = Job(type=job_type, payload=payload, compute_mode=compute_mode)
        job.queue = select_queue(job)
        decision = router.route(job)
        job.compute_target = decision.target
        store.create_job(job)
        return {"job": job.to_dict(), "route": decision.__dict__}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/gallery")
    def gallery(limit: int = 100, favorites_only: bool = False) -> dict[str, Any]:
        return {"items": store.list_gallery(limit=limit, favorites_only=favorites_only)}

    @app.post("/trace-image")
    def trace_image(payload: dict[str, Any]) -> dict[str, Any]:
        image_path = payload.get("image_path")
        if not image_path:
            raise HTTPException(status_code=400, detail="image_path is required")
        trace = tracer.trace_image(image_path=image_path, user_hint=payload.get("hint", ""))
        store.save_trace(trace)
        return trace.to_dict()

    @app.post("/runpod/provision-request")
    def provision_request(payload: dict[str, Any]) -> dict[str, Any]:
        request = runpod.build_request(payload.get("gpu_type"))
        result = runpod.provision(request)
        return {"request": request.__dict__, "result": result.__dict__}
else:
    app = None
