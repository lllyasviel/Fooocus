# Fooocus Control Plane Scaffold

This package is the first non-breaking backend layer for the Image Command Center.

It adds contracts for:

- persistent jobs
- gallery metadata
- queue selection
- local/runpod/auto compute routing
- safe RunPod provisioning policy
- image trace prompt building
- optional FastAPI endpoints

It does not replace the legacy Gradio worker yet. That is intentional: backend contracts first, integration second.

## Smoke Test

```bash
python - <<'PY'
from control_plane.models import Job, JobType
from control_plane.queues import select_queue
from control_plane.compute_router import ComputeRouter

job = Job(type=JobType.GENERATE, payload={"prompt": "test", "width": 1024, "height": 1024})
job.queue = select_queue(job)
print(job.to_dict())
print(ComputeRouter().route(job))
PY
```

## Optional API Run

After installing FastAPI/uvicorn in the environment:

```bash
uvicorn control_plane.api:app --host 0.0.0.0 --port 7870
```

## RunPod Safety

This scaffold intentionally does not create a paid RunPod pod yet. The RunPod manager enforces policy first and prevents silent spend until a real provider adapter is added.
