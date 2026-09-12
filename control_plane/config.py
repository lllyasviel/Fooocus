from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    output_dir: Path
    db_path: Path
    compute_mode: str
    local_worker_url: str
    runpod_worker_url: str
    runpod_api_key: str
    runpod_enabled: bool
    local_enabled: bool
    runpod_auto_provision: bool
    runpod_require_approval: bool
    runpod_max_hourly_cost: float
    runpod_max_session_cost: float
    runpod_idle_shutdown_minutes: int
    runpod_allowed_gpus: tuple[str, ...]


def bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    data_dir = Path(os.getenv("FOOOCUS_DATA_DIR", "/workspace/fooocus-data"))
    output_dir = Path(os.getenv("FOOOCUS_OUTPUT_DIR", str(data_dir / "outputs")))
    db_path = Path(os.getenv("FOOOCUS_DB_PATH", str(data_dir / "control-plane.sqlite3")))
    allowed_gpus = tuple(
        gpu.strip()
        for gpu in os.getenv("RUNPOD_ALLOWED_GPUS", "RTX_4090,L40S,A40,A100").split(",")
        if gpu.strip()
    )

    return Settings(
        data_dir=data_dir,
        output_dir=output_dir,
        db_path=db_path,
        compute_mode=os.getenv("COMPUTE_MODE", "auto"),
        local_worker_url=os.getenv("LOCAL_WORKER_URL", "http://127.0.0.1:7865"),
        runpod_worker_url=os.getenv("RUNPOD_WORKER_URL", ""),
        runpod_api_key=os.getenv("RUNPOD_API_KEY", ""),
        runpod_enabled=bool_env("RUNPOD_ENABLED", False),
        local_enabled=bool_env("LOCAL_ENABLED", True),
        runpod_auto_provision=bool_env("RUNPOD_AUTO_PROVISION", False),
        runpod_require_approval=bool_env("RUNPOD_REQUIRE_APPROVAL", True),
        runpod_max_hourly_cost=float(os.getenv("RUNPOD_MAX_HOURLY_COST", "1.00")),
        runpod_max_session_cost=float(os.getenv("RUNPOD_MAX_SESSION_COST", "5.00")),
        runpod_idle_shutdown_minutes=int(os.getenv("RUNPOD_IDLE_SHUTDOWN_MINUTES", "15")),
        runpod_allowed_gpus=allowed_gpus,
    )


def ensure_runtime_dirs(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
