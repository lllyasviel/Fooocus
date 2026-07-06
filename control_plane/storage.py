from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import ensure_runtime_dirs, get_settings
from .models import GalleryItem, Job, JobStatus, TraceResult, utc_now

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    queue TEXT NOT NULL,
    compute_mode TEXT NOT NULL,
    compute_target TEXT,
    worker_id TEXT,
    payload_json TEXT NOT NULL,
    result_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_queue ON jobs(status, queue, created_at);

CREATE TABLE IF NOT EXISTS trace_results (
    id TEXT PRIMARY KEY,
    image_path TEXT NOT NULL,
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gallery_items (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    output_path TEXT NOT NULL,
    prompt TEXT NOT NULL,
    data_json TEXT NOT NULL,
    favorite INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    deleted_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_gallery_created_at ON gallery_items(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gallery_favorite ON gallery_items(favorite, created_at DESC);
"""


class ControlPlaneStore:
    def __init__(self, db_path: Path | None = None) -> None:
        settings = get_settings()
        ensure_runtime_dirs(settings)
        self.db_path = db_path or settings.db_path
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def create_job(self, job: Job) -> Job:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, type, status, queue, compute_mode, compute_target, worker_id,
                    payload_json, result_json, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.type.value,
                    job.status.value,
                    job.queue.value,
                    job.compute_mode.value,
                    job.compute_target,
                    job.worker_id,
                    json.dumps(job.payload),
                    json.dumps(job.result),
                    job.error,
                    job.created_at,
                    job.updated_at,
                ),
            )
        return job

    def update_job(self, job_id: str, **changes: Any) -> None:
        allowed = {"status", "queue", "compute_target", "worker_id", "result", "error"}
        sets: list[str] = []
        values: list[Any] = []
        for key, value in changes.items():
            if key not in allowed:
                continue
            column = "result_json" if key == "result" else key
            sets.append(f"{column} = ?")
            values.append(json.dumps(value) if key == "result" else value)
        if not sets:
            return
        sets.append("updated_at = ?")
        values.append(utc_now())
        values.append(job_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", values)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job_row_to_dict(row) if row else None

    def list_jobs(self, status: JobStatus | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM jobs"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status.value)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._job_row_to_dict(row) for row in rows]

    def save_trace(self, trace: TraceResult) -> TraceResult:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO trace_results (id, image_path, data_json, created_at) VALUES (?, ?, ?, ?)",
                (trace.id, trace.image_path, json.dumps(trace.to_dict()), trace.created_at),
            )
        return trace

    def add_gallery_item(self, item: GalleryItem) -> GalleryItem:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO gallery_items (id, job_id, output_path, prompt, data_json, favorite, created_at, deleted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id,
                    item.job_id,
                    item.output_path,
                    item.prompt,
                    json.dumps(item.to_dict()),
                    int(item.favorite),
                    item.created_at,
                    item.deleted_at,
                ),
            )
        return item

    def list_gallery(self, limit: int = 100, favorites_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM gallery_items WHERE deleted_at IS NULL"
        params: list[Any] = []
        if favorites_only:
            sql += " AND favorite = 1"
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [json.loads(row["data_json"]) for row in rows]

    def set_favorite(self, image_id: str, favorite: bool) -> None:
        with self.connect() as conn:
            row = conn.execute("SELECT data_json FROM gallery_items WHERE id = ?", (image_id,)).fetchone()
            if not row:
                return
            data = json.loads(row["data_json"])
            data["favorite"] = favorite
            conn.execute(
                "UPDATE gallery_items SET favorite = ?, data_json = ? WHERE id = ?",
                (int(favorite), json.dumps(data), image_id),
            )

    def soft_delete_gallery_item(self, image_id: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE gallery_items SET deleted_at = ? WHERE id = ?", (utc_now(), image_id))

    @staticmethod
    def _job_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "type": row["type"],
            "status": row["status"],
            "queue": row["queue"],
            "compute_mode": row["compute_mode"],
            "compute_target": row["compute_target"],
            "worker_id": row["worker_id"],
            "payload": json.loads(row["payload_json"]),
            "result": json.loads(row["result_json"]),
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
