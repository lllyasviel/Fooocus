from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MarkupSession:
    source_image: str
    instruction: str
    prompt: str
    negative_prompt: str
    mode: str
    mask_image: Optional[str] = None
    output_image: Optional[str] = None
    seed: Optional[int] = None
    model: Optional[str] = None
    id: str = field(default_factory=lambda: f"markup_{uuid4().hex}")
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def save_markup_session(session: MarkupSession, output_dir: str) -> Path:
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{session.id}.json"
    path.write_text(json.dumps(session.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path
