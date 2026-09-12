from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Dict, List, Optional


@dataclass(frozen=True)
class StudioHistoryItem:
    item_id: str
    prompt: str
    negative_prompt: str
    workflow: str
    image_path: Optional[str] = None
    seed: Optional[int] = None
    rating: Optional[int] = None
    notes: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time)


@dataclass(frozen=True)
class StudioHistoryStore:
    items: List[StudioHistoryItem] = field(default_factory=list)

    def add(self, item: StudioHistoryItem) -> "StudioHistoryStore":
        return StudioHistoryStore(items=[item, *self.items])

    def by_workflow(self, workflow: str) -> List[StudioHistoryItem]:
        return [item for item in self.items if item.workflow == workflow]

    def favorites(self) -> List[StudioHistoryItem]:
        return [item for item in self.items if item.rating is not None and item.rating >= 4]

    def latest(self, limit: int = 20) -> List[StudioHistoryItem]:
        return self.items[: max(0, limit)]


def create_history_item(
    item_id: str,
    prompt: str,
    negative_prompt: str,
    workflow: str,
    image_path: Optional[str] = None,
    seed: Optional[int] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> StudioHistoryItem:
    return StudioHistoryItem(
        item_id=item_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        workflow=workflow,
        image_path=image_path,
        seed=seed,
        metadata=metadata or {},
    )
