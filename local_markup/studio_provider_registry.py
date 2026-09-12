from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from local_markup.studio_adapter_contract import ManualHandoffAdapter, ProviderAdapter


@dataclass(frozen=True)
class ProviderDescriptor:
    key: str
    label: str
    description: str
    supports_submit: bool


PROVIDER_DESCRIPTORS: Dict[str, ProviderDescriptor] = {
    "manual_handoff": ProviderDescriptor(
        key="manual_handoff",
        label="Manual Fooocus Handoff",
        description="Creates copy/paste steps for the current Fooocus UI without starting generation.",
        supports_submit=True,
    ),
    "local_dry_run": ProviderDescriptor(
        key="local_dry_run",
        label="Local Dry Run",
        description="Validates job conversion locally without touching the active Fooocus worker.",
        supports_submit=True,
    ),
}


def list_provider_descriptors() -> Iterable[ProviderDescriptor]:
    return tuple(PROVIDER_DESCRIPTORS.values())


def get_provider_descriptor(key: str) -> Optional[ProviderDescriptor]:
    return PROVIDER_DESCRIPTORS.get(key)


def get_provider_adapter(key: str) -> ProviderAdapter:
    if key == "manual_handoff":
        return ManualHandoffAdapter()
    if key == "local_dry_run":
        from local_markup.local_fooocus_adapter import LocalDryRunFooocusAdapter

        return LocalDryRunFooocusAdapter()
    raise KeyError(f"Unknown provider adapter: {key}")
