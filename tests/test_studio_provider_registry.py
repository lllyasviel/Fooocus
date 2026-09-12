from __future__ import annotations

import pytest

from local_markup.studio_adapter_contract import ManualHandoffAdapter
from local_markup.studio_provider_registry import (
    get_provider_adapter,
    get_provider_descriptor,
    list_provider_descriptors,
)


def test_provider_registry_lists_manual_handoff() -> None:
    descriptors = {item.key: item for item in list_provider_descriptors()}

    assert "manual_handoff" in descriptors
    assert descriptors["manual_handoff"].supports_submit is True


def test_provider_registry_returns_manual_adapter() -> None:
    adapter = get_provider_adapter("manual_handoff")

    assert isinstance(adapter, ManualHandoffAdapter)


def test_provider_registry_returns_descriptor_by_key() -> None:
    descriptor = get_provider_descriptor("local_dry_run")

    assert descriptor is not None
    assert descriptor.label == "Local Dry Run"


def test_provider_registry_rejects_unknown_provider() -> None:
    with pytest.raises(KeyError):
        get_provider_adapter("missing")
