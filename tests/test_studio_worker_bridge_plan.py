from __future__ import annotations

from local_markup.studio_worker_bridge_plan import BRIDGE_PLAN, BridgeStepState, bridge_plan_markdown


def test_bridge_plan_keeps_live_generation_blocked() -> None:
    states = {step.name: step.state for step in BRIDGE_PLAN}

    assert states["Studio dry-run job"] == BridgeStepState.READY
    assert states["Worker bridge"] == BridgeStepState.DEFERRED
    assert states["Live generation"] == BridgeStepState.BLOCKED


def test_bridge_plan_markdown_is_readable() -> None:
    markdown = bridge_plan_markdown()

    assert "Local Worker Bridge Plan" in markdown
    assert "Worker bridge" in markdown
    assert "Live generation" in markdown
