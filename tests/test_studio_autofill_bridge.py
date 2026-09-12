from __future__ import annotations

from pathlib import Path

import ai_studio_app


BRIDGE_PATH = Path("javascript/studio-autofill-bridge.js")
GRADIO_EXTENSIONS_PATH = Path("modules/ui_gradio_extensions.py")


def test_fooocus_autofill_bridge_script_exists_and_listens_for_studio_message() -> None:
    source = BRIDGE_PATH.read_text(encoding="utf-8")

    assert "fooocus-studio-autofill" in source
    assert "window.addEventListener(\"message\"" in source
    assert "allowedOrigins" in source
    assert "127.0.0.1:7872" in source
    assert "localhost:7872" in source


def test_fooocus_autofill_bridge_targets_prompt_fields() -> None:
    source = BRIDGE_PATH.read_text(encoding="utf-8")

    assert "positive_prompt" in source
    assert "negative_prompt" in source
    assert "setNativeValue" in source
    assert "dispatchEvent(new Event(\"input\"" in source
    assert "dispatchEvent(new Event(\"change\"" in source


def test_gradio_extension_loads_studio_autofill_bridge_without_importing_runtime_args() -> None:
    source = GRADIO_EXTENSIONS_PATH.read_text(encoding="utf-8")

    assert "studio-autofill-bridge.js" in source
    assert "studio_autofill_bridge_js_path" in source


def test_studio_and_engine_bridge_share_message_contract() -> None:
    studio_script = ai_studio_app.send_to_engine_js()
    engine_script = BRIDGE_PATH.read_text(encoding="utf-8")

    for key in [
        "type",
        "workflow",
        "fooocus_area",
        "prompt",
        "negative_prompt",
        "setup_steps",
        "next_shots",
    ]:
        assert key in studio_script
        assert key in engine_script

    assert "fooocus-studio-autofill" in studio_script
    assert "fooocus-studio-autofill" in engine_script
    assert "payload.type !== \"fooocus-studio-autofill\"" in engine_script


def test_fooocus_bridge_sends_acknowledgement_back_to_studio() -> None:
    engine_script = BRIDGE_PATH.read_text(encoding="utf-8")
    studio_script = ai_studio_app.studio_engine_bridge_script()

    assert "fooocus-studio-autofill-result" in engine_script
    assert "fooocus-studio-autofill-result" in studio_script
    assert "event.source.postMessage" in engine_script
    assert "promptFilled" in engine_script
    assert "negativeFilled" in engine_script
    assert "Engine fields filled" in studio_script
    assert "partial-fill" in studio_script


def test_fooocus_bridge_does_not_auto_click_generate() -> None:
    source = BRIDGE_PATH.read_text(encoding="utf-8").lower()

    assert "generate_button" not in source
    assert ".click()" not in source
    assert "click(" not in source


def test_fooocus_bridge_records_last_autofill_for_debugging() -> None:
    source = BRIDGE_PATH.read_text(encoding="utf-8")

    assert "window.__fooocusStudioLastAutofill" in source
    assert "promptFilled" in source
    assert "negativeFilled" in source
    assert "updatedAt" in source


def test_studio_status_panel_tracks_real_handoff_states() -> None:
    script = ai_studio_app.studio_engine_bridge_script()

    assert "Planning image workflow" in script
    assert "Plan ready" in script
    assert "Sending to engine" in script
    assert "Sent, waiting for confirmation" in script
    assert "Engine fields filled" in script
    assert "Studio is selecting" in script
    assert "This progress reflects planning only" in script
