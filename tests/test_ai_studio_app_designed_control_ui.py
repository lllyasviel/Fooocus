from __future__ import annotations

import inspect

import ai_studio_app
from ai_studio_app import build_app, fooocus_iframe_html, send_to_engine_js, studio_engine_bridge_script
from local_markup.studio_control_ui import launcher_controls_markdown


def test_designed_control_ui_hides_engine_and_adds_downloads() -> None:
    source = inspect.getsource(ai_studio_app.build_app)

    assert "Studio Control Center" in source
    assert "Hidden Fooocus engine" in source
    assert "open=False" in source
    assert "Download Prompt Pack" in source
    assert "Download Session History" in source
    assert "Generated image history" in source
    assert "Fooocus Engine" not in source.split("with gr.Tab(")[1]


def test_designed_control_ui_adds_safe_send_to_engine_autofill() -> None:
    source = inspect.getsource(ai_studio_app.build_app)

    assert "Send to Engine" in source
    assert "build_engine_handoff_text" in source
    assert "Engine send status" in source
    assert "postMessage" in studio_engine_bridge_script()
    assert "postMessage" in send_to_engine_js()
    assert "_js=send_to_engine_js()" in source
    assert "studio_send_to_engine_button" in source
    assert "studio_primary_prompt" in source
    assert "studio_negative_prompt" in source


def test_designed_control_ui_adds_animated_status_panel() -> None:
    source = inspect.getsource(ai_studio_app.build_app)
    script = studio_engine_bridge_script()

    assert "studio_status_panel_html" in source
    assert "studio_build_plan_button" in source
    assert "fooocusStudioSetStatus" in script
    assert "studio_progress_fill" in script
    assert "studio_toast" in script
    assert "fooocus-studio-autofill-result" in script


def test_designed_control_ui_adds_launcher_reset_tab() -> None:
    source = inspect.getsource(ai_studio_app.build_app)
    launcher_text = launcher_controls_markdown()

    assert "Launcher / Reset" in source
    assert "launcher_controls_markdown" in source
    assert "START_AI_IMAGE_STUDIO.bat" in launcher_text
    assert "Hot reset" in launcher_text
    assert "Cold reset" in launcher_text


def test_fooocus_iframe_is_available_but_not_the_primary_ui() -> None:
    html = fooocus_iframe_html()

    assert "127.0.0.1:7865" in html
    assert "iframe" in html
    assert "fooocus_engine_iframe" in html


def test_studio_engine_bridge_posts_to_iframe() -> None:
    script = studio_engine_bridge_script()

    assert "fooocus_engine_iframe" in script
    assert "fooocus-studio-autofill" in script
    assert "iframe.contentWindow.postMessage" in script
    assert "studio_selected_tool" in script
    assert "studio_selected_area" in script
    assert "studio_primary_prompt" in script
    assert "studio_negative_prompt" in script


def test_send_to_engine_js_posts_directly_from_button_inputs() -> None:
    script = send_to_engine_js()

    assert "fooocus_engine_iframe" in script
    assert "fooocus-studio-autofill" in script
    assert "workflow: workflow" in script
    assert "prompt: prompt" in script
    assert "negative_prompt: negative_prompt" in script
    assert "return [workflow, fooocus_area, prompt, negative_prompt, setup_steps, next_shots]" in script


def test_designed_control_ui_builds_without_launching() -> None:
    app = build_app()

    assert app is not None
