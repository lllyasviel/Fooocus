from __future__ import annotations

from pathlib import Path


def test_keepalive_runner_exists_and_runs_launch_path() -> None:
    content = Path("scripts/run_fooocus_keepalive.py").read_text(encoding="utf-8")

    assert "runpy.run_path" in content
    assert "launch.py" in content
    assert "Keeping this process alive" in content


def test_one_ui_launcher_uses_keepalive_runner() -> None:
    content = Path("RUN_STUDIO_ONE_UI.bat").read_text(encoding="utf-8")

    assert "scripts\\run_fooocus_keepalive.py" in content
    assert "--disable-in-browser" in content
    assert "%TEMP%\\fooocus" in content
