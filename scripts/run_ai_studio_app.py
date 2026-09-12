from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def log_boot(message: str) -> None:
    print(f"[{datetime.now().isoformat()}] {message}", flush=True)


def main() -> None:
    log_boot(f"AI Studio boot: repo root is {REPO_ROOT}")
    log_boot("AI Studio boot: importing ai_studio_app")
    import ai_studio_app

    log_boot("AI Studio boot: building Gradio UI")
    app = ai_studio_app.build_app()

    log_boot("AI Studio boot: launching on http://127.0.0.1:7872")
    app.launch(server_name="127.0.0.1", server_port=7872, inbrowser=True)

    log_boot("AI Studio boot: launch returned")


if __name__ == "__main__":
    main()
