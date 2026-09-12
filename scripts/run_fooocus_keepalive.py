from __future__ import annotations

import runpy
import time
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    launch_path = repo_root / "launch.py"
    runpy.run_path(str(launch_path), run_name="__main__")
    print("Fooocus launch returned. Keeping this process alive so the local server stays reachable.")
    print("Keep this window open. Press Ctrl+C to stop Fooocus.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Fooocus keepalive stopped by user.")


if __name__ == "__main__":
    main()
