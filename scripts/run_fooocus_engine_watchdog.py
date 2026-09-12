from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path


ENGINE_HOST = "127.0.0.1"
ENGINE_PORT = 7865
MAX_RESTARTS = 2
RESTART_DELAY_SECONDS = 5
READY_TIMEOUT_SECONDS = 180


def port_is_listening(host: str = ENGINE_HOST, port: int = ENGINE_PORT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def wait_for_ready(timeout_seconds: int = READY_TIMEOUT_SECONDS) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout_seconds:
        if port_is_listening():
            elapsed = int(time.monotonic() - start)
            print(f"Fooocus watchdog: engine is listening on http://{ENGINE_HOST}:{ENGINE_PORT} after {elapsed}s", flush=True)
            return True
        time.sleep(1)
    return False


def run_once(repo_root: Path, extra_args: list[str]) -> int:
    launch_py = repo_root / "launch.py"
    command = [sys.executable, "-u", str(launch_py), *extra_args]
    print("Fooocus watchdog: starting child process", flush=True)
    print("Fooocus watchdog: command: " + " ".join(command), flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        command,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    ready_reported = False
    start = time.monotonic()

    try:
        while True:
            line = process.stdout.readline() if process.stdout else ""
            if line:
                print(line.rstrip(), flush=True)

            if not ready_reported and port_is_listening():
                elapsed = int(time.monotonic() - start)
                print(f"Fooocus watchdog: engine became ready on http://{ENGINE_HOST}:{ENGINE_PORT} after {elapsed}s", flush=True)
                ready_reported = True

            code = process.poll()
            if code is not None:
                if process.stdout:
                    for remaining in process.stdout:
                        print(remaining.rstrip(), flush=True)
                print(f"Fooocus watchdog: child exited with code {code}", flush=True)
                return int(code)

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Fooocus watchdog: stopping child due to Ctrl+C", flush=True)
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        raise


def should_restart(exit_code: int, attempt: int) -> bool:
    if attempt >= MAX_RESTARTS:
        return False
    if exit_code == 0:
        return False
    return True


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    extra_args = sys.argv[1:]

    print("Fooocus watchdog: active", flush=True)
    print(f"Fooocus watchdog: repo root: {repo_root}", flush=True)
    print(f"Fooocus watchdog: max restarts: {MAX_RESTARTS}", flush=True)

    attempt = 0
    while True:
        exit_code = run_once(repo_root, extra_args)
        if not should_restart(exit_code, attempt):
            if exit_code != 0:
                print(
                    "Fooocus watchdog: engine still failed after restart attempts. "
                    "Review logs\\studio\\latest-fooocus-engine.log and try option 4 Cold reset.",
                    flush=True,
                )
            sys.exit(exit_code)

        attempt += 1
        print(
            f"Fooocus watchdog: restarting Fooocus after exit code {exit_code}. "
            f"Restart {attempt}/{MAX_RESTARTS} in {RESTART_DELAY_SECONDS}s.",
            flush=True,
        )
        time.sleep(RESTART_DELAY_SECONDS)


if __name__ == "__main__":
    main()
