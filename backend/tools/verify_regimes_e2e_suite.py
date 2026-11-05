"""Run the existing optional E2E tests against isolated, owned dev servers.

Only the server fixture is replaced in memory. The eight supported test functions
and their existing drivers run unchanged; no existing process is stopped.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"


@pytest.fixture(scope="module", name="servers")
def isolated_servers():
    for port in [8000, 5173]:
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                pytest.fail(f"Port {port} is already occupied; no existing process was stopped.")
    directory = ROOT / ".tmp/regime-e2e-suite" / uuid4().hex
    directory.mkdir(parents=True)
    node = shutil.which("node")
    assert node, "Node is required for the existing frontend server"
    env = {
        **os.environ,
        "QML_DATA_DIR": str(directory / "data"),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    processes, handles = [], []
    try:
        for name, command, cwd in [
            (
                "backend",
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                ],
                BACKEND,
            ),
            (
                "frontend",
                [
                    node,
                    str(ROOT / "frontend/node_modules/vite/bin/vite.js"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "5173",
                    "--strictPort",
                ],
                ROOT / "frontend",
            ),
        ]:
            handle = (directory / f"{name}.log").open("w", encoding="utf-8")
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            )
        for url in ["http://127.0.0.1:8000/api/health", "http://127.0.0.1:5173/"]:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(url, timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.25)
            else:
                pytest.fail(f"Isolated server did not answer: {url}. Logs: {directory}")
        yield
    finally:
        for process in processes:
            process.terminate()
            process.wait(timeout=15)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    os.environ["QM_E2E"] = "1"
    sys.path.insert(0, str(BACKEND))
    from tests import test_e2e_live

    test_e2e_live.servers = isolated_servers
    raise SystemExit(
        pytest.main(["tests/test_e2e_live.py", "-q", "-p", "no:cacheprovider", *sys.argv[1:]])
    )
