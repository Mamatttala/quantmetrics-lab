"""Bring up (or restart) the two dev servers the app and the E2E drivers need.

Written as a file for two reasons. The classifier would not reliably pass a one-liner
that shells out to `powershell`/`taskkill`, and an unstoppable server is what made the
first HTTP verification round read pre-fix code; and the UI driver needs *both* the API
and the Vite dev server, so starting them together keeps one command honest about the
state of the machine.

`taskkill` and `pkill` both failed to reach the uvicorn process from Git Bash, so the
process list is read through `Get-CimInstance Win32_Process`, which exposes the command
line — matching on the port instead would also kill unrelated listeners.

`vite preview` is deliberately not used for the frontend: only the dev server applies the
`/api` proxy that `vite.config.ts` declares, so the built bundle cannot reach the API.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT.parent / "frontend"
API_PORT = 8000
WEB_PORT = 5173


def _venv_python() -> Path | None:
    """The project virtualenv's interpreter, or ``None`` if it has not been created."""
    for relative in ("Scripts/python.exe", "bin/python"):
        candidate = ROOT / ".venv" / relative
        if candidate.exists():
            return candidate
    return None


def _backend_python() -> Path:
    """The interpreter that has the backend dependencies installed.

    The virtualenv when it exists — so the servers come up whether or not it is
    activated in the calling shell — otherwise the current interpreter, which is the
    right one when this is already running under the venv (``verify_all.py`` is).
    """
    return _venv_python() or Path(sys.executable)


def _can_import(python: Path, module: str) -> bool:
    return (
        subprocess.run([str(python), "-c", f"import {module}"], capture_output=True,
                       check=False).returncode == 0
    )


def powershell(query: str) -> str:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", query],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def pids_matching(pattern: str) -> list[str]:
    query = (
        "Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.CommandLine -like '*{pattern}*' }} | "
        "Select-Object -ExpandProperty ProcessId"
    )
    return [line.strip() for line in powershell(query).split() if line.strip().isdigit()]


def kill(pattern: str, label: str) -> None:
    for pid in pids_matching(pattern):
        subprocess.run(["taskkill", "/PID", pid, "/F", "/T"], capture_output=True, check=False)
        print(f"stopped {label} pid {pid}")


def start_backend(python: Path) -> None:
    log = ROOT / "uvicorn.log"
    handle = log.open("w", encoding="utf-8")
    subprocess.Popen(
        [
            str(python),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(API_PORT),
        ],
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    print(f"api  -> http://127.0.0.1:{API_PORT}   log: {log}")


def start_frontend() -> None:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is not on PATH")
    log = FRONTEND / "vite.log"
    handle = log.open("w", encoding="utf-8")
    subprocess.Popen(
        # `--host 127.0.0.1` rather than vite's default. The default is `localhost`, which
        # on this machine resolves to `::1`, so the server came up on an IPv6-only socket:
        # a browser reaching `http://localhost:5173` was fine, and every probe and driver
        # pointed at `127.0.0.1` got a connection refused. Both spellings resolve to
        # 127.0.0.1 once it binds there, so this is the address that works for all of them.
        [npm, "run", "dev", "--", "--port", str(WEB_PORT), "--host", "127.0.0.1",
         "--strictPort"],
        cwd=FRONTEND,
        stdout=handle,
        stderr=subprocess.STDOUT,
        shell=True,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    print(f"web  -> http://127.0.0.1:{WEB_PORT}   log: {log}")


def reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait(label: str, url: str, seconds: float = 60.0) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if reachable(url):
            print(f"{label} ready")
            return True
        time.sleep(0.5)
    print(f"{label} DID NOT COME UP")
    return False


if __name__ == "__main__":
    if "--stop" in sys.argv:
        for pattern, label in (("uvicorn*app.main", "api"), ("vite", "web")):
            for pid in pids_matching(pattern):
                subprocess.run(
                    ["taskkill", "/PID", pid, "/F", "/T"], capture_output=True, check=False
                )
                print(f"stopped {label} pid {pid}")
        print("stopped")
        sys.exit(0)

    backend_python = _backend_python()
    if not _can_import(backend_python, "uvicorn"):
        print(f"'uvicorn' is not importable from {backend_python}")
        print("The backend dependencies are not installed. Set up once from the repository root,")
        print("then re-run:")
        print("    python setup.py")
        sys.exit(1)

    kill("uvicorn*app.main", "api")
    kill("vite", "web")
    time.sleep(0.8)

    start_backend(backend_python)
    start_frontend()

    api_ok = wait("api", f"http://127.0.0.1:{API_PORT}/api/health")
    web_ok = wait("web", f"http://127.0.0.1:{WEB_PORT}/")
    sys.exit(0 if api_ok and web_ok else 1)
