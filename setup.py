"""One-time setup for Quantmetrics Lab. Run from the repository root:

    python setup.py

Creates the backend virtualenv, installs the backend package and its dev extras,
and installs the frontend dependencies. It never asks you to activate the venv:
every step calls the venv's interpreter by path, so it is safe under PowerShell's
default script-execution policy. It is idempotent — re-running it to pick up a
dependency change is a no-op where things already exist.

After it finishes, start the app with `python dev.py`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"


def _venv_python() -> Path | None:
    for relative in ("Scripts/python.exe", "bin/python"):
        candidate = BACKEND / ".venv" / relative
        if candidate.exists():
            return candidate
    return None


def _run(args: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(str(arg) for arg in args))
    subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)


def _npm() -> str:
    return shutil.which("npm.cmd") or shutil.which("npm") or "npm"


def main() -> int:
    # 1. Backend virtualenv — created once, reused after.
    venv_python = _venv_python()
    if venv_python is None:
        print("creating backend/.venv ...")
        _run([sys.executable, "-m", "venv", str(BACKEND / ".venv")])
        venv_python = _venv_python()
        if venv_python is None:
            print("the virtualenv was created but its interpreter was not found")
            return 1

    # 2. Backend package plus the dev extras (linters, type checker, test runner).
    print("installing the backend package and its dev extras ...")
    _run([venv_python, "-m", "pip", "install", "-e", ".[dev]"], cwd=BACKEND)

    # 3. Frontend dependencies.
    print("installing the frontend dependencies ...")
    _run([_npm(), "install"], cwd=FRONTEND)

    print("\nsetup complete. Start the app with:")
    print("    python dev.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
