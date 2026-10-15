"""Start (or stop) the Quantmetrics Lab dev servers, or run the full gate. Run from
the repository root:

    python dev.py            # start the API and the web, wait until both answer
    python dev.py --stop     # stop both
    python dev.py --verify   # run the whole verification gate (lint, types, tests,
                             # build, servers, and all eight end-to-end drivers)

This does not create anything; it only checks that the backend virtualenv exists and
then hands off to `backend/tools/` — `serve.py` starts the two servers, `verify_all.py`
runs the gates. Running everything under the virtualenv's own interpreter means you
never have to activate the venv, so it works under PowerShell's default script-execution
policy. If the virtualenv is missing it points you at `python setup.py`.

Once both answer: API at http://127.0.0.1:8000, the app at http://localhost:5173
(Vite proxies /api to the API, so the browser stays same-origin).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"


def _venv_python() -> Path | None:
    for relative in ("Scripts/python.exe", "bin/python"):
        candidate = BACKEND / ".venv" / relative
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    args = sys.argv[1:]
    venv_python = _venv_python()
    if venv_python is None:
        print("The backend virtualenv does not exist yet.")
        print("Set up once from the repository root, then try again:")
        print("    python setup.py")
        return 1

    if "--verify" in args:
        # The full gate: verify_all.py drives ruff/mypy/pytest/build and the drivers.
        rest = [arg for arg in args if arg != "--verify"]
        return subprocess.call(
            [str(venv_python), str(BACKEND / "tools" / "verify_all.py"), *rest]
        )

    return subprocess.call(
        [str(venv_python), str(BACKEND / "tools" / "serve.py"), *args]
    )


if __name__ == "__main__":
    sys.exit(main())
