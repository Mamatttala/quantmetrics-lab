"""Run every verification gate in one shot, in the order they depend on each other.

The gates have a chain of preconditions — the drivers need the servers, the UI driver
needs a dataset the fetch driver would have produced — and running them by hand means
eight shell invocations and one chance each to run them against stale state. This runs
them in order and reports a single verdict.

Unit tests come first because they are the fastest signal and need nothing running; the
servers are then started once and shared by every driver after.

    python tools/verify_all.py            # everything
    python tools/verify_all.py --no-lint  # skip ruff/mypy/pytest
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT.parent / "frontend"

results: list[tuple[str, bool, float]] = []


def run(label: str, args: list[str], cwd: Path, timeout: int = 900) -> bool:
    print(f"\n{'=' * 72}\n== {label}\n{'=' * 72}", flush=True)
    started = time.time()
    # The child's stdout is this process's stdout, and when that is a pipe rather than a
    # console Python encodes with the locale's code page — cp1256 here, which cannot
    # represent the characters the drivers print and raised UnicodeEncodeError inside a
    # driver that had already passed every check. The parent reads the output back as
    # UTF-8, so both ends are told the same thing. It is passed explicitly rather than
    # inherited because the parent's own encoding is not what the child will use.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(
        args, cwd=cwd, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env
    )
    elapsed = time.time() - started
    results.append((label, completed.returncode == 0, elapsed))
    print(f"-- {label}: exit {completed.returncode} in {elapsed:.1f}s", flush=True)
    return completed.returncode == 0


def main() -> int:
    lint = "--no-lint" not in sys.argv

    if lint:
        run("ruff check .", [sys.executable, "-m", "ruff", "check", "."], ROOT)
        run("mypy app", [sys.executable, "-m", "mypy", "app"], ROOT)
        run("pytest", [sys.executable, "-m", "pytest", "-q"], ROOT)
        run("npm run typecheck", ["npm.cmd", "run", "typecheck"], FRONTEND)
        run("npm run build", ["npm.cmd", "run", "build"], FRONTEND)

    # The drivers need both processes. serve.py is idempotent: it stops whatever is
    # listening on its patterns and starts fresh, so this cannot accidentally exercise
    # a server running the previous revision of the code.
    if not run("tools/serve.py", [sys.executable, "tools/serve.py"], ROOT, timeout=180):
        print("\nthe servers did not come up; the drivers cannot run")
        return 1

    run("e2e_part_b.py", [sys.executable, "tools/e2e_part_b.py"], ROOT)
    run("e2e_part_b_merge.py", [sys.executable, "tools/e2e_part_b_merge.py"], ROOT)
    run("e2e_features.py", [sys.executable, "tools/e2e_features.py"], ROOT)
    run("e2e_regime.py", [sys.executable, "tools/e2e_regime.py"], ROOT)
    run("e2e_splits.py", [sys.executable, "tools/e2e_splits.py"], ROOT)
    run("e2e_scaling.py", [sys.executable, "tools/e2e_scaling.py"], ROOT)
    run("e2e_volatility.py", [sys.executable, "tools/e2e_volatility.py"], ROOT)
    run("e2e_ui.py", [sys.executable, "tools/e2e_ui.py"], ROOT)

    print(f"\n{'=' * 72}\n== summary\n{'=' * 72}")
    for label, passed, elapsed in results:
        print(f"  [{'ok  ' if passed else 'FAIL'}] {label:<24} {elapsed:6.1f}s")
    failed = [label for label, passed, _ in results if not passed]
    if failed:
        print(f"\n{len(failed)} gate(s) failed: {', '.join(failed)}")
        return 1
    print("\nall gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
