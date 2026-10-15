"""The end-to-end drivers, run as tests.

`tools/e2e_part_b.py`, `tools/e2e_part_b_merge.py`, `tools/e2e_features.py`,
`tools/e2e_regime.py`, `tools/e2e_splits.py`, `tools/e2e_scaling.py`,
`tools/e2e_volatility.py`, and `tools/e2e_ui.py`
are scripts because they need a live server and a browser, which `pytest` should not require
of every run. This wraps them so the same checks are reachable from the ordinary test entry
point:

    QM_E2E=1 pytest -m e2e -q

Skipped by default. A run that silently started two servers and a browser would make the
suite's runtime and its failures unpredictable, and neither is a property a unit suite
should have.

What this adds over running the scripts by hand is the server lifecycle: the fixture
restarts both processes first, so the drivers can never be pointed at a server still
running the previous revision of the code — which is exactly how an earlier round of
verification read stale behaviour and reported it as a defect.
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
API_HEALTH = "http://127.0.0.1:8000/api/health"
WEB_ROOT = "http://localhost:5173/"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("QM_E2E") != "1",
        reason="set QM_E2E=1 to run the drivers (starts uvicorn, vite, and a browser)",
    ),
]


def reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


@pytest.fixture(scope="module")
def servers() -> None:
    """Restart both dev processes and wait for them to answer."""
    started = subprocess.run(
        [sys.executable, "tools/serve.py"],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if started.returncode != 0:
        pytest.fail(f"tools/serve.py failed:\n{started.stdout}\n{started.stderr}")
    for label, url in (("api", API_HEALTH), ("web", WEB_ROOT)):
        if not reachable(url):
            pytest.fail(f"{label} did not come up at {url}")


def run_driver(name: str, timeout: int = 900) -> None:
    completed = subprocess.run(
        [sys.executable, f"tools/{name}"],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        # The driver's own report is the useful part of a failure; pytest's default
        # captured-output view buries it behind the assertion line.
        pytest.fail(f"{name} reported failures:\n{completed.stdout}\n{completed.stderr}")


def test_part_b_endpoints(servers: None) -> None:
    """Every cleaning policy, its refusals, a written child, and both export formats."""
    run_driver("e2e_part_b.py")


def test_resample_and_merge_endpoints(servers: None) -> None:
    """Resample, alignment, join kinds, tolerances, column ops, and a written child."""
    run_driver("e2e_part_b_merge.py")


def test_feature_endpoints(servers: None) -> None:
    """The catalogue, every offered feature computed at once, a written child, refusals."""
    run_driver("e2e_features.py")


def test_regime_endpoints(servers: None) -> None:
    """The 202-then-poll handshake, a written regime frame, the crisis override, refusals."""
    run_driver("e2e_regime.py")


def test_split_endpoints(servers: None) -> None:
    """Plan geometry, the stored plan table, the observation column, immutability, refusals."""
    run_driver("e2e_splits.py")


def test_scaling_endpoints(servers: None) -> None:
    """The fit/transform split, the stored parameters, rolling causality, refusals."""
    run_driver("e2e_scaling.py")


def test_volatility_endpoints(servers: None) -> None:
    """The scored walk-forward span, determinism of the simulated paths, conditioning, refusals."""
    run_driver("e2e_volatility.py")


def test_the_ui_in_a_browser(servers: None) -> None:
    """Every sidebar entry, the fetch form, the chart, the grid menu, and the B and C pages."""
    run_driver("e2e_ui.py")
