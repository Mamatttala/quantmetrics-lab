"""Click through the running application in a real browser.

The type-checker and the build cannot see a page that throws when it mounts, and the
HTTP drivers cannot see a button that does nothing. This drives Microsoft Edge over the
DevTools Protocol — the same engine the user opens — and asserts on the DOM after real
clicks, so a dead control or a render crash fails here rather than in the browser.

**Why CDP by hand.** Playwright or Puppeteer would be a new dependency for one script.
Edge is already installed on this machine and `websockets` already ships with
`uvicorn[standard]`, so the whole driver is stdlib plus a library that is present anyway.

Requires both processes running: uvicorn on 8000 and `npm run dev` on 5173. The dev
server is what is driven rather than `vite preview`, because only the dev server has the
`/api` proxy configured.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any

from websockets.sync.client import connect

# This driver prints the route it checked as `badge → path`, and the `→` is not encodable
# in this machine's console code page (cp1256). `verify_all.py` sets PYTHONIOENCODING for
# the children it starts; this covers a driver started on its own — which is how the
# header above says to run it — where every check could pass and the run still end in
# UnicodeEncodeError while printing its own verdict.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 127.0.0.1 rather than `localhost`: the dev server binds IPv4, and this machine resolves
# `localhost` to `::1` first, which is how the UI driver would have opened a connection to
# an address nothing was listening on and called it a failed page.
FRONTEND = "http://127.0.0.1:5173"
CDP_PORT = 9222
API = "http://127.0.0.1:8000/api"

EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]

FAILURES: list[str] = []
CONSOLE_ERRORS: list[str] = []
#: Uncaught exceptions, kept apart from the console log. A failed subresource load (a
#: favicon, say) is a console error the page can do nothing about; an exception that
#: escapes a handler is this application's own defect, and the module docstring's promise
#: that a render crash fails here rests on these being failures rather than a printout.
PAGE_EXCEPTIONS: list[str] = []
#: Every HTTP response the page received with a 4xx or 5xx status, as "status url". Printed
#: rather than failed, because several of these are refusals the driver goes out of its way
#: to provoke — the same treatment `CONSOLE_ERRORS` gets. They are listed at all because the
#: console message for a refused fetch is "Failed to load resource: the server responded
#: with a status of 422", which names the status and not the request: a run reporting one
#: could not be diagnosed without going back to the browser.
REFUSALS: list[str] = []

#: The temp profile `launch` created, so the exit path can delete it. See `launch`.
_PROFILE: Path | None = None


def check(label: str, condition: bool, detail: object = "") -> None:
    """`detail` is for the human reading the log and is whatever the evaluation returned —
    a bool from `.disabled`, a count, a list of header names. It is stringified rather
    than concatenated: several call sites pass a bool, and `detail` is only rendered when
    it is truthy, so a *failing* check used to raise TypeError from inside the reporter
    itself and the run ended in a traceback instead of a verdict."""
    mark = "ok  " if condition else "FAIL"
    shown = "" if detail is None or detail == "" else f" — {detail}"
    print(f"  [{mark}] {label}{shown}")
    if not condition:
        FAILURES.append(label)


def find_browser() -> Path:
    for candidate in EDGE_CANDIDATES:
        if candidate.exists():
            return candidate
    print("no Edge or Chrome found")
    sys.exit(1)


class Page:
    """One CDP page target."""

    def __init__(self, ws_url: str) -> None:
        self.ws = connect(ws_url, max_size=64 * 1024 * 1024)
        self.next_id = 0
        self.send("Page.enable")
        self.send("Runtime.enable")
        self.send("Log.enable")
        # `Log.enable` is what makes failed subresource loads and uncaught errors visible;
        # `Runtime.enable` alone reports only exceptions that escape a handler.
        self.send("Network.enable")
        # Headless Edge opens a 500x450 viewport and ignores `--window-size`, and at that
        # width the shell's third grid track — the main pane, `1fr` — collapses to zero:
        # the sidebar and the explorer are fixed-width `auto` tracks that do not shrink,
        # so the page renders into a 52px column and the data grid into 2px. Every check
        # about the grid would then be measuring the window rather than the application.
        # A desktop research tool is driven at a desktop size, so the viewport is set
        # through the protocol, which does take effect, rather than through a launch flag
        # that does not.
        self.send(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1440, "height": 1000, "deviceScaleFactor": 1, "mobile": False},
        )

    def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.next_id += 1
        message_id = self.next_id
        self.ws.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        while True:
            raw = self.ws.recv(timeout=60)
            message = json.loads(raw)
            if message.get("id") == message_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})
            self._observe(message)

    def _observe(self, message: dict[str, Any]) -> None:
        """Record console errors and uncaught exceptions as they arrive."""
        method = message.get("method")
        params = message.get("params") or {}
        if method == "Runtime.exceptionThrown":
            detail = params.get("exceptionDetails") or {}
            text = detail.get("exception", {}).get("description") or detail.get("text", "")
            PAGE_EXCEPTIONS.append(text.splitlines()[0] if text else "unknown")
        elif method == "Log.entryAdded":
            entry = params.get("entry") or {}
            if entry.get("level") == "error":
                CONSOLE_ERRORS.append(f"console: {entry.get('text', '')}")
        elif method == "Network.responseReceived":
            response = params.get("response") or {}
            try:
                status = int(response.get("status") or 0)
            except (TypeError, ValueError):
                return
            if status >= 400:
                url = str(response.get("url", ""))
                REFUSALS.append(f"{status} {url[len(API):] if url.startswith(API) else url}")

    def drain(self, seconds: float = 0.6) -> None:
        """Collect events that arrived while nothing was being awaited."""
        end = time.time() + seconds
        while time.time() < end:
            try:
                self._observe(json.loads(self.ws.recv(timeout=max(0.05, end - time.time()))))
            except TimeoutError:
                return

    def evaluate(self, expression: str) -> Any:
        result = self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
                "userGesture": True,
            },
        )
        if "exceptionDetails" in result:
            detail = result["exceptionDetails"]
            text = (detail.get("exception") or {}).get("description") or detail.get("text")
            raise RuntimeError(f"evaluate failed: {text}")
        return (result.get("result") or {}).get("value")

    def goto(self, path: str) -> None:
        self.send("Page.navigate", {"url": f"{FRONTEND}{path}"})

    def wait_for(self, expression: str, timeout: float = 25.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            try:
                if self.evaluate(expression):
                    return True
            except RuntimeError:
                pass
            time.sleep(0.25)
        return False

    def text(self) -> str:
        return self.evaluate("document.body.innerText") or ""

    def close(self) -> None:
        with suppress(Exception):
            self.ws.close()

    def set_viewport(self, width: int, height: int = 1000) -> None:
        """Resize the emulated browser viewport without navigating away."""
        self.send(
            "Emulation.setDeviceMetricsOverride",
            {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
        )


# --------------------------------------------------------------- page helpers

def shows(page: Page, phrase: str) -> bool:
    """Whether the rendered page says `phrase`, ignoring case.

    Not `phrase in page.text()`. The stylesheet upper-cases most of the interface's
    chrome — `.section__title`, `.meta__key`, `.policy__legend`, `.report__key`,
    `.dtable th`, every `label` — and `innerText` reports text as *rendered*, so a
    heading written "Data quality" is read back as "DATA QUALITY". Comparing without
    case keeps a check about the words on the page from turning into a check about the
    font's letter case, which is not what any of these labels claim.
    """
    return phrase.lower() in (page.text() or "").lower()


# Clicking the first element whose visible text matches. React attaches its listeners to
# the root container, so a synthetic `.click()` is delivered the same way a real one is.
CLICK_BY_TEXT = """
(() => {
  const wanted = %s;
  const nodes = Array.from(document.querySelectorAll('button, a, label'));
  const hit = nodes.find((n) =>
    (n.innerText || '').trim().toLowerCase().includes(wanted.toLowerCase()));
  if (!hit) return false;
  hit.click();
  return true;
})()
"""

# React replaces the `value` setter on the element prototype, so assigning `el.value`
# directly does not notify it. Going through the original descriptor does.
SET_SELECT = """
(() => {
  const el = document.querySelector(%s);
  if (!el) return false;
  const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;
  setter.call(el, %s);
  el.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()
"""

SET_INPUT = """
(() => {
  const el = document.querySelector(%s);
  if (!el) return false;
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
  setter.call(el, %s);
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()
"""

# The same, for a control reached by an expression rather than by a CSS selector — an
# input inside the nth table row has no stable selector of its own.
SET_INPUT_EXPR = """
(() => {
  const el = %s;
  if (!el) return false;
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
  setter.call(el, %s);
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()
"""

CLICK_SELECTOR = """
(() => {
  const el = document.querySelector(%s);
  if (!el) return false;
  el.click();
  return true;
})()
"""

COUNT = "document.querySelectorAll(%s).length"

#: Every `.report__key` on the page, lower-cased. The stylesheet upper-cases these labels
#: (`data.css`), and `innerText` reports text as *rendered*, so the keys arrive as "ROWS
#: IN" and "PRICE SERIES". Comparisons against them are made on this form: the checks are
#: about which facts the report states, not about the letter case the stylesheet chose.
REPORT_KEYS = (
    "Array.from(document.querySelectorAll('.report__key'))"
    ".map((n) => n.innerText.trim().toLowerCase())"
)

#: Bar length in seconds, shortest first. A resample target must be strictly coarser than
#: its source — the backend refuses the other direction, because building four 15-minute
#: bars from one hourly bar fabricates detail no market produced — so the target is looked
#: up on this ladder rather than hard-coded. A hard-coded "4h" is refused outright against
#: a daily base, and the report this driver waits for would then never render.
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}


#: The shortest dataset the D1 page can label. The page always sends the server's own
#: defaults — `min_train_bars` 250 and `volatility_window` 24 — because the sliders it
#: exposes do not reach either, so 274 bars are consumed before the first label exists.
#: A run on anything shorter succeeds and produces an empty dashboard, which is a correct
#: outcome and a useless subject: every result check below would then be measuring the
#: "no labelled rows" panel rather than the gauge, the figures and the diagnostics.
REGIME_MIN_ROWS = 300


#: The schedule the E1 section drives. Small enough that two arch models finish in seconds,
#: and comfortably clear of the *page's own* floor of 30 scored steps. The margin is the
#: point: a step size of 2 would score exactly `ceil(VOL_TEST / VOL_STEP)` = 30 steps, so a
#: single step whose optimiser failed to converge would drop the model under the floor,
#: leave it unscored, and take the pairwise comparison down with it — after a two-minute
#: walk-forward, reported as an application failure. Sixty steps at step 1 means 30 would
#: have to fail, and it exercises the refit-every-bar path the causality argument rests on.
VOL_TRAIN = 150
VOL_TEST = 60
VOL_STEP = 1
VOL_HORIZON = 4
#: Two Family A models, deliberately: they are the fast path, and a pair is the fewest that
#: makes the Diebold–Mariano table exist at all — one model produces no comparison to draw.
VOL_MODELS = ("garch", "gjr_garch")


def coarser_than(timeframe: str | None) -> str | None:
    """The next rung above `timeframe`, or None when it is absent or already the top."""
    seconds = TIMEFRAME_SECONDS.get(timeframe or "")
    if seconds is None:
        return None
    return next((name for name, value in TIMEFRAME_SECONDS.items() if value > seconds), None)


def api(path: str) -> Any:
    with urllib.request.urlopen(f"{API}{path}", timeout=60) as response:
        return json.loads(response.read().decode())


def api_mutation(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    """Call a JSON mutation endpoint used by the browser action checks."""
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{API}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    return None if not raw else json.loads(raw.decode())


def launch(headless: bool = True) -> subprocess.Popen[bytes]:
    close_stale_browser()
    profile = Path(tempfile.mkdtemp(prefix="qml-cdp-"))
    # Kept for the exit path to delete. A browser profile is not small — a run's is tens of
    # megabytes — and `mkdtemp` never removes it, so a driver that only closed its browser
    # still left a directory behind on every run. Twenty-three of them had accumulated here
    # at five gigabytes by the time this was noticed.
    global _PROFILE
    _PROFILE = profile
    args = [
        str(find_browser()),
        f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-gpu",
    ]
    if headless:
        args.append("--headless=new")
    args.append("about:blank")
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_debugger(timeout: float = 30.0) -> str:
    """Return the page target's WebSocket URL."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/list", timeout=2) as r:
                targets = json.loads(r.read().decode())
            for target in targets:
                if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                    return target["webSocketDebuggerUrl"]
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.4)
    raise RuntimeError("the browser never exposed a debuggable page")


def browser_endpoint() -> str | None:
    """The *browser*-level WebSocket URL of whatever holds the debugging port, if anything.

    Distinct from the page target `wait_for_debugger` returns: `Browser.close` is a
    browser-domain method, and the browser endpoint is the only one that accepts it.
    """
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{CDP_PORT}/json/version", timeout=2
        ) as response:
            url = json.loads(response.read().decode()).get("webSocketDebuggerUrl")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    return str(url) if url else None


def close_browser(endpoint: str) -> None:
    """Ask a browser to exit, through its own protocol.

    Nothing is asserted about the answer, because `Browser.close` takes the socket down
    with it: the reply is a connection closing rather than a message, and a browser that
    was already gone is the outcome this wanted anyway.
    """
    with suppress(Exception), connect(endpoint, max_size=1024 * 1024) as socket:
        socket.send(json.dumps({"id": 1, "method": "Browser.close"}))


def close_stale_browser() -> None:
    """Close a browser an earlier run left behind, before launching this run's.

    The debugging port is fixed, and `terminate()` kills the process this driver *started*,
    which on Windows is the launcher — Edge hands the window to a browser process of its
    own that survives it, keeps the port and stays on whatever page it was left on. So the
    port is usually still held when the next run starts, and `wait_for_debugger` asks the
    *port* rather than this run's process: it adopts the stale browser, drives its tab, and
    inherits its console errors. That is where a clean run's four console errors came from
    — a leftover page on the frontend polling while the API restarted underneath it.

    Only a browser showing this application is closed. If something else holds the port,
    it is left alone and `launch` fails the way it always did, which is the honest outcome:
    a driver that silently drove a stranger's browser would be worse than one that refuses.
    """
    endpoint = browser_endpoint()
    if endpoint is None:
        return
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/list", timeout=2) as r:
            targets = json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return
    if not any(str(target.get("url", "")).startswith(FRONTEND) for target in targets):
        return
    print("  (closing a browser left over from an earlier run)")
    close_browser(endpoint)
    # The port is released by the browser's exit, not by the message, so give it a moment
    # before this run's own browser tries to bind it.
    end = time.time() + 10
    while time.time() < end and browser_endpoint() is not None:
        time.sleep(0.2)


# ------------------------------------------------------------------- fixtures

def pick_datasets() -> dict[str, Any]:
    """Choose a dataset that can carry every check below.

    OHLC first, and not merely "a dataset with a timeframe": A2 draws candlesticks, so a
    metrics dataset — which has a `create_time` and no prices — would render the
    "No bars to draw" panel and fail a check about the chart rather than about the app.

    Then a dataset this suite did not write. `parent_dataset_id` is the store's own record
    of that: every write these drivers make forks a child and stamps the parent on it, so a
    row without one is an input — a fetch or an upload — and not an output. This matters
    most for C5. Choosing from the whole listing let the base drift onto whatever the
    *previous run* had saved, and a saved feature set is the one dataset that section
    cannot run against: it already carries the columns the catalogue generates, so selecting
    those features again is refused as a name collision. The refusal is correct and by
    design (`features.py`, "These columns already exist in the dataset"), but the driver
    read it as a broken page and reported eight failures against a working application.

    A second dataset is chosen for D1, because that section needs a different property
    rather than a different column set. See `REGIME_MIN_ROWS`.
    """
    rows = api("/datasets")
    columns = {
        row["dataset_id"]: {c["name"] for c in api(f"/datasets/{row['dataset_id']}")["columns"]}
        for row in rows
    }
    ohlc = {"open", "high", "low", "close"}
    candidates = [
        row
        for row in rows
        if ohlc.issubset(columns[row["dataset_id"]]) and row["row_count"] > 4
    ]
    sources = [row for row in candidates if row.get("parent_dataset_id") is None]
    if not sources:
        # A store holding nothing but derived datasets is not a reason to fail: the checks
        # below are about the application, and an output is a poorer subject but a usable
        # one. Which of the two lists was used is printed, so a surprising failure can be
        # read against the dataset it was measured on.
        print(f"  (every one of the {len(candidates)} OHLC datasets is derived; using one)")
        sources = candidates
    # A base already at the top of the ladder — a weekly frame — has no legal resample
    # target, and the resample half of B4 is checked below. Prefer one with a rung above
    # it, and fall back to whatever is there so the failure names the real reason.
    klines = next((r for r in sources if coarser_than(r.get("timeframe"))), None) or (
        sources[0] if sources else None
    )
    if klines is None:
        print("no OHLC dataset with more than 4 rows; run a klines fetch first")
        print(f"  ({len(rows)} dataset(s) in the catalog)")
        sys.exit(1)
    # No `sys.exit` when this one is absent. The resample, cleaning, features and grid
    # sections all run on whatever `klines` turned out to be, and a store too small for a
    # regime fit is a *thinner* store, not a broken application — so D1 reports the gap and
    # checks the page's own refusal instead of ending the run here.
    regime = next(
        (
            row
            for row in sources
            if row.get("timeframe") is not None and row["row_count"] >= REGIME_MIN_ROWS
        ),
        None,
    )
    return {"klines": klines, "regime": regime}


#: The two operations whose datasets are artifacts rather than frames: a split plan is a
#: table of row ranges and a scaler-parameter table holds `center_`/`scale_` per feature.
#: Both are deliberately hidden from C6's and C7's dataset pickers, and re-listed here so
#: the check for "a dataset with no plan" does not land on one of them.
ARTIFACT_OPERATIONS = {"split", "scaler_params"}


def dataset_without_a_plan(rows: list[dict[str, Any]], skip: str) -> dict[str, Any] | None:
    """A dataset C6 has never been pointed at, for the refusal check.

    Not `klines`: C6 runs before C7 in this driver and leaves two plans on it, and a plan
    outlives the run — so a second run of this driver would find `klines` planned and a
    check written against it would pass for the wrong reason on the first run and fail on
    the second. Asking the server which datasets have no plan is the only form of this
    question that stays true across runs.

    Bounded to the first 25 candidates: the listing holds every dataset the drivers have
    ever written, and this is a search for one example, not an inventory. The newest
    datasets are the ones this run's own C5 section just saved, and none of those has a
    plan, so the search ends almost immediately in practice.
    """
    seen = 0
    for row in rows:
        if row["dataset_id"] == skip or row["row_count"] <= 0:
            continue
        if (row.get("operation") or "") in ARTIFACT_OPERATIONS:
            continue
        seen += 1
        if seen > 25:
            return None
        if not api(f"/splits/plans?dataset_id={row['dataset_id']}"):
            return row
    return None


def volatility_subject(preferred: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """A dataset in the catalog that can carry the E1 section's schedule, or None.

    Asked of `/volatility/options` rather than inferred from the row count, because the
    row count is not the number that matters: the walk-forward runs over the *return*
    series, which is one bar shorter than the frame, and a dataset with no `close` column
    has no price source to derive one from. Reading `row_count` here would pick a subject
    the page then refuses, and the run checks below would report a broken application
    against a store that simply holds shorter frames.

    The dataset `e2e_volatility.py` uploads would do, but that driver is a sibling rather
    than a precondition: run on its own, this section has to find its own subject. So
    ``preferred`` — the dataset the rest of this section already runs against — is tried
    first, and the scan below is the fallback for a store where that one is too short.

    Returns the options payload itself, which carries everything the checks below compare
    against — the usable return count, the warnings, and the models offered.
    """
    needed = VOL_TRAIN + VOL_TEST

    def options_for(dataset_id: str) -> dict[str, Any] | None:
        try:
            payload = api(f"/volatility/options?dataset_id={dataset_id}")
        except (urllib.error.HTTPError, urllib.error.URLError):
            return None
        # Feasible is not required: the page derives a default schedule the dataset can run,
        # so what has to hold is that *this section's* schedule fits, plus the absence of an
        # error-level warning — `too_short` or `no_close`, either of which keeps the Run
        # button disabled however the windows are set.
        if payload["usable_returns"] < needed:
            return None
        if any(entry["level"] == "error" for entry in payload["warnings"]):
            return None
        return payload

    if preferred is not None:
        payload = options_for(str(preferred["dataset_id"]))
        if payload is not None:
            return payload

    seen = 0
    for row in api("/datasets"):
        if row["row_count"] < needed + 1:
            continue
        # The column check comes before the options call, and before the budget below is
        # spent, and it is what lets this scan find anything at all. Most of the catalog is
        # *derived* — hundreds of forecast, regime, scaling and feature frames — and none of
        # those carries a price, so each one used to consume one of the thirty slots while
        # `usable_returns` was never reached. The scan then returned None and the E1 section
        # printed that its run was skipped, which is how a page whose result path had never
        # been exercised came to look green: the skip is not a failure, so nothing failed.
        columns = {c["name"] for c in api(f"/datasets/{row['dataset_id']}")["columns"]}
        if "close" not in columns and "log_return" not in columns:
            continue
        seen += 1
        if seen > 30:
            return None
        payload = options_for(str(row["dataset_id"]))
        if payload is not None:
            return payload
    return None


def choose_scaling_plan(page: Page, pattern: str) -> str | None:
    """Select the first plan in C7's dropdown whose text matches `pattern`, as a regex.

    The plan's id is a UUID this driver has no way to know, so the option is found by its
    label — `{name} · {split_type} · {folds} fold(s) · {rows} rows` — and the select is set
    through the prototype's own setter, exactly as `SET_SELECT` does for a static value.
    """
    return page.evaluate(
        "(() => {"
        "  const s = document.querySelector('#scaling-plan');"
        "  if (!s) return null;"
        f"  const opt = Array.from(s.options).find((o) => o.value && {pattern}.test(o.innerText));"
        "  if (!opt) return null;"
        "  const setter = Object.getOwnPropertyDescriptor("
        "HTMLSelectElement.prototype, 'value').set;"
        "  setter.call(s, opt.value);"
        "  s.dispatchEvent(new Event('change', { bubbles: true }));"
        "  return opt.innerText;"
        "})()"
    )


def main() -> int:
    browser = launch()
    ui_disposable_id: str | None = None
    try:
        page = Page(wait_for_debugger())
        datasets = pick_datasets()
        klines = datasets["klines"]
        regime = datasets["regime"]
        # A child of the fixture is safe to delete: it has no descendants of its own,
        # while deleting the fixture itself would break the preparation checks below.
        disposable = api_mutation(
            "POST",
            f"/datasets/{klines['dataset_id']}/derive",
            {
                "name": "e2e-ui-action-delete",
                "operation": "edit",
                "edits": [{"row": 0, "column": "close", "value": 1.0}],
                "column_ops": [],
            },
        )
        ui_disposable_id = str(disposable["dataset_id"])
        # The base is printed with its size and category: several checks below read the
        # numbers on the page rather than a fixed expectation, so a failure is only
        # readable against the dataset it was measured on.
        print(
            f"using {klines['name']} ({klines['dataset_id']}) "
            f"{klines['row_count']} rows @ {klines.get('timeframe')} "
            f"[{klines.get('category')}]"
        )
        print(
            "regime on "
            + (
                f"{regime['name']} ({regime['dataset_id']}) {regime['row_count']} rows "
                f"@ {regime.get('timeframe')}"
                if regime is not None
                else "nothing — no OHLC dataset with a timeframe and "
                f"{REGIME_MIN_ROWS}+ rows, so D1 checks the refusal instead"
            )
        )
        # ------------------------------------------------------------ shell
        print("== shell ==")
        page.goto("/")
        check("the app mounts", page.wait_for("document.querySelector('.sidebar') !== null"))
        check("'/' redirects to /fetch", page.wait_for("location.pathname === '/fetch'"),
              page.evaluate("location.pathname"))
        nav = page.evaluate(
            "Array.from(document.querySelectorAll('.navitem__badge')).map((n) => n.innerText)"
        )
        check("every registry entry is in the sidebar",
              nav == ["DM", "A1", "A2", "B3", "B4", "C5", "C6", "C7", "D1", "E1"], f"{nav}")

        check("the sidebar collapses",
              bool(page.evaluate(CLICK_SELECTOR % json.dumps(".sidebar__collapse")))
              and page.wait_for("document.querySelector('.sidebar--collapsed') !== null"))
        check("the sidebar expands again",
              bool(page.evaluate(CLICK_SELECTOR % json.dumps(".sidebar__collapse")))
              and page.wait_for("document.querySelector('.sidebar--collapsed') === null"))

        # CSS owns the narrow-layout collapse so it also works when the persisted React
        # state says "expanded". Check the geometry at the breakpoint and below it rather
        # than only checking that a media rule exists: the header controls were previously
        # left in a horizontal 62px row inside the 52px rail.
        print("== responsive shell widths ==")
        for width in (1200, 960, 600):
            page.set_viewport(width)
            ready = page.wait_for(f"window.innerWidth === {width}")
            geometry = page.evaluate(
                "(() => {"
                "  const sidebar = document.querySelector('.sidebar');"
                "  const header = document.querySelector('.sidebar__header');"
                "  const actions = document.querySelector('.sidebar__header-actions');"
                "  if (!sidebar || !header || !actions) return null;"
                "  const buttons = Array.from(actions.querySelectorAll('button'));"
                "  return {"
                "    width: Math.round(sidebar.getBoundingClientRect().width),"
                "    overflow: sidebar.scrollWidth > sidebar.clientWidth,"
                "    headerDirection: getComputedStyle(header).flexDirection,"
                "    actionsDirection: getComputedStyle(actions).flexDirection,"
                "    buttonCount: buttons.length,"
                "    actionsWidth: Math.round(actions.getBoundingClientRect().width),"
                "    buttonWidths: buttons.map((button) => Math.round("
                "button.getBoundingClientRect().width))"
                "  };"
                "})()"
            )
            expected_collapsed = width <= 960
            buttons_fit = isinstance(geometry, dict) and all(
                button_width <= geometry["actionsWidth"]
                for button_width in geometry["buttonWidths"]
            )
            check(
                f"{width}px sidebar geometry is responsive",
                ready
                and isinstance(geometry, dict)
                and geometry["width"] == (52 if expected_collapsed else 220)
                and geometry["buttonCount"] == 2
                and not geometry["overflow"],
                f"{geometry}",
            )
            check(
                f"{width}px sidebar header controls fit the rail",
                isinstance(geometry, dict)
                and geometry["headerDirection"] == ("column" if expected_collapsed else "row")
                and geometry["actionsDirection"] == ("column" if expected_collapsed else "row")
                and (not expected_collapsed or buttons_fit),
                f"{geometry}",
            )

        page.set_viewport(1440)
        page.wait_for("window.innerWidth === 1440")

        # ------------------------------------------------ every sidebar entry
        # The literal reading of "debug every menu button": press each one and see that
        # the route changes and a titled page renders behind it.
        print("== sidebar · every entry ==")
        expected = [
            ("DM", "/data", "Data Management"),
            ("A1", "/fetch", "Raw Data Fetch"),
            ("A2", "/view", "Raw Data View"),
            ("B3", "/cleaning", "Cleaning"),
            ("B4", "/resample", "Resample & Merge"),
            ("C5", "/features", "Features Engineering"),
            ("C6", "/splits", "Chronological Splits"),
            ("C7", "/scaling", "Scaling"),
            ("D1", "/regime", "Regime & Trend Detection"),
            # The sidebar's label for this entry is "Volatility Forecast" and the page's own
            # heading is "Volatility Forecasting". The title checked is the one `PageHeader`
            # renders into `.page__title`, which is the page's, not the registry's — the same
            # distinction D1 has ("Regime & Trend" in the rail, "...Detection" on the page).
            ("E1", "/volatility", "Volatility Forecasting"),
        ]
        for badge, path, title in expected:
            badge_js = json.dumps(badge)
            clicked = page.evaluate(
                "(() => {"
                "  const link = Array.from(document.querySelectorAll('.navitem'))"
                "    .find((n) =>"
                f"      n.querySelector('.navitem__badge').innerText.trim() === {badge_js});"
                "  if (!link) return false;"
                "  link.click();"
                "  return true;"
                "})()"
            )
            landed = page.wait_for(f"location.pathname === {json.dumps(path)}", timeout=10)
            # The path changes with the router's navigation; the heading is rendered a
            # frame or two later. Reading `.page__title` once, the moment the path settled,
            # sampled the *previous* page's title about one run in ten — "A1 → /fetch —
            # title 'Data Management'" was reported against a sidebar that had navigated
            # correctly — so the check waits for the heading instead of sampling it, and
            # reports whichever title it last saw so a real mismatch still names what was
            # on screen.
            titled = page.wait_for(
                "(() => { const h = document.querySelector('.page__title');"
                f" return h !== null && h.innerText === {json.dumps(title)}; }})()",
                timeout=10,
            )
            heading = page.evaluate(
                "(() => { const h = document.querySelector('.page__title');"
                " return h ? h.innerText : null; })()"
            )
            check(f"{badge} → {path}", clicked and landed and titled,
                  f"path={page.evaluate('location.pathname')} title={heading!r}")
            # §13's promise, kept as a negative now that every entry is built: a route that
            # renders a "not implemented" bar instead of a page is a failure. It is read
            # here, on the document the `titled` wait above already proved had rendered.
            # What this replaced navigated to /splits and /scaling again and asserted they
            # said exactly that — which was correct until C6 and C7 were built, and then
            # failed a correct application.
            body = page.text().lower()
            check(f"{badge} {path} is a built page, not a placeholder",
                  "not implemented" not in body, body[:80].replace("\n", " "))

        page.goto("/nope")
        check("an unknown address renders the not-found page",
              page.wait_for("document.body.innerText.includes('Not found')"),
              page.evaluate("location.pathname"))

        # ----------------------------------------------- Data Explorer actions
        print("== Data Explorer · row actions ==")
        page.goto("/data")
        check(
            "the explorer renders dataset rows with per-row actions",
            page.wait_for("document.querySelector('.explorer__row') !== null"),
        )
        row_selector = f".explorer__row[data-dataset-id={json.dumps(klines['dataset_id'])}]"
        row_selector_js = json.dumps(row_selector)
        check(
            "each row exposes rename and delete icon buttons",
            page.evaluate(
                f"(() => {{ const row = document.querySelector({row_selector_js});"
                " return !!row && !!row.querySelector('.icon-btn:not(.danger)')"
                " && !!row.querySelector('.icon-btn.danger'); })()"
            ),
        )

        renamed_name = f"{klines['name']} (UI renamed)"
        page.evaluate(f"window.prompt = () => {json.dumps(renamed_name)}")
        check(
            "rename action calls the backend and refreshes the explorer",
            bool(page.evaluate(
                f"(() => {{ const row = document.querySelector({row_selector_js});"
                " const button = row && row.querySelector('.icon-btn:not(.danger)');"
                " if (!button) return false; button.click(); return true; })()"
            ))
            and page.wait_for(
                f"document.querySelector({row_selector_js})?.querySelector('.explorer__item')?"
                f".innerText === {json.dumps(renamed_name)}"
            ),
            renamed_name,
        )
        page.evaluate(f"window.prompt = () => {json.dumps(klines['name'])}")
        page.evaluate(
            f"document.querySelector({row_selector_js})?.querySelector('.icon-btn:not(.danger)')?.click()"
        )
        check(
            "a renamed dataset can be restored from the same row",
            page.wait_for(
                f"document.querySelector({row_selector_js})?.querySelector('.explorer__item')?"
                f".innerText === {json.dumps(klines['name'])}"
            ),
        )

        for label, path in (("view", "view"), ("clean", "cleaning"), ("merge", "resample")):
            page.goto("/data")
            check(
                f"the {label} quick action opens its module",
                page.wait_for("document.querySelector('.explorer__row') !== null")
                and bool(page.evaluate(
                    f"(() => {{ const row = document.querySelector({row_selector_js});"
                    " const button = Array.from(row?.querySelectorAll('.explorer__quick') || [])"
                    f" .find((candidate) => candidate.innerText.trim() === {json.dumps(label)});"
                    " if (!button) return false; button.click(); return true; })()"
                ))
                and page.wait_for(
                    f"location.pathname === '/{path}/{klines['dataset_id']}'"
                ),
            )

        page.goto("/data")
        assert ui_disposable_id is not None
        deleted_id = ui_disposable_id
        disposable_selector = f".explorer__row[data-dataset-id={json.dumps(deleted_id)}]"
        disposable_selector_js = json.dumps(disposable_selector)
        page.evaluate("window.confirm = () => true")
        check(
            "delete action removes the dataset from the explorer",
            page.wait_for(f"document.querySelector({json.dumps(disposable_selector)}) !== null")
            and bool(page.evaluate(
                f"document.querySelector({disposable_selector_js})?"
                ".querySelector('.icon-btn.danger')?.click(); true"
            ))
            and page.wait_for(
                f"document.querySelector({json.dumps(disposable_selector)}) === null"
            ),
            deleted_id,
        )
        catalog_missing = False
        try:
            api(f"/datasets/{deleted_id}")
        except urllib.error.HTTPError as error:
            catalog_missing = error.code == 404
        check(
            "delete action removes the dataset from the catalog",
            catalog_missing,
            deleted_id,
        )
        ui_disposable_id = None

        # ------------------------------------------------------- A1 · fetch
        print("== A1 · raw data fetch ==")
        page.goto("/fetch")
        check("the fetch form renders", page.wait_for("document.querySelector('#market') !== null"))
        markets = page.evaluate(
            "Array.from(document.querySelectorAll('#market option')).map((o) => o.value)"
        )
        check("every market is offered", len(markets) >= 3, f"{markets}")

        # The field is prefilled with BTCUSDT, so the disabled state is reached by
        # clearing it rather than assumed on first paint.
        SUBMIT = "document.querySelector('form button[type=submit]')"
        page.evaluate(SET_INPUT % (json.dumps("#symbol"), json.dumps("")))
        check("Fetch is disabled with an empty symbol",
              page.evaluate(f"{SUBMIT}.disabled") is True,
              page.evaluate(f"{SUBMIT}.innerText"))

        page.evaluate(SET_INPUT % (json.dumps("#symbol"), json.dumps("BTCUSDT")))
        check("Fetch enables once a symbol is typed",
              page.evaluate(f"{SUBMIT}.disabled") is False)

        page.evaluate(SET_SELECT % (json.dumps("#market"), json.dumps("equity_spot")))
        check("choosing equity hides the crypto-only controls",
              page.wait_for("document.querySelector('#dtype') === null"))
        page.evaluate(SET_SELECT % (json.dumps("#market"), json.dumps("crypto_spot")))
        check("choosing crypto restores them",
              page.wait_for("document.querySelector('#dtype') !== null"))

        # Funding is a futures-only product, and the data-type select offers exactly the
        # products the chosen market serves (`offeredTypes`): under spot it holds klines
        # alone. Assigning `#dtype` a value its `<select>` does not offer changes nothing,
        # so the check below would have failed against a page that was behaving correctly.
        # The market is switched first, and the option's presence asserted before it is
        # used, so the check is about the timeframe control hiding and nothing else.
        page.evaluate(SET_SELECT % (json.dumps("#market"), json.dumps("crypto_futures")))
        check("futures offers funding",
              page.wait_for("document.querySelector('#dtype option[value=funding]') !== null"))
        page.evaluate(SET_SELECT % (json.dumps("#dtype"), json.dumps("funding")))
        check("choosing funding hides the timeframe control",
              page.wait_for("document.querySelector('#tf') === null"))
        page.evaluate(SET_SELECT % (json.dumps("#dtype"), json.dumps("klines")))
        check("choosing klines restores it",
              page.wait_for("document.querySelector('#tf') !== null"))
        page.evaluate(SET_SELECT % (json.dumps("#market"), json.dumps("crypto_spot")))

        # -------------------------------------------------------- A2 · view
        print("== A2 · raw data view ==")
        page.goto(f"/view/{klines['dataset_id']}")
        check("the chart renders", page.wait_for("document.querySelector('.chart') !== null"))
        check("candles reach the canvas",
              page.wait_for("document.querySelector('.chart canvas') !== null"))
        check("the quality panel lists every check",
              page.evaluate(COUNT % json.dumps(".meta__item")) >= 9,
              f"{page.evaluate(COUNT % json.dumps('.meta__item'))}")
        check("a quality report is present",
              shows(page, "Data quality"))

        page.evaluate(CLICK_BY_TEXT % json.dumps("Structural anomalies"))
        page.drain()
        check("the anomaly overlay can be switched on",
              page.evaluate("document.querySelector('.chip input').checked") is True)

        # ----------------------------------------------------- DM · manage
        print("== DM · data management ==")
        page.goto(f"/data/{klines['dataset_id']}")
        check("the 5+5 preview renders",
              page.wait_for("document.querySelector('.preview table') !== null"),
              f"{page.evaluate(COUNT % json.dumps('.preview table tbody tr'))} preview rows")
        check("the grid loads rows",
              page.wait_for("document.querySelectorAll('.ag-row').length > 0", timeout=30),
              f"{page.evaluate(COUNT % json.dumps('.ag-row'))} rows")

        # The toolbar holds several `.mono` spans; the row counter is the one written as
        # "0–100 of 5,000", so match on that shape rather than on position.
        RANGE = (
            "Array.from(document.querySelectorAll('.grid__toolbar .mono'))"
            ".map((n) => n.innerText).find((t) => t.includes(' of '))"
        )
        first_page = page.evaluate(RANGE)
        check("the toolbar counts rows against the total",
              first_page is not None and " of " in first_page, f"{first_page!r}")

        # Whether `Next` has anywhere to go depends on the dataset this driver happened to
        # pick, which it does not choose: the first OHLC dataset in the store can be
        # smaller than one page — 50 rows is the smallest page size, and 33 rows is not
        # unusual — and clicking `Next` there correctly does nothing. The old check
        # asserted that the range *changes* on a click, which is a property of the store
        # rather than of the pager. The pager's own page count now decides which of the
        # two correct behaviours is asserted, and both branches assert something real:
        # either the arrow moves the window, or it is disabled because there is no
        # second page to move to.
        NEXT = (
            "Array.from(document.querySelectorAll('.grid__toolbar button'))"
            ".find((b) => b.innerText.includes('Next'))"
        )
        PREV = (
            "Array.from(document.querySelectorAll('.grid__toolbar button'))"
            ".find((b) => b.innerText.includes('Prev'))"
        )
        pager = page.evaluate(
            "(() => {"
            "  const t = Array.from(document.querySelectorAll('.grid__toolbar .mono'))"
            "    .map((n) => n.innerText.trim()).find((s) => s.startsWith('page '));"
            "  return t || null;"
            "})()"
        )
        check("the pager reports how many pages there are", pager is not None, f"{pager!r}")
        if pager is not None and pager.endswith("/ 1"):
            check("a one-page dataset disables both arrows",
                  page.evaluate(f"{NEXT}.disabled") is True
                  and page.evaluate(f"{PREV}.disabled") is True,
                  f"{pager}")
        else:
            page.evaluate(f"{NEXT}.click()")
            check("Next advances the page",
                  page.wait_for(f"{RANGE} !== {json.dumps(first_page)}"),
                  f"{first_page!r} -> {page.evaluate(RANGE)!r}")
            page.evaluate(f"{PREV}.click()")
            page.drain()

        # The count the grid operations are measured against is ag-Grid's own
        # `aria-colcount`, not the number of `.ag-header-cell` elements in the DOM.
        # ag-Grid virtualises columns horizontally, so only the columns that fit the
        # viewport are rendered: a column appended at the end is genuinely in the model
        # and absent from the DOM, and a check counting rendered header cells would
        # report a working "Add column" as broken. `aria-colcount` is
        # `ColumnModel.getCols().length` — the grid's own count, minus hidden columns —
        # and does not depend on how wide the window is.
        COLCOUNT_GRID = "Number(document.querySelector('.ag-root').getAttribute('aria-colcount'))"
        # Headless Chrome answers `window.prompt` with null unless a dialog handler is
        # installed, which would make Add column and Rename column silently do nothing
        # and let their checks pass against a page that never changed. Stubbing it is
        # what turns those two into real assertions.
        page.evaluate("window.prompt = () => 'probe_column'")
        before_columns = page.evaluate(COLCOUNT_GRID)
        page.evaluate(CLICK_BY_TEXT % json.dumps("Add column"))
        check("Add column adds a column",
              page.wait_for(f"{COLCOUNT_GRID} > {before_columns}")
              # The notice is what names the column the prompt returned, so the check
              # cannot pass on a page whose prompt was ignored.
              and shows(page, "probe_column"),
              f"{before_columns} -> {page.evaluate(COLCOUNT_GRID)} columns")

        # The header menu: a right-click on a column header must open the three
        # operations the spec asks for.
        #
        # The cell is chosen by having text, not by being first: ag-Grid renders a
        # selection-checkbox header with no column behind it, and a menu opened on that
        # one would be about nothing.
        HEADER_CELL = (
            "Array.from(document.querySelectorAll('.ag-header-cell'))"
            ".find((c) => c.innerText.trim())"
        )
        RIGHT_CLICK_HEADER = (
            f"(() => {{ const cell = {HEADER_CELL}; if (!cell) return false;"
            "  const box = cell.getBoundingClientRect();"
            "  cell.dispatchEvent(new MouseEvent('contextmenu', {"
            "    bubbles: true, cancelable: true,"
            "    clientX: box.left + 5, clientY: box.top + 5 }));"
            "  return true; })()"
        )
        opened = page.evaluate(RIGHT_CLICK_HEADER)
        # The menu is rendered by React on the next commit, not synchronously by the event,
        # so it is waited for rather than sampled. Reading it on the following round trip
        # usually works and occasionally does not — and when it did not, the check below
        # raised `TypeError: 'NoneType' object is not subscriptable` from `menu_items[:3]`
        # and the whole run ended there, so the sections after this one never ran. A menu
        # that fails to open is a finding; it is not a reason to lose the rest of the log.
        menu_shown = page.wait_for("document.querySelector('.ctxmenu') !== null", timeout=10)
        menu_items = (
            page.evaluate(
                "Array.from(document.querySelector('.ctxmenu').querySelectorAll('.ctxmenu__item'))"
                ".map((n) => n.innerText)"
            )
            if menu_shown
            else None
        )
        # The three operations the spec asks for must lead the menu: rename, remove,
        # select all. The header menu also carries clipboard items, "Hide column" and
        # "Add empty column", so an exact-equality assertion would fail on a correct
        # page; comparing the leading three still catches a missing or reordered
        # rename/remove/select.
        check("right-clicking a header opens the column menu",
              opened
              and (menu_items or [])[:3] == ["Rename column…", "Remove column", "Select all"],
              f"{menu_items}")

        # A second stub value: reusing "probe_column" would collide with the column just
        # added, and the rename would be refused as a duplicate — passing the check for
        # the wrong reason.
        # The menu is opened on the first *rendered* header cell, so the rename lands on a
        # column that is in the DOM and the new name can be read back from it.
        after_add = page.evaluate(COLCOUNT_GRID)
        page.evaluate("window.prompt = () => 'renamed_col'")
        page.evaluate(
            "(() => { const m = document.querySelector('.ctxmenu__item'); if (m) m.click(); })()"
        )
        page.drain()
        check("Rename column… applies the new name",
              shows(page, "Renamed")
              and page.evaluate(
                  "Array.from(document.querySelectorAll('.ag-header-cell'))"
                  ".some((c) => c.innerText.includes('renamed_col'))"
              ),
              page.evaluate(
                  "Array.from(document.querySelectorAll('.ag-header-cell'))"
                  ".map((c) => c.innerText.trim()).join(' | ').slice(0, 140)"
              ))

        # Re-open the menu on the same header and take "Hide column" this time.
        page.evaluate(RIGHT_CLICK_HEADER)
        page.wait_for("document.querySelector('.ctxmenu') !== null", timeout=10)
        page.drain()
        hidden = page.evaluate(
            "(() => {"
            "  const item = Array.from(document.querySelectorAll('.ctxmenu__item'))"
            "    .find((n) => n.innerText.includes('Hide column'));"
            "  if (!item) return false;"
            "  item.click();"
            "  return true;"
            "})()"
        )
        page.drain()
        check("Hide column removes the header and offers to restore it",
              hidden and page.wait_for(f"{COLCOUNT_GRID} === {after_add - 1}")
              and shows(page, "hidden"),
              f"{after_add} -> {page.evaluate(COLCOUNT_GRID)} columns")

        # The hidden column's own way back. It is a button that only exists while something
        # is hidden, so pressing it here is the only chance this run gets.
        page.evaluate(CLICK_BY_TEXT % json.dumps("Show all columns"))
        check("Show all columns restores the hidden column",
              page.wait_for(f"{COLCOUNT_GRID} === {after_add}"),
              f"{page.evaluate(COLCOUNT_GRID)} columns")

        # ------------------------------------------------------ DM · sorting
        # Sorting is server-side — `GET /rows?sort=close:asc` — so a click on a header has
        # to come back with a reordered frame and a toolbar that says what it is sorted by.
        # A sorted grid is also the one state that locks editing, and "Clear sort" exists to
        # undo exactly that, so both halves are pressed.
        #
        # The click lands on `.ag-header-cell-text`, not on the `.ag-header-cell` around it:
        # ag-Grid binds the sort handler to the header *label*
        # (`sortSvc.setupHeader(this, column, this.eLabel)`), so a click dispatched on the
        # wrapper bubbles up and away from a descendant's listener and nothing sorts. A
        # check that clicked the cell would wait out its timeout against a working grid.
        # The text span is also where a real click lands.
        SORTED_BY = (
            "(() => {"
            "  const n = Array.from(document.querySelectorAll('.grid__toolbar .mono'))"
            "    .map((m) => m.innerText.trim()).find((s) => s.startsWith('sorted by'));"
            "  return n || null;"
            "})()"
        )
        refusals_before = len(REFUSALS)
        page.evaluate(
            "(() => {"
            "  const label = Array.from(document.querySelectorAll('.ag-header-cell-text'))"
            "    .find((n) => n.innerText.trim().toLowerCase() === 'close');"
            "  if (!label) return false;"
            "  label.click();"
            "  return true;"
            "})()"
        )
        check("clicking a header sorts the frame and says by what",
              page.wait_for(f"{SORTED_BY} !== null", timeout=20),
              f"{page.evaluate(SORTED_BY)!r}")
        # The toolbar's own text comes from local state, so it would read "sorted by"
        # whether or not the server accepted the request — a refused sort would leave the
        # grid showing an unordered frame under a label claiming otherwise. The `/rows`
        # request is the half that proves a sort was applied, and a 422 is the only trace
        # a refusal leaves on the page.
        page.drain(1.0)
        refused_sort = [item for item in REFUSALS[refusals_before:] if "/rows" in item]
        check("the sort request was served, not refused", not refused_sort, f"{refused_sort}")
        check("a sorted grid locks editing and says how to unlock it",
              shows(page, "editing locked while sorted"),
              page.evaluate(
                  "(() => { const t = document.body.innerText;"
                  "  const i = t.indexOf('locked while sorted');"
                  "  return i < 0 ? null : t.slice(i, i + 40); })()"
              ))
        page.evaluate(CLICK_BY_TEXT % json.dumps("Clear sort"))
        check("Clear sort unlocks editing again",
              page.wait_for(f"{SORTED_BY} === null")
              and not shows(page, "editing locked while sorted"))

        # ---------------------------------------------------- DM · page size
        # What the rows-per-page control *shows* depends on how many rows this run's dataset
        # holds — 49 here, and 50 is the smallest page size offered — so the check branches
        # on the total the toolbar reports rather than asserting a row count that belongs to
        # the store. `total_rows` is 0 only when the toolbar said nothing that parses, which
        # is a failure in its own right and is asserted as one rather than read as "a
        # dataset with no rows, so nothing should widen".
        #
        # `size_before` is a *settled* range too, and waiting for it is not symmetry with the
        # wait below — it is the fix for a real failure this section produced. The step
        # before it is `Clear sort`, which refetches the window; reading the toolbar the
        # instant the sort label cleared sampled "loading…", which contains no " of " and so
        # parsed as `total_rows == 0`, and the branch that reads "this dataset has fewer rows
        # than one page" was taken against a toolbar that had simply not answered yet. It
        # reported a failure against a page whose very next sample read "0–49 of 49".
        settled_before = page.wait_for(
            f"(() => {{ const t = {RANGE};"
            " return typeof t === 'string' && t.includes(' of '); })()",
            timeout=20,
        )
        size_before = page.evaluate(RANGE) or ""
        total_rows = (
            int(size_before.split(" of ")[-1].replace(",", "")) if " of " in size_before else 0
        )
        page.evaluate(
            SET_SELECT % (json.dumps("select[aria-label='Rows per page']"), json.dumps("100"))
        )
        page.drain()
        # Both branches read a *settled* range. The control shows "loading…" while the new
        # window is in flight, and a sample taken then compares that placeholder against the
        # old text — reporting a widening that never happened, or a failure that did not.
        settled = page.wait_for(
            f"(() => {{ const t = {RANGE};"
            " return typeof t === 'string' && t.includes(' of '); })()",
            timeout=20,
        )
        size_after = page.evaluate(RANGE)
        if total_rows > 100:
            check("a larger page size widens the window",
                  settled_before and settled and size_after != size_before,
                  f"{size_before!r} -> {size_after!r}")
        else:
            check("a page size above the row count leaves the window at every row",
                  settled_before and total_rows > 0 and settled and size_after == size_before,
                  f"{size_before!r} -> {size_after!r}")

        # ------------------------------------------------------ DM · editing
        # The cell menu, which the header menu cannot stand in for: its items act on a cell,
        # and "Clear cell" is the editing path this suite drives. It calls the same `onEdit`
        # the inline editor calls on commit, and it is reachable by a click — where ag-Grid's
        # inline editor would have to be opened with a synthetic double-click and then typed
        # into through its own value API, which a check cannot distinguish from an editor
        # that never opened.
        #
        # The cell is the `close` column's rather than simply the first one with text. The
        # first cell on screen is `open_time`, and blanking the time column would test the
        # menu by damaging the dataset's own timeline — a save the server may refuse for a
        # reason that has nothing to do with the menu. A float column can be blanked and
        # still leave a frame every later step can read; `col-id` is the attribute ag-Grid
        # stamps on each cell with the column's id, which is the field name here.
        RIGHT_CLICK_CELL = (
            "(() => {"
            "  const cell = Array.from(document.querySelectorAll('.ag-cell'))"
            "    .find((c) => c.getAttribute('col-id') === 'close');"
            "  if (!cell) return false;"
            "  const box = cell.getBoundingClientRect();"
            "  cell.dispatchEvent(new MouseEvent('contextmenu', {"
            "    bubbles: true, cancelable: true,"
            "    clientX: box.left + 5, clientY: box.top + 5 }));"
            "  return true;"
            "})()"
        )
        clear_cell = (
            "(() => {"
            "  const item = Array.from(document.querySelectorAll('.ctxmenu__item'))"
            "    .find((n) => n.innerText.startsWith('Clear cell'));"
            "  if (!item) return false;"
            "  item.click();"
            "  return true;"
            "})()"
        )
        page.evaluate(RIGHT_CLICK_CELL)
        page.wait_for("document.querySelector('.ctxmenu') !== null", timeout=10)
        page.drain()
        cell_menu = page.evaluate(
            "(() => { const m = document.querySelector('.ctxmenu'); if (!m) return null;"
            " return Array.from(m.querySelectorAll('.ctxmenu__item')).map((n) => n.innerText); })()"
        )
        check("right-clicking a cell opens the cell menu",
              cell_menu is not None
              and sum(1 for item in cell_menu if item.startswith("Clear cell")) == 1,
              f"{cell_menu}")
        page.evaluate(clear_cell)
        page.drain()
        check("Clear cell becomes a pending edit",
              page.wait_for("document.body.innerText.includes('cell edit')")
              and not shows(page, "no pending changes"),
              page.evaluate(
                  "(() => { const t = document.body.innerText;"
                  "  const i = t.indexOf('cell edit');"
                  "  return i < 0 ? null : t.slice(Math.max(0, i - 12), i + 40); })()"
              ))
        page.evaluate(CLICK_BY_TEXT % json.dumps("Discard changes"))
        page.drain()
        check("Discard changes drops the pending edit",
              shows(page, "no pending changes"),
              page.evaluate(
                  "(() => { const t = document.body.innerText;"
                  "  const i = t.indexOf('pending change');"
                  "  return i < 0 ? null : t.slice(Math.max(0, i - 30), i + 20); })()"
              ))

        # And a save, which is the only control on this page that writes. The edit is made
        # again first: `Discard changes` above cleared the column operations along with it,
        # so this save carries one cell edit and forks a child from it.
        page.evaluate(RIGHT_CLICK_CELL)
        page.drain()
        page.evaluate(clear_cell)
        page.drain()
        page.evaluate("window.prompt = () => 'E2E grid edit'")
        page.evaluate(CLICK_BY_TEXT % json.dumps("Save as new dataset"))
        # Compared without case, for the reason `shows` is: the notice is chrome, and which
        # letter case the stylesheet gives it is not what this check is about.
        check("a grid edit saves as a new dataset",
              page.wait_for(
                  "(document.body.innerText || '').toLowerCase().includes('saved as')",
                  timeout=90,
              ),
              page.text()[-160:].replace("\n", " ")[:140])

        # ------------------------------------------------------ B3 · clean
        print("== B3 · cleaning ==")
        page.goto(f"/cleaning/{klines['dataset_id']}")
        check("the quality report loads",
              page.wait_for("document.querySelector('.policy') !== null", timeout=30))
        groups = page.evaluate(COUNT % json.dumps(".policy"))
        check("every policy group is rendered", groups >= 3, f"{groups} groups")

        RUN_BUTTON = (
            "Array.from(document.querySelectorAll('button'))"
            ".find((b) => b.innerText.includes('Preview'))"
        )
        # "Do nothing" for all three policies is a legal plan — it re-runs the checks and
        # reports an unchanged dataset. Only an *incomplete* plan is refused, so this
        # starts enabled and the refusals below are about a missing choice, not a no-op.
        check("Preview is available on the default plan",
              page.evaluate(f"{RUN_BUTTON}.disabled") is False,
              page.evaluate(f"{RUN_BUTTON}.innerText"))

        page.evaluate(CLICK_BY_TEXT % json.dumps("Winsorization"))
        page.drain()
        check("choosing a fence with no column is refused in the UI",
              shows(page, "no column is selected")
              and page.evaluate(f"{RUN_BUTTON}.disabled") is True,
              page.evaluate(f"{RUN_BUTTON}.disabled"))

        picked = page.evaluate(
            "(() => {"
            "  const box = document.querySelector('#outlier-columns input[type=checkbox]');"
            "  if (!box) return null;"
            "  box.click();"
            "  return box.closest('.columnpick__row').querySelector('.columnpick__name').innerText;"
            "})()"
        )
        page.drain()
        check("picking a column clears the refusal",
              picked is not None and page.evaluate(f"{RUN_BUTTON}.disabled") is False,
              f"column {picked!r}")

        page.evaluate(CLICK_BY_TEXT % json.dumps("Preview"))
        # `.meta__item` belongs to the quality section, which is on the page before any
        # preview runs, so waiting on it would pass without the preview having happened.
        # `.report__key` exists only inside the result block.
        check("preview returns a report",
              page.wait_for("document.querySelectorAll('.report__key').length > 0", timeout=60)
              and {"rows before", "rows after"} <= set(page.evaluate(REPORT_KEYS)),
              page.evaluate(REPORT_KEYS))
        # `.preview` is the head/tail window's own wrapper. A bare `table tbody tr` count
        # would also match the per-column quality table above it, so the check could pass
        # on a page whose preview never rendered at all.
        PREVIEW_ROWS = COUNT % json.dumps(".preview table tbody tr")
        check("preview renders the 5+5 table",
              page.evaluate(PREVIEW_ROWS) >= 10,
              f"{page.evaluate(PREVIEW_ROWS)} rows")
        check("preview stored nothing",
              shows(page, "Preview only"))

        page.evaluate(CLICK_BY_TEXT % json.dumps("Reset plan"))
        page.drain()
        check("resetting the plan clears the outlier selection",
              page.evaluate(COUNT % json.dumps("#outlier-columns input:checked")) == 0
              and page.evaluate(f"{RUN_BUTTON}.disabled") is False,
              page.evaluate(f"{RUN_BUTTON}.disabled"))

        # ----------------------------------------------------- B4 · merge
        print("== B4 · resample & merge ==")
        page.goto(f"/resample/{klines['dataset_id']}")

        RUN_BUTTON = (
            "Array.from(document.querySelectorAll('button'))"
            ".find((b) => b.innerText.includes('Preview'))"
        )
        # Both, in one wait. `.field` is the dataset picker, which the page renders as soon
        # as it mounts; the button every check below indexes is rendered once the dataset
        # list has arrived, so waiting on the field alone returns on a page whose actions do
        # not exist yet and the next line throws `Cannot read properties of undefined`.
        #
        # That race was latent for as long as the driver reused a leftover browser: a warm
        # vite server answers fast enough to hide it. It surfaced the first time the driver
        # ran on a browser it had launched itself, immediately after `serve.py` had
        # restarted both processes — which is the normal case, and the reason this is a fix
        # to the wait rather than a retry around the throw.
        check("the base page loads",
              page.wait_for(
                  f"document.querySelector('.field') !== null && ({RUN_BUTTON}) !== undefined",
                  timeout=30,
              ))
        # The page opens with no timeframe chosen, and says why nothing can run yet. The
        # regression this section covers is the one that used to follow: Preview stayed
        # disabled until a *second dataset* was added, which made the resample half of the
        # step — a supported operation — unreachable.
        check("nothing runs until a timeframe is chosen",
              page.evaluate(f"{RUN_BUTTON}.disabled") is True
              and shows(page, "Choose a timeframe"),
              page.evaluate(f"{RUN_BUTTON}.innerText"))

        check("the timeframe selector is offered",
              page.evaluate("document.querySelector('#timeframe') !== null"))
        other = coarser_than(klines.get("timeframe")) or "4h"
        page.evaluate(SET_SELECT % (json.dumps("#timeframe"), json.dumps(other)))
        page.drain()
        check("resample-only is runnable without adding a dataset",
              page.evaluate(f"{RUN_BUTTON}.disabled") is False,
              page.evaluate(f"{RUN_BUTTON}.innerText"))

        before = page.evaluate(f"{RUN_BUTTON}.innerText")
        page.evaluate(CLICK_BY_TEXT % json.dumps("Preview"))
        check("a resample-only preview returns a report",
              page.wait_for("document.querySelectorAll('.report__key').length > 0", timeout=60),
              before)
        keys = page.evaluate(REPORT_KEYS)
        # Not `keys[:2]`: the report opens with a "Base" row naming the dataset, so the two
        # counts are the second and third keys. Asserting on their order rather than on
        # their position keeps the check about what the report says.
        check("the report states rows in and rows out",
              "rows in" in keys
              and "rows out" in keys
              and keys.index("rows in") < keys.index("rows out"),
              f"{keys}")
        check("the resample report shows the narrowed row count",
              page.evaluate(
                  "(() => {"
                  "  const k = Array.from(document.querySelectorAll('.report__key'))"
                  "    .find((n) => n.innerText.trim().toLowerCase() === 'rows');"
                  "  return k ? k.nextElementSibling.innerText : null;"
                  "})()"
              )
              is not None)
        check("preview stored nothing",
              shows(page, "Preview only"))

        # ------------------------------------------- B4 · editing the columns
        # The spec asks for remove/rename controls on the output columns, so they are
        # exercised rather than assumed: the count in the heading is the observable.
        print("== B4 · column editor ==")
        COLCOUNT = (
            "(() => {"
            "  const h = Array.from(document.querySelectorAll('.section__title'))"
            "    .find((n) => n.innerText.toLowerCase().includes('first 5 and last 5'));"
            "  return h ? parseInt(h.innerText, 10) : null;"
            "})()"
        )
        check("the editor lists every output column",
              page.wait_for("document.querySelectorAll('.dtable tbody tr').length > 0"),
              f"{page.evaluate(COUNT % json.dumps('.dtable tbody tr'))} rows")
        total = page.evaluate(COLCOUNT)
        check("the heading counts the kept columns",
              isinstance(total, int) and total > 0, f"{total}")

        KEEP_BOX = (
            "Array.from(document.querySelectorAll('.dtable tbody tr'))"
            "[0].querySelector('input[type=checkbox]')"
        )
        page.evaluate(f"{KEEP_BOX}.click()")
        page.drain()
        check("unchecking a column removes it from the output",
              page.evaluate(COLCOUNT) == total - 1,
              f"{total} -> {page.evaluate(COLCOUNT)}")

        page.evaluate(f"{KEEP_BOX}.click()")
        page.drain()
        check("re-checking it restores the column",
              page.evaluate(COLCOUNT) == total, f"{page.evaluate(COLCOUNT)}")

        NAME_INPUT = (
            "Array.from(document.querySelectorAll('.dtable tbody tr'))"
            "[0].querySelector('input[type=text]')"
        )
        page.evaluate(SET_INPUT_EXPR % (NAME_INPUT, json.dumps("renamed_probe")))
        page.drain()
        check("renaming a column reaches the output table",
              page.evaluate(
                  "Array.from(document.querySelectorAll('table thead th'))"
                  ".some((th) => th.innerText.toLowerCase().includes('renamed_probe'))"
              ),
              page.evaluate(
                  "Array.from(document.querySelectorAll('table thead th'))"
                  ".map((th) => th.innerText.trim()).join(' | ').slice(0, 140)"
              ))

        page.evaluate(SET_INPUT_EXPR % (NAME_INPUT, json.dumps("")))
        page.drain()
        check("an empty column name is refused before the round trip",
              shows(page, "cannot be empty"),
              page.text()[-160:].replace("\n", " ")[:120])

        page.evaluate(SET_INPUT_EXPR % (NAME_INPUT, json.dumps("renamed_probe")))
        page.drain()

        page.evaluate(SET_SELECT % (json.dumps("#timeframe"), json.dumps("")))
        page.drain()
        check("clearing the timeframe re-refuses the run",
              page.evaluate(f"{RUN_BUTTON}.disabled") is True
              and shows(page, "Choose a timeframe"),
              page.evaluate(f"{RUN_BUTTON}.disabled"))

        # The join half of the step. Whether it can be exercised at all depends on there
        # being a second dataset, so it is skipped with a note rather than failed when
        # the store holds only one — the resample half above is unconditional.
        add_button = (
            "Array.from(document.querySelectorAll('button'))"
            ".find((b) => b.innerText.includes('Add dataset to merge'))"
        )
        if page.evaluate(f"{add_button}.disabled"):
            print("    no second dataset available; the join row is disabled (expected)")
            check("the disabled join control explains itself",
                  shows(page, "No other dataset is available"))
        else:
            page.evaluate(f"({add_button}).click()")
            check("a join row appears",
                  page.wait_for("document.querySelectorAll('.join').length === 1"))

            # Alignment and Match are radio groups, not selects — the dataset picker is the
            # row's only `<select>`. Asserting on the radios is also the stronger check:
            # the method list is the leakage guarantee, and `forward` or `nearest` would
            # stamp a bar with a value that had not been published when the bar closed.
            check("the join row offers an alignment and a match group",
                  page.evaluate("document.querySelectorAll('.join .policy').length") == 2,
                  page.evaluate(
                      "Array.from(document.querySelectorAll('.join .policy__legend'))"
                      ".map((n) => n.innerText).join(' | ')"
                  ))
            # Only the *alignment* group's radios, chosen by its legend. Both groups are
            # `.join .policy`, so the query this check used to run collected the Match
            # group's `left`/`inner` alongside the alignment methods and compared the
            # union against `["backward", "exact"]` — four values against two, a failure
            # against a page that offers exactly the two safe ones.
            methods = page.evaluate(
                "(() => {"
                "  const align = Array.from(document.querySelectorAll('.join .policy'))"
                "    .find((p) => p.querySelector('.policy__legend').innerText.trim()"
                "      .toLowerCase().includes('alignment'));"
                "  return align"
                "    ? Array.from(align.querySelectorAll('input[type=radio]')).map((n) => n.value)"
                "    : null;"
                "})()"
            )
            check("only look-ahead-safe alignment methods are offered",
                  methods is not None and sorted(methods) == ["backward", "exact"],
                  f"{methods}")

            # Choosing a dataset is what makes the row meaningful; the backend refuses a
            # join with no dataset, and the page should say so before the round trip.
            options = page.evaluate(
                "(() => {"
                "  const s = document.querySelector('.join select');"
                "  return s ? Array.from(s.options).map((o) => o.value).filter((v) => v) : [];"
                "})()"
            )
            check("the join row offers the other datasets", len(options) >= 1, f"{options}")
            if options:
                page.evaluate(SET_SELECT % (json.dumps(".join select"), json.dumps(options[0])))
                page.drain()
                check("choosing a dataset leaves Preview enabled",
                      page.evaluate(f"{RUN_BUTTON}.disabled") is False)

            page.evaluate(
                "(() => {"
                "  const b = Array.from(document.querySelectorAll('.join button'))"
                "    .find((n) => n.innerText.trim() === 'remove');"
                "  if (b) b.click();"
                "})()"
            )
            check("a join row can be removed",
                  page.wait_for("document.querySelectorAll('.join').length === 0"))

        # ---------------------------------------------------------- C5 · features
        # Every control on the page is pressed, not merely looked at: the catalogue's
        # grouping, an info icon, a tick, the select-all, a custom window typed into the
        # blank box, both Run buttons, and the clear. A page whose controls are rendered
        # but do nothing is exactly what this section exists to catch.
        print("== C5 · features engineering ==")
        page.goto(f"/features/{klines['dataset_id']}")
        check("the catalogue loads for the chosen dataset",
              page.wait_for("document.querySelector('.feature__row') !== null", timeout=60),
              page.evaluate("location.pathname"))
        check("every group is rendered",
              page.evaluate(COUNT % json.dumps(".feature__group")) == 7,
              f"{page.evaluate(COUNT % json.dumps('.feature__group'))} groups")

        off_rows = page.evaluate(COUNT % json.dumps(".feature__row--off"))
        explained = page.evaluate(
            COUNT % json.dumps(".feature__row--off .feature__reason")
        )
        check("every unavailable feature says why it is unavailable",
              off_rows == explained, f"{off_rows} disabled, {explained} explained")

        check("both Run buttons start disabled",
              page.evaluate(f"{RUN_BUTTON}.disabled") is True
              and shows(page, "Tick at least one feature"),
              page.evaluate(f"{RUN_BUTTON}.innerText"))

        # The info icon is a real button opening a panel — a `title` attribute would not
        # survive this check, which is the point of it.
        page.evaluate(CLICK_SELECTOR % json.dumps(".infotip__button"))
        check("the info icon opens a panel carrying the formula",
              page.wait_for("document.querySelector('.infotip__panel') !== null")
              and page.evaluate(
                  "(document.querySelector('.infotip__formula') || {}).innerText.length > 0"
              ),
              page.evaluate(
                  "(() => { const n = document.querySelector('.infotip__formula');"
                  " return n ? n.innerText.slice(0, 60) : null; })()"
              ))
        page.evaluate(CLICK_SELECTOR % json.dumps(".infotip__button"))
        check("pressing it again closes the panel",
              page.wait_for("document.querySelector('.infotip__panel') === null"))

        # One tick of a parameterised feature must arrive with a window already chosen,
        # or the first Run would be refused for a reason the user did not cause. The row
        # is the first *available* one: a dataset can legitimately offer the first
        # catalogue entry disabled, and a disabled checkbox ignores the click.
        FIRST_ON_ROW = "document.querySelector('.feature__row:not(.feature__row--off)')"
        first_box = json.dumps(
            ".feature__row:not(.feature__row--off) .feature__pick input[type=checkbox]"
        )
        page.evaluate(CLICK_SELECTOR % first_box)
        page.drain()
        check("ticking a feature enables the run",
              page.evaluate(f"{RUN_BUTTON}.disabled") is False,
              page.text()[-160:].replace("\n", " ")[:120])
        check("a ticked parameterised feature already has a window chosen",
              page.evaluate(
                  f"(() => {{"
                  f"  const row = {FIRST_ON_ROW};"
                  "  const boxes = Array.from(row.querySelectorAll('.feature__tick input'));"
                  "  return boxes.length === 0 || boxes.some((b) => b.checked);"
                  "})()"
              ))

        # A window the ticks do not list, typed into the box the specification describes
        # as blank plus "an option of adding more".
        custom_box = json.dumps(".feature__input")
        page.evaluate(SET_INPUT % (custom_box, json.dumps("37")))
        page.evaluate(
            "(() => {"
            "  const b = Array.from(document.querySelectorAll('button'))"
            "    .find((n) => n.innerText.trim() === 'Add another');"
            "  if (!b) return false; b.click(); return true;"
            "})()"
        )
        page.drain()
        check("a window typed into the blank box becomes a removable choice",
              page.wait_for("document.querySelector('.feature__chip') !== null"),
              page.evaluate(
                  "(() => { const n = document.querySelector('.feature__chip');"
                  " return n ? n.innerText.trim() : null; })()"
              ))
        page.evaluate(CLICK_SELECTOR % json.dumps(".feature__chip"))
        page.drain()
        check("removing it takes it back out",
              page.wait_for("document.querySelector('.feature__chip') === null"))

        # Select-all on a whole group, which is the control most likely to be wired to
        # nothing.
        page.evaluate(
            "(() => {"
            "  const group = document.querySelector('.feature__group');"
            "  const b = Array.from(group.querySelectorAll('button'))"
            "    .find((n) => n.innerText.includes('Select all'));"
            "  if (!b) return false; b.click(); return true;"
            "})()"
        )
        page.drain()
        check("select-all ticks every available feature in the group",
              page.evaluate(
                  "(() => {"
                  "  const rows = Array.from(document.querySelectorAll('.feature__group'))[0]"
                  "    .querySelectorAll('.feature__row:not(.feature__row--off)');"
                  "  return Array.from(rows).every((r) =>"
                  "    r.querySelector('.feature__pick input').checked);"
                  "})()"
              ),
              page.evaluate(
                  "(() => {"
                  "  const spacer = document.querySelector('.feature__group .actions__spacer');"
                  "  const next = spacer ? spacer.nextElementSibling : null;"
                  "  return next ? next.innerText : null;"
                  "})()"
              ))

        KEYS_BEFORE = page.evaluate(COUNT % json.dumps(".report__key"))
        page.evaluate(CLICK_BY_TEXT % json.dumps("Preview"))
        check("a feature preview returns a report",
              page.wait_for(
                  f"document.querySelectorAll('.report__key').length > {KEYS_BEFORE}",
                  timeout=90,
              ))
        feature_keys = page.evaluate(REPORT_KEYS)
        check("the report names the price series and the warm-up",
              "price series" in feature_keys and "warm-up" in feature_keys,
              f"{feature_keys}")
        check("the report names the selection hash it can be reproduced from",
              "selection" in feature_keys, f"{feature_keys}")
        check("the preview stored nothing",
              shows(page, "Preview only"),
              page.text()[-200:].replace("\n", " ")[:140])
        check("the output table renders the computed columns",
              page.evaluate(COUNT % json.dumps(".preview tbody tr")) > 0)
        check("the first rows show the warm-up as empty, not as zero",
              page.evaluate(
                  "(() => {"
                  "  const cell = document.querySelector('.preview tbody tr td.is-null');"
                  "  return cell !== null;"
                  "})()"
              ))

        page.evaluate(CLICK_BY_TEXT % json.dumps("Apply and save"))
        check("a feature save writes a dataset and says so",
              page.wait_for("document.body.innerText.includes('Saved as')", timeout=90),
              page.text()[-200:].replace("\n", " ")[:140])
        check("the saved dataset is reachable from the result bar",
              page.evaluate(
                  "Array.from(document.querySelectorAll('a'))"
                  ".some((a) => a.innerText.includes('open in Data Management'))"
              ))

        page.evaluate(CLICK_BY_TEXT % json.dumps("Clear selection"))
        page.drain()
        check("clearing the selection returns the page to its disabled state",
              page.evaluate(f"{RUN_BUTTON}.disabled") is True
              and shows(page, "Tick at least one feature"))

        # ---------------------------------------------------------- C6 · splits
        # The page's whole output is a table of row ranges, so the checks are about the two
        # things a plausible-looking mock would get wrong: that the spans the engine returns
        # are actually drawn — the timeline positions every bar by row index, so a gap the
        # plan left shows as one — and that the controls which depend on the split type
        # appear and disappear with it. Both plans are generated for real, because the plan
        # is what C7 needs two sections below.
        print("== C6 · chronological splits ==")
        SPLIT_RADIOS = COUNT % json.dumps(".policy__option input[type=radio]")
        SPLIT_SPANS = COUNT % json.dumps(".split__table tbody tr")
        SPLIT_LANES = COUNT % json.dumps(".timeline__lane")
        SPLIT_BARS = COUNT % json.dumps(".timeline__bar")
        DATE_INPUTS = COUNT % json.dumps(".split__dates input[type=datetime-local]")
        GENERATE = (
            "Array.from(document.querySelectorAll('button'))"
            ".find((b) => b.innerText.includes('Generate Split Plan'))"
        )
        page.goto(f"/splits/{klines['dataset_id']}")
        check("the split form loads for the chosen dataset",
              page.wait_for("document.querySelector('#split-dataset') !== null", timeout=60)
              and page.wait_for(f"{SPLIT_RADIOS} === 6", timeout=30),
              f"{page.evaluate(SPLIT_RADIOS)} split types")
        check("the default plan needs no dates and states its proportions",
              page.evaluate(COUNT % json.dumps(".split__dates")) == 0
              and shows(page, "70% train"),
              page.evaluate(COUNT % json.dumps(".split__dates")))
        check("the derived controls carry the embargo and the purge",
              page.evaluate("document.querySelector('#split-gap') !== null")
              and page.evaluate("document.querySelector('#split-purge') !== null"))

        page.evaluate(CLICK_BY_TEXT % json.dumps("Generate Split Plan"))
        check("a plan renders a row per span",
              page.wait_for(f"{SPLIT_SPANS} >= 3", timeout=90),
              f"{page.evaluate(SPLIT_SPANS)} spans · {page.text()[-120:].strip()[:100]}")
        check("the timeline draws a bar per span from index zero",
              page.evaluate(SPLIT_BARS) >= 3
              and page.evaluate(SPLIT_LANES) >= 1
              and page.evaluate(
                  "(() => { const b = document.querySelector('.timeline__bar');"
                  " return b !== null && b.style.left === '0%'; })()"
              ),
              f"{page.evaluate(SPLIT_BARS)} bars in {page.evaluate(SPLIT_LANES)} lane(s)")
        check("every span is labelled with its role",
              page.evaluate(COUNT % json.dumps(".split__role--train")) >= 1
              and page.evaluate(COUNT % json.dumps(".split__role--val")) >= 1
              and page.evaluate(COUNT % json.dumps(".split__role--test")) >= 1)
        check("the report names the observation column and the source checksum",
              {"time column", "source checksum"} <= set(page.evaluate(REPORT_KEYS)),
              page.evaluate(REPORT_KEYS))
        check("the plan is stored and the page says where",
              page.wait_for("document.body.innerText.includes('Plan stored as')", timeout=30)
              and shows(page, "existing plan"),
              page.text()[-200:].replace("\n", " ")[:140])

        # Walk-forward. The fold slider replaces the read-only proportions line, and the
        # plan that comes back is one lane per fold rather than three spans in one lane.
        page.evaluate(CLICK_BY_TEXT % json.dumps("Walk-forward — rolling window"))
        page.drain()
        check("a walk-forward type swaps the proportions for a fold slider",
              page.wait_for("document.querySelector('#split-folds') !== null")
              and shows(page, "the first block is never tested"),
              page.evaluate(
                  "(() => { const f = document.querySelector('#split-folds');"
                  " return f ? f.value : null; })()"
              ))
        # Kept, not discarded: the plan generated two lines below is the one C7's fold
        # selector is built from, so this is the number that selector has to agree with.
        folds_made = int(
            page.evaluate(
                "(() => { const f = document.querySelector('#split-folds');"
                " return f ? f.value : 0; })()"
            )
        )
        page.evaluate(CLICK_BY_TEXT % json.dumps("Generate Split Plan"))
        check("a walk-forward plan draws one lane per fold",
              page.wait_for(f"{SPLIT_LANES} > 1", timeout=90),
              f"{page.evaluate(SPLIT_LANES)} lanes · {page.text()[-120:].strip()[:100]}")

        # The client-side refusal. Six boundaries are required and none has been typed, so
        # the run is blocked with every missing field named — rather than sent and refused
        # by the server, which would report the first one only.
        page.evaluate(CLICK_BY_TEXT % json.dumps("Three spans by date"))
        page.drain()
        check("a date split asks for all six boundaries at once",
              page.wait_for(f"{DATE_INPUTS} === 6"),
              f"{page.evaluate(DATE_INPUTS)} inputs")
        check("the incomplete plan is refused in the UI, naming each missing boundary",
              page.evaluate(f"{GENERATE}.disabled") is True
              and shows(page, "the plan cannot be generated yet")
              and shows(page, "Train from needs a date and time")
              and shows(page, "Test to needs a date and time"),
              page.evaluate(f"{GENERATE}.disabled"))

        # --------------------------------------------------------- C7 · scaling
        # The page's rule is "fit on the training span, transform everything", and the
        # checks below are arranged around the one control that enforces it: a fitted
        # scaler cannot run without a plan, and a walk-forward plan cannot run without a
        # fold. Both are checked by choosing in the dropdown and watching the form change,
        # then by running for real — the result block, with its parameters table and its
        # head/tail preview, exists only after a job has come back through the poll loop.
        print("== C7 · scaling ==")
        SCALER_OFF = COUNT % json.dumps(".policy__option--off")
        PICKED_COLUMNS = COUNT % json.dumps(".columnpick__row")
        RUN_SCALING = (
            "Array.from(document.querySelectorAll('button'))"
            ".find((b) => b.innerText.includes('Run Scaling'))"
        )
        page.goto(f"/scaling/{klines['dataset_id']}")
        check("every scaler is available now that C6 has made a plan",
              page.wait_for("document.querySelector('#scaling-plan') !== null", timeout=60)
              and page.wait_for(f"{SCALER_OFF} === 0", timeout=30),
              f"{page.evaluate(SCALER_OFF)} still unavailable")
        check("the plan just made is the one the dropdown offers",
              page.evaluate(COUNT % json.dumps("#scaling-plan option")) >= 2,
              (
                  page.evaluate(
                      "(() => { const s = document.querySelector('#scaling-plan');"
                      " return s ? Array.from(s.options).map((o) => o.innerText).join(' | ')"
                      " : null; })()"
                  )
                  or ""
              )[:150])
        check("the features default to every numeric column, all ticked",
              page.evaluate(PICKED_COLUMNS) >= 1
              and page.evaluate(COUNT % json.dumps(".columnpick__row input:checked"))
              == page.evaluate(PICKED_COLUMNS),
              f"{page.evaluate(COUNT % json.dumps('.columnpick__row input:checked'))}"
              f" of {page.evaluate(PICKED_COLUMNS)} ticked")

        # The fold control appears exactly when the chosen plan is a walk-forward one. Both
        # kinds are chosen in turn rather than reading whichever happens to be preselected.
        FOLD_OPTIONS = COUNT % json.dumps("#scaling-fold option")
        fixed_plan = choose_scaling_plan(page, "/percentage/")
        page.drain()
        check("a percentage plan asks for no fold",
              fixed_plan is not None
              and page.evaluate("document.querySelector('#scaling-fold') === null"),
              f"{fixed_plan!r}")
        walk_plan = choose_scaling_plan(page, "/tscv/")
        page.drain()
        check("a walk-forward plan asks which fold to fit on",
              walk_plan is not None
              and page.wait_for("document.querySelector('#scaling-fold') !== null"),
              f"{walk_plan!r}")
        # Both numbers, and both against the plan C6 just generated: the select must offer
        # one option per fold the plan holds, and the label the user reads must say the same
        # number. Asserting against `folds_made` rather than a literal is what makes this
        # catch the failure it was written for — the plan's recorded fold count once ran one
        # past the folds it contained, and a hard-coded 5 would have passed while the page
        # advertised a fold that did not exist.
        check("the fold control offers exactly the folds the plan holds",
              page.evaluate(FOLD_OPTIONS) == folds_made
              and f"{folds_made} fold(s)" in (walk_plan or ""),
              f"{page.evaluate(FOLD_OPTIONS)} options, C6 made {folds_made}, "
              f"label {walk_plan!r}")

        page.evaluate(CLICK_BY_TEXT % json.dumps("Run Scaling"))
        check("the run polls to a result and reports what it fitted on",
              page.wait_for(
                  "document.querySelectorAll('.report__key').length > 0", timeout=180
              )
              and "fitted on" in page.evaluate(REPORT_KEYS),
              f"{page.evaluate(REPORT_KEYS)}")
        # Read from the report values rather than the whole page: the fold <select> in the
        # form already contains an option reading "fold 1", so a page-wide text search would
        # pass whether or not the report said anything about the fold it used.
        REPORT_VALUES = (
            "Array.from(document.querySelectorAll('.report__value'))"
            ".map((n) => n.innerText.trim()).join(' | ')"
        )
        fitted_on = page.evaluate(REPORT_VALUES) or ""
        check("the report names the fold it was fitted for",
              "fold 1" in fitted_on,
              fitted_on[:140])
        check("the parameters table names the inversion pair it stored",
              shows(page, "the inversion pair") and shows(page, "fitted rows"))
        check("the result preview shows the rescaled frame",
              page.evaluate(COUNT % json.dumps(".preview table tbody tr")) >= 10,
              f"{page.evaluate(COUNT % json.dumps('.preview table tbody tr'))} rows")
        check("both artifacts are named and reachable",
              shows(page, "inspect the parameters")
              and page.evaluate(
                  "Array.from(document.querySelectorAll('a'))"
                  ".some((a) => a.innerText.includes('open in Data Management'))"
              ))

        # The refusal the page is built around, on a dataset that has no plan. The reason
        # has to be on the radio itself: a control that is merely greyed out teaches the
        # user nothing about why, and this one is the difference between a backtest that is
        # honest and one that is not.
        unplanned = dataset_without_a_plan(api("/datasets"), klines["dataset_id"])
        if unplanned is None:
            check("a dataset with no plan exists to check the refusal against", False,
                  "every eligible dataset in the store already has a plan")
        else:
            page.goto(f"/scaling/{unplanned['dataset_id']}")
            # `>= 3` rather than `== 3`: on a dataset with no numeric column the rolling
            # z-score is unavailable too, and that is a correct page. What this check is
            # about is that the three *fitted* scalers are off and that the reason beside
            # them names C6 — which is true whenever there is no plan, whatever else the
            # dataset is.
            check("without a plan the three fitted scalers are off, and the reason names C6",
                  page.wait_for(f"{SCALER_OFF} >= 3", timeout=60)
                  and page.evaluate(
                      COUNT % json.dumps(".policy__option--off input[value='rolling_zscore']")
                  )
                  == 0
                  and shows(page, "c6 chronological splits"),
                  f"{page.evaluate(SCALER_OFF)} unavailable on {unplanned['name']!r}")
            check("the plan dropdown says no plan exists rather than offering an empty one",
                  page.evaluate(
                      "(() => { const s = document.querySelector('#scaling-plan');"
                      " return s !== null && s.disabled && s.options.length === 1; })()"
                  ))
            check("the run is refused in the UI before anything is sent",
                  page.evaluate(f"{RUN_SCALING}.disabled") is True
                  and shows(page, "the run cannot start yet"))

        # -------------------------------------------------------------- D1 · regime
        # The regime page is the one screen where a run is a *job* rather than a request, so
        # this section drives the whole handshake — press Run, watch the button become a job
        # handle, wait for the dashboard behind it — instead of asserting on markup that was
        # rendered without anything having run. The gauge, the two figures and the whole
        # diagnostics block exist only after a fit has come back through the poll loop, so
        # every check below is evidence that path works end to end.
        print("== D1 · regime & trend ==")
        page.goto("/regime")
        check("with nothing chosen the page says what it needs",
              page.wait_for("document.querySelector('#regime-dataset') !== null")
              and shows(page, "No dataset selected"),
              page.text()[-140:].replace("\n", " ")[:120])

        # `undefined` while a run is in flight, because the button's text is "Running…" then.
        # Every use below is on a page that is not running.
        RUN_DETECTION = (
            "Array.from(document.querySelectorAll('button'))"
            ".find((b) => b.innerText.includes('Run Detection'))"
        )
        subject = regime if regime is not None else klines
        # The picker's options arrive from their own request; assigning a value the `<select>`
        # does not yet offer is silently ignored, so the navigation below would never happen
        # and the check would time out against a working page.
        page.wait_for(
            "Array.from(document.querySelectorAll('#regime-dataset option'))"
            f".some((o) => o.value === {json.dumps(subject['dataset_id'])})",
            timeout=30,
        )
        page.evaluate(
            SET_SELECT % (json.dumps("#regime-dataset"), json.dumps(subject["dataset_id"]))
        )
        check("choosing a dataset from the picker loads its options",
              page.wait_for(f"location.pathname === '/regime/{subject['dataset_id']}'", timeout=10)
              and page.wait_for("document.querySelector('#regime-duration') !== null", timeout=30),
              page.evaluate("location.pathname"))

        check("the source panel states the annualisation the crisis threshold rides on",
              page.evaluate("document.querySelector('.regime__annualisation') !== null")
              and shows(page, "Annualisation"),
              page.evaluate(
                  "(() => { const n = document.querySelector('.regime__annualisation');"
                  " return n ? n.innerText.replace(/\\n/g, ' ').slice(0, 90) : null; })()"
              ))

        methods = page.evaluate(COUNT % json.dumps(".policy__option"))
        unavailable = page.evaluate(COUNT % json.dumps(".policy__option--off"))
        explained = page.evaluate(COUNT % json.dumps(".policy__option--off .policy__caution"))
        check("all four detection methods are listed, and an unavailable one says why",
              methods == 4 and unavailable == explained,
              f"{methods} methods, {unavailable} unavailable, {explained} explained")

        page.evaluate(
            "(() => {"
            "  const r = document.querySelector('input[name=regime-model][value=rules]');"
            "  if (!r || r.disabled || r.checked) return false;"
            "  r.click(); return true;"
            "})()"
        )
        page.drain()
        check("picking a method selects it and only it",
              page.evaluate("document.querySelector('input[name=regime-model][value=rules]')"
                            ".checked") is True
              and page.evaluate(
                  "Array.from(document.querySelectorAll('input[name=regime-model]'))"
                  ".filter((r) => r.checked).length") == 1,
              page.evaluate(
                  "Array.from(document.querySelectorAll('input[name=regime-model]'))"
                  ".map((r) => r.value + '=' + r.checked).join(' ')"))

        check("the observation set is seeded with the causal columns the dataset has",
              page.evaluate(COUNT % json.dumps(".feature__pick input:checked")) >= 1,
              page.evaluate(
                  "Array.from(document.querySelectorAll('.feature__pick'))"
                  ".map((n) => n.innerText.split('\\n')[0] + '=' +"
                  " n.querySelector('input').checked).join(' | ')"))

        page.evaluate(SET_INPUT % (json.dumps("#regime-duration"), json.dumps("10")))
        page.drain()
        # Compared without case, for the reason `shows` is: the stylesheet upper-cases every
        # `label`, so the rendered text is "MINIMUM STATE DURATION 10 BARS". The check is
        # about the slider having moved the number, not about the font's letter case.
        DURATION_LABEL = (
            "(() => { const n = document.querySelector('label[for=regime-duration]');"
            " return n ? n.innerText.trim() : null; })()"
        )
        check("moving the anti-thrashing slider moves its label with it",
              "10 bars" in (page.evaluate(DURATION_LABEL) or "").lower(),
              page.evaluate(DURATION_LABEL))

        page.evaluate(SET_INPUT % (json.dumps("#regime-cost"), json.dumps("-1")))
        page.drain()
        check("a negative switching cost is refused before the round trip",
              shows(page, "Switching cost penalty must be a number at or above zero")
              and page.evaluate(f"{RUN_DETECTION}.disabled") is True,
              page.evaluate(f"{RUN_DETECTION}.disabled"))

        page.evaluate(SET_INPUT % (json.dumps("#regime-cost"), json.dumps("0.005")))
        page.drain()

        if regime is None:
            # Nothing in the store can produce a label, so there is no dashboard to reach.
            # The store's contents are not the application's defect and are not asserted;
            # what is checked is that the page still renders its configuration rather than
            # failing on a dataset it cannot fit.
            print("    no regime-capable dataset; the D1 run is skipped")
            check("the configuration panel renders for a dataset that cannot be fitted",
                  page.evaluate("document.querySelector('#regime-duration') !== null")
                  and page.evaluate(COUNT % json.dumps(".policy__option")) == 4)
        else:
            check("restoring the cost re-arms the run",
                  page.evaluate(f"{RUN_DETECTION}.disabled") is False,
                  page.text()[-180:].replace("\n", " ")[:140])

            page.evaluate(
                "(() => {"
                "  const r = document.querySelector('input[name=regime-model][value=hmm]');"
                "  if (!r || r.disabled || r.checked) return false;"
                "  r.click(); return true;"
                "})()"
            )
            page.drain()

            # The button's running label is transient by design — the fit here finishes in
            # well under a second — so it cannot be sampled reliably from a second round
            # trip, and reading it in the *same* evaluation as the click does not work
            # either: this was tried, and React had not yet committed the update when the
            # expression returned ("Run Detection" with Stop waiting still disabled). A
            # MutationObserver installed before the click records the label every time React
            # changes it, which turns "did this ever show a job handle" from a race into a
            # question with a definite answer.
            started = page.evaluate(
                "(() => {"
                f"  const b = {RUN_DETECTION};"
                "  if (!b || b.disabled) return null;"
                "  const seen = [];"
                "  window.__regimeRunLabels = seen;"
                "  new MutationObserver(() => seen.push(b.innerText))"
                "    .observe(b, { childList: true, subtree: true, characterData: true });"
                "  b.click();"
                "  return true;"
                "})()"
            )

            # Waits on the failure notice as well as the gauge: a fit that is refused must
            # end this check with the reason on screen rather than after five minutes of
            # polling a page that will never change.
            arrived = page.wait_for(
                "document.querySelector('.regime__gauge') !== null"
                " || document.querySelector('.result__bar--error') !== null",
                timeout=300,
            )
            check("the run returns a labelled frame rather than an error",
                  arrived
                  and page.evaluate("document.querySelector('.regime__gauge') !== null"),
                  page.text()[-220:].replace("\n", " ")[:180])

            labels = page.evaluate("window.__regimeRunLabels || []") or []
            check("the button becomes a polling job handle while the fit runs",
                  bool(started)
                  and any(str(label).startswith("Running") for label in labels),
                  f"{labels}")
            check("the page comes back to Run Detection once the job lands",
                  bool(labels) and str(labels[-1]).strip() == "Run Detection",
                  f"{labels[-3:]}")

            state = page.evaluate(
                "(() => { const n = document.querySelector('.regime__gauge-state');"
                " return n ? n.innerText.trim() : null; })()"
            )
            percent = page.evaluate(
                "(() => { const n = document.querySelector('.regime__gauge-value');"
                " return n ? n.innerText.trim() : null; })()"
            )
            check("the gauge names one of the five states and sizes its probability",
                  state in {"Trending Up", "Trending Down", "Ranging", "Crisis", "Transition"}
                  and isinstance(percent, str) and percent.endswith("%"),
                  f"{state} at {percent}")

            check("the timeline and the transition matrix are both offered",
                  shows(page, "Regime timeline") and shows(page, "Transition matrix"))
            check("both figures actually drew",
                  page.evaluate(COUNT % json.dumps(".js-plotly-plot .main-svg")) >= 2,
                  page.evaluate(COUNT % json.dumps(".js-plotly-plot .main-svg")))

            table_keys = page.evaluate(
                "Array.from(document.querySelectorAll('.regime__table-key'))"
                ".map((n) => n.innerText.trim().toLowerCase())"
            )
            check("the diagnostics table carries the fit statistics",
                  {"bic (latest refit)", "switch count", "average state duration"}
                  <= set(table_keys),
                  f"{table_keys}")
            report_keys = page.evaluate(REPORT_KEYS)
            check("the report explains the score it prints",
                  "net value-add" in report_keys
                  and {"refits", "crisis override"} <= set(report_keys),
                  f"{report_keys}")
            # The two naming defects this driver was extended to catch, asserted as the
            # words on the page: the score's cost is charged per *position* change while the
            # report's switch count counts state changes, and the null prefix is the warm-up
            # *plus* the training window rather than the warm-up alone.
            check("the switching cost is charged per position change, not per state change",
                  shows(page, "position change"))
            check("the unlabelled prefix is named as the warm-up plus the training window",
                  shows(page, "Unlabelled (warm-up + training)"))
            check("the score is stated to be in-sample",
                  shows(page, "In-sample"))
            check("the output preview renders the labelled frame",
                  page.evaluate(COUNT % json.dumps(".preview tbody tr")) >= 1,
                  f"{page.evaluate(COUNT % json.dumps('.preview tbody tr'))} preview rows")
            check("the run saved a child dataset and offers to open it",
                  page.evaluate(
                      "Array.from(document.querySelectorAll('button'))"
                      ".some((b) => b.innerText.trim().startsWith('Open '))"),
                  page.evaluate(
                      "Array.from(document.querySelectorAll('button'))"
                      ".map((b) => b.innerText.trim()).find((t) => t.startsWith('Open '))"))

        # ------------------------------------------------- E1 · volatility forecast
        # The first screen in this application whose subject is a bar that has not happened
        # yet, and the first whose run is *scored* rather than merely rendered. The section
        # drives the whole handshake — press Run, watch the button become a job handle, wait
        # for the report behind it — because the fan chart, the metrics table and the
        # comparison table exist only after a walk-forward has come back through the poll
        # loop. Asserting on markup rendered without anything having run would check the
        # empty state twice.
        print("== E1 · volatility forecast ==")
        page.goto("/volatility")
        check("with nothing chosen the page says what it needs",
              page.wait_for("document.querySelector('#vol-dataset') !== null")
              and shows(page, "No dataset selected"),
              page.text()[-150:].replace("\n", " ")[:130])

        RUN_FORECAST = (
            "Array.from(document.querySelectorAll('button'))"
            ".find((b) => b.innerText.includes('Run Forecast'))"
        )
        subject = volatility_subject(klines)
        run_subject = subject if subject is not None else klines
        # The picker's options arrive from their own request, so assigning a value the
        # `<select>` does not yet offer is silently ignored and the navigation below would
        # never happen — the same race the D1 section waits out, for the same reason.
        page.wait_for(
            "Array.from(document.querySelectorAll('#vol-dataset option'))"
            f".some((o) => o.value === {json.dumps(run_subject['dataset_id'])})",
            timeout=30,
        )
        page.evaluate(
            SET_SELECT % (json.dumps("#vol-dataset"), json.dumps(run_subject["dataset_id"]))
        )
        check("choosing a dataset from the picker loads its options",
              page.wait_for(
                  f"location.pathname === '/volatility/{run_subject['dataset_id']}'", timeout=10
              )
              and page.wait_for("document.querySelector('#vol-train') !== null", timeout=30),
              page.evaluate("location.pathname"))

        families = page.evaluate(
            "Array.from(document.querySelectorAll('.vol__family')).map((n) => n.innerText.trim())"
        )
        # Compared without regard to case, for the reason `shows` gives: the stylesheet
        # upper-cases these group headings, and which letter case it chose is not what this
        # check is about. The heading is the server's — `VOL_MODEL_FAMILY` decides which model
        # sits under which — so what is asserted is that there are two groups, in order, with
        # the arch estimators first and the hybrids second.
        lowered = [entry.lower() for entry in families]
        check("the two model families are grouped and named",
              len(families) == 2 and "family a" in lowered[0] and "family b" in lowered[1],
              f"{families}")

        offered = page.evaluate(
            "Array.from(document.querySelectorAll('.policy__option input[type=checkbox]'))"
            ".map((n) => n.value)"
        )
        check("every model is offered, and a hybrid states what it costs",
              set(offered)
              == {"garch", "egarch", "gjr_garch", "arma_egarch_xgboost", "tdnn_gjr_garch"}
              and shows(page, "Refits a second model at every scored step"),
              f"{offered}")

        check("all four shock distributions and both mean equations are offered",
              page.evaluate(COUNT % json.dumps("input[name=vol-distribution]")) == 4
              and page.evaluate(COUNT % json.dumps("input[name=vol-mean]")) == 2,
              f"{page.evaluate(COUNT % json.dumps('input[name=vol-distribution]'))} "
              f"distributions, {page.evaluate(COUNT % json.dumps('input[name=vol-mean]'))} means")

        # The controls the server's `needs` table drives. What makes this worth asserting is
        # that the same field goes live and dark again as models are ticked and unticked: a
        # page that merely rendered `o` disabled would pass a check written on the initial
        # state alone, and that page would leave a control dead for a model that reads it.
        check("only the parameters a selected model consumes are live",
              page.evaluate("document.querySelector('#vol-o').disabled") is True
              and page.evaluate("document.querySelector('#vol-lags').disabled") is True,
              page.evaluate(
                  "(() => { const o = document.querySelector('#vol-o');"
                  " const l = document.querySelector('#vol-lags');"
                  " return `o=${o.disabled} lags=${l.disabled}`; })()"))

        page.evaluate(CLICK_SELECTOR % json.dumps("input[type=checkbox][value=gjr_garch]"))
        page.drain()
        check("ticking an asymmetric model arms the asymmetry order and nothing else",
              page.evaluate("document.querySelector('#vol-o').disabled") is False
              and page.evaluate("document.querySelector('#vol-lags').disabled") is True)

        page.evaluate(
            CLICK_SELECTOR % json.dumps("input[type=checkbox][value=arma_egarch_xgboost]")
        )
        page.drain()
        check("ticking a hybrid arms the feature lags it reads",
              page.evaluate("document.querySelector('#vol-lags').disabled") is False)
        # Unticked again. The hybrid refits a second model inside the walk-forward loop and
        # is the slowest option on this page by a wide margin; it is exercised here as a
        # *control* over the form, not as a run.
        page.evaluate(
            CLICK_SELECTOR % json.dumps("input[type=checkbox][value=arma_egarch_xgboost]")
        )
        page.drain()
        check("unticking it disarms the lags again, leaving the chosen pair selected",
              page.evaluate("document.querySelector('#vol-lags').disabled") is True
              and page.evaluate(COUNT % json.dumps("input[type=checkbox]:checked")) == 2,
              page.evaluate(
                  "Array.from(document.querySelectorAll('.policy__option input[type=checkbox]'))"
                  ".map((n) => n.value + '=' + n.checked).join(' ')"))

        if subject is None:
            # Not a failure: a store holding nothing longer than a few hundred bars is a
            # thinner store, not a broken page. What is checked is that the configuration
            # still renders against a dataset the section cannot drive. Said plainly, because
            # this branch means the run path below was *not* exercised — and while it prints
            # like a note, it silently narrows what the section covers.
            print(
                "    the E1 run is SKIPPED: no dataset in the catalog has both a price column "
                f"and {VOL_TRAIN + VOL_TEST} usable returns, so the fan chart, the metrics "
                "table and the report card were not driven"
            )
            check("the configuration still renders for a dataset with too few returns",
                  page.evaluate("document.querySelector('#vol-train') !== null")
                  and page.evaluate(COUNT % json.dumps(".vol__family")) == 2)
        else:
            usable = subject["usable_returns"]
            for selector, value in (
                ("#vol-train", VOL_TRAIN),
                ("#vol-test", VOL_TEST),
                ("#vol-step", VOL_STEP),
                ("#vol-horizon", VOL_HORIZON),
            ):
                page.evaluate(SET_INPUT % (json.dumps(selector), json.dumps(str(value))))
            page.drain()
            check("a schedule the series can carry leaves the run armed",
                  page.evaluate(f"{RUN_FORECAST}.disabled") is False
                  and not shows(page, "the run cannot start yet"),
                  page.text()[-180:].replace("\n", " ")[:140])

            page.evaluate(SET_INPUT % (json.dumps("#vol-train"), json.dumps(str(usable + 1000))))
            page.drain()
            check("an impossible window is refused before the round trip, with both numbers",
                  shows(page, "the run cannot start yet")
                  and shows(page, f"{usable} usable returns")
                  and page.evaluate(f"{RUN_FORECAST}.disabled") is True,
                  page.text()[-220:].replace("\n", " ")[:180])

            page.evaluate(SET_INPUT % (json.dumps("#vol-train"), json.dumps(str(VOL_TRAIN))))
            page.drain()
            check("restoring the window re-arms the run",
                  page.evaluate(f"{RUN_FORECAST}.disabled") is False,
                  page.text()[-200:].replace("\n", " ")[:160])

            # The running label is transient — this configuration finishes in seconds — so
            # it cannot be sampled from a second round trip. A MutationObserver installed
            # before the click records every label React commits, which turns "did this ever
            # show a job handle" from a race into a question with a definite answer. Same
            # technique and same reason as the D1 run above.
            started = page.evaluate(
                "(() => {"
                f"  const b = {RUN_FORECAST};"
                "  if (!b || b.disabled) return null;"
                "  const seen = [];"
                "  window.__volRunLabels = seen;"
                "  new MutationObserver(() => seen.push(b.innerText))"
                "    .observe(b, { childList: true, subtree: true, characterData: true });"
                "  b.click();"
                "  return true;"
                "})()"
            )
            # Waits on the failure notice as well as the chart: a refused run has to end
            # this check with the reason on screen rather than after minutes of polling a
            # page that will never change.
            arrived = page.wait_for(
                "document.querySelector('.vol__path-picker') !== null"
                " || document.querySelector('.result__bar--error') !== null",
                timeout=300,
            )
            check("the walk-forward returns a report rather than an error",
                  arrived
                  and page.evaluate("document.querySelector('.vol__path-picker') !== null"),
                  page.text()[-260:].replace("\n", " ")[:200])

            labels = page.evaluate("window.__volRunLabels || []") or []
            check("the button becomes a polling job handle while the walk-forward runs",
                  bool(started)
                  and any(str(label).startswith("Running") for label in labels),
                  f"{labels}")
            check("the page comes back to Run Forecast once the job lands",
                  bool(labels) and str(labels[-1]).strip() == "Run Forecast",
                  f"{labels[-3:]}")

            check("the fan chart drew",
                  page.evaluate(COUNT % json.dumps(".js-plotly-plot .main-svg")) >= 1,
                  page.evaluate(COUNT % json.dumps(".js-plotly-plot .main-svg")))
            paths = page.evaluate(
                "Array.from(document.querySelectorAll('.vol__path-picker option'))"
                ".map((o) => o.value)"
            )
            check("every model that was scored has a projection to draw",
                  set(paths) == set(VOL_MODELS), f"{paths}")

            check("the metrics table and the pairwise comparison both render",
                  page.evaluate(COUNT % json.dumps(".vol__table")) == 2
                  and page.evaluate(COUNT % json.dumps(".vol__table tbody tr")) == 3,
                  f"{page.evaluate(COUNT % json.dumps('.vol__table'))} tables, "
                  f"{page.evaluate(COUNT % json.dumps('.vol__table tbody tr'))} rows for "
                  f"{len(VOL_MODELS)} models")
            check("the comparison is read in words rather than left as a bare p-value",
                  shows(page, "no significant difference at 5%")
                  or shows(page, "has the lower mean loss"))

            # The two honesty statements the page exists to make: what the scores were
            # measured against, and that the projection on the chart is not the frame that
            # was saved.
            check("the Mincer–Zarnowitz caveat is stated rather than left to the reader",
                  shows(page, "squared log return")
                  and shows(page, "not evidence of a broken one"))
            check("the page states that the stored frame is the scored span, not the fan",
                  shows(page, "scored walk-forward span only"))

            report_keys = page.evaluate(REPORT_KEYS)
            check("the report states the series, the scored region and the projection",
                  {"return series", "rows", "scored region", "stationarity (adf)",
                   "fitting scale", "forward path"} <= set(report_keys),
                  f"{report_keys}")
            check("the output preview renders the scored frame",
                  page.evaluate(COUNT % json.dumps(".preview tbody tr")) >= 1,
                  f"{page.evaluate(COUNT % json.dumps('.preview tbody tr'))} preview rows")
            check("the run saved a child dataset and offers to open it",
                  page.evaluate(
                      "Array.from(document.querySelectorAll('button'))"
                      ".some((b) => b.innerText.trim().startsWith('Open '))"),
                  page.evaluate(
                      "Array.from(document.querySelectorAll('button'))"
                      ".map((b) => b.innerText.trim()).find((t) => t.startsWith('Open '))"))

        print()
        if PAGE_EXCEPTIONS:
            print(f"{len(PAGE_EXCEPTIONS)} uncaught page exception(s):")
            for item in dict.fromkeys(PAGE_EXCEPTIONS):
                print("  -", item[:200])
            FAILURES.append(f"{len(PAGE_EXCEPTIONS)} uncaught page exception(s)")
        if CONSOLE_ERRORS:
            # Reported, not failed: most of these are subresource loads the page is not
            # responsible for. They are printed so a real one is visible in the log.
            print(f"{len(CONSOLE_ERRORS)} console error(s):")
            for item in dict.fromkeys(CONSOLE_ERRORS):
                print("  -", item[:200])
        if REFUSALS:
            print(f"{len(REFUSALS)} refused request(s):")
            for item in dict.fromkeys(REFUSALS):
                print("  -", item[:200])
        if FAILURES:
            print(f"{len(FAILURES)} FAILURE(S):")
            for item in FAILURES:
                print("  -", item)
            return 1
        print("all UI checks passed")
        return 0
    finally:
        if ui_disposable_id is not None:
            with suppress(Exception):
                api_mutation("DELETE", f"/datasets/{ui_disposable_id}")
        page_ref = locals().get("page")
        if page_ref is not None:
            page_ref.close()
        # Through the protocol first, and off the process only as a fallback — the reverse
        # of the obvious order, for the reason `close_stale_browser` documents: the process
        # this driver started is the launcher, and killing it leaves the browser running.
        endpoint = browser_endpoint()
        if endpoint is not None:
            close_browser(endpoint)
        try:
            browser.wait(timeout=10)
        except subprocess.TimeoutExpired:
            browser.terminate()
            try:
                browser.wait(timeout=10)
            except subprocess.TimeoutExpired:
                browser.kill()
        # After the browser is gone, never before: Windows holds handles on a profile that
        # is still in use, and a tree deleted out from under a live browser is how a driver
        # leaves a half-removed directory that its next run cannot re-create. Nothing is
        # asserted about the outcome — a profile Windows still has open is a dead temporary
        # directory, not a failed check.
        #
        # Retried, because one attempt is not enough: `wait` returns when the *launcher*
        # exits, and Edge's renderer and utility children let go of their handles a moment
        # later. A single `rmtree` there left 129 files behind and reported success, which
        # is exactly the failure mode `ignore_errors` hides.
        for _ in range(10):
            if _PROFILE is None or not _PROFILE.exists():
                break
            shutil.rmtree(_PROFILE, ignore_errors=True)
            if _PROFILE.exists():
                time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main())
