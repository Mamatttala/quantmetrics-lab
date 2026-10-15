"""End-to-end exercise of the Module E volatility endpoints over HTTP.

Not a unit test: it drives the running server exactly as the browser does, so it catches
contract drift the in-process TestClient cannot see — the 202-then-poll handshake, the
`error` envelope arriving through a job record rather than through a request, and the
provenance a saved forecast actually leaves on the dataset a user keeps.

Five things make this driver worth its runtime beyond re-checking the unit tests:

**A forecast is the one artifact that cannot be checked against its input.** Everything the
modules before E produced could be verified by re-reading the frame it came from. A
forecast can only be checked against *another* run, or against the protocol that produced
it. So the driver checks the protocol: the scored span has to be the last `test_window`
bars of the *return* series, which is one bar shorter than the frame when returns are
derived from close — the off-by-one the whole schedule rests on. If the engine scored the
frame instead of the returns, the output's timestamps would land one bar late and nothing
in the payload would say so.

**The handshake is a protocol, not a function call.** ``POST /api/volatility/forecast`` has
to return *before* the work is done. An in-process test can call the service and await it,
which would pass even if the endpoint had been written to block — the very thing the design
exists to prevent. Only a real HTTP client can observe that a 202 arrives with no result in
it.

**Determinism is asserted on the forward path, because that is where the randomness is.**
A one-step GARCH forecast is closed-form, so it is deterministic whether or not anyone
seeded anything. A multi-step EGARCH path is *simulated*: arch draws shocks from the fitted
distribution, and by default it draws fresh ones on every call, so two identical requests
would return two different fan charts. Nothing in a unit test that runs within one process
and one seeded RNG catches that as well as two live runs compared to each other.

**The dataset and the report are two different objects, and the difference is the point.**
The scored walk-forward span goes to the store; the projection beyond the last observation
does not. A projection of the next `h` bars is not an observation, and a store that holds
it holds rows no cleaning step, no split plan and no backtest is entitled to treat as data.
The driver pages the saved frame and asserts every timestamp it holds is at or before the
parent's last observation — which is what proves the projection stayed out.

**Conditioning is exercised against a real Module D output**, not a synthetic probability
frame. That is also the end-to-end proof of the timestamp repair the module depends on: the
conditioning dropdown offers a regime dataset only when it carries an observation-time
column, so a regime output still stamped `close_time` would be missing from the list rather
than silently aligned one bar late. Running one here is what closes that loop.

Run it against a live server (``python tools/serve.py`` in another shell):

    python tools/e2e_volatility.py
"""

from __future__ import annotations

import json
import math
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Any

# A driver run by hand inherits its console's encoding, which on this machine is cp1256 and
# cannot encode the arrow it prints. ``verify_all.py`` sets PYTHONIOENCODING for the children
# it starts; this covers running it directly, as the header above says to.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8000/api"
FAILURES: list[str] = []

#: The exact output schema Module E specifies. Asserted as an ordered tuple: a consumer
#: reading these by position is a thing that will eventually happen.
#:
#: Repeated here rather than imported. The check is precisely that the server's schema is
#: what the specification says it is; importing the server's own constant would make the
#: assertion agree with whatever the server happened to do.
#:
#: The time column is `open_time`, the instant the bar was *observed* — the earliest instant
#: at which a forecast for it could have been made. There is no `close_time`, and there is no
#: `is_forward_path`: the forward path is not in the dataset at all, so there is no row in
#: here that could be mistaken for an observation of a bar that has not happened.
VOL_FORECAST_COLUMNS = (
    "open_time",
    "model",
    "forecast_variance",
    "forecast_volatility",
    "actual_variance",
    "actual_volatility",
    "error_variance",
    "ratio",
    "finite",
)

MODELS = ("garch", "gjr_garch", "arma_egarch_xgboost")

#: The schedule the main run uses, and the arithmetic it implies. Repeated rather than
#: imported for the same reason as the schema: the driver asserts the numbers the
#: specification's guardrails produce, so it has to state them.
#:
#: **Why 180 and not 100.** The four-hour fixture is 299 returns long, so a 100-bar window
#: is a *third* of the series — and at that length the EGARCH leg of the hybrid is not
#: identified. Measured on this exact fixture at the positions the run scores: with a
#: 100-bar window the hybrid's one-step forecast came back at ``3.6e-130`` and its EGARCH
#: companion at ``1.6e-130``, against a window mean squared return of ``~2e-6`` — a
#: collapsed recursion, which the engine now refuses (§118), turning the strongest checks in
#: this file (every row finite, every model scored on every step) into failures that are
#: about the fixture rather than the code. At 180 every model returns a forecast between
#: 0.06 and 10.6 times its own window's mean squared return and nothing is refused. The
#: window is the shortest one measured to be estimable for all five models, so it is the
#: one that keeps the fixture honest without making the schedule trivial. The scored region
#: does not move when this changes: ``WalkForwardSpec.schedule`` starts at
#: ``n_rows - test_window`` and never consults ``train_window``, so the indices asserted
#: below are the same at 100 and at 180.
TRAIN = 180
TEST = 80
STEP = 2
N_SCORED = -(-TEST // STEP)  # 40
HORIZON = 12

#: The guardrails, as literals. If the server moves one, the driver should fail loudly
#: rather than follow it — a silently relaxed minimum is a change nobody reviewed.
MIN_TRAIN_WINDOW = 50
MIN_SCORED_STEPS = 30

#: The four-hour bar the resampled frame declares, in seconds. The forward path's timestamps
#: are projected with the *declared* timeframe, so this is what they must be spaced by.
FOUR_HOURS = 14400.0


def call(method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    """One request, via urllib, for a JSON response.

    Not httpx: on this machine a pooled httpx connection to uvicorn was reset before the
    request reached the server, which made a transport quirk look like a failing endpoint.
    The stdlib client opens a fresh connection per call, so a failure here is a failure of
    the API.

    ``path`` is relative to ``BASE``. The prefix is already in ``BASE``, and repeating it
    turns ``/jobs/x`` into ``/api/api/jobs/x`` — which 404s in a way indistinguishable from
    a job that does not exist, so the mistake would read as a missing feature.
    """
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read().decode()
            return response.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as error:
        raw = error.read().decode()
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, {"raw": raw}


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}{(' - ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(label)


def error_message(payload: Any) -> str:
    return str(((payload or {}).get("error") or {}).get("message", ""))


def job_error(job: dict[str, Any]) -> str:
    """The message a *failed job* carries, which arrives through a different field.

    A request-level refusal is an HTTP status with an ``error`` envelope. A refusal that
    needs the file read cannot be either — the response was a 202 twenty seconds ago — so it
    is written onto the job record. Reading the two through one helper would hide which one
    happened, and which one happened is half of what is being asserted.
    """
    return str(((job or {}).get("error") or {}).get("message", ""))


def failed_job(dataset_id: str, **overrides: Any) -> tuple[int, dict[str, Any]]:
    """POST a forecast and poll it, expecting the run to be refused inside the job."""
    status, payload = forecast(dataset_id, **overrides)
    if status != 202 or not isinstance(payload, dict) or "job_id" not in payload:
        return status, {}
    return status, poll(str(payload["job_id"]))


def rejection_message(status: int, payload: Any) -> str:
    """The most informative refusal text available, from either refusal shape.

    A request-level refusal is an HTTP status carrying ``{"error": {"message": ...}}``, and a
    refusal that needs the file read arrives on the job record instead (see ``job_error``).
    Both are folded into one string here and the *fragment* is what each check must find in
    it — a check that read only one of them and searched the raw JSON would pass on a field
    name appearing anywhere in the payload, which is how a driver ends up asserting nothing.

    The ``{"detail": [...]}`` fallback is kept for the shape FastAPI produced before
    ``app/core/errors.py`` grew its ``RequestValidationError`` handler (``DECISIONS §117``).
    Every 422 now leaves in the application's envelope, so the fallback should never be the
    one that fires — and because a fallback would make a regression *look* like a pass, the
    refusal loop below asserts the envelope's ``code`` in addition to the message.
    """
    return error_message(payload) or json.dumps(payload)[:400]


def upload(name: str, body: str) -> str:
    """POST a CSV as multipart/form-data, built by hand to keep the driver stdlib-only."""
    boundary = "----quantmetrics-e2e-volatility-boundary"
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        "Content-Type: text/csv\r\n\r\n"
        f"{body}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    request = urllib.request.Request(
        f"{BASE}/datasets/upload",
        data=payload,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            created = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        print(f"upload failed: {error.code} {error.read().decode()[:300]}")
        sys.exit(1)
    return str(created["dataset_id"])


# --------------------------------------------------------------------- the frame


def _gaussian(state: int) -> tuple[float, int]:
    """One standard normal draw, from a hand-rolled linear congruential generator.

    Deliberately not ``random``: the driver asserts that two identical requests produce
    identical forecasts, and a generator whose stream changes between Python releases would
    make that check fail for a reason that has nothing to do with the module.
    """
    state = (1103515245 * state + 12345) % (2**31)
    u1 = max(state / (2**31), 1e-12)
    state = (1103515245 * state + 12345) % (2**31)
    u2 = state / (2**31)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2), state


def garch_hourly_csv(
    count: int,
    *,
    seed: int = 20240101,
    omega: float = 2.0e-6,
    alpha: float = 0.07,
    beta: float = 0.88,
) -> str:
    """A deterministic hourly OHLCV frame whose returns follow a GARCH(1,1).

    Clustered volatility rather than white noise, because a volatility model tested on a
    constant-variance series is a model tested on nothing: every estimator converges to the
    same flat answer and the metrics are all tie. The recursion is the textbook one,
    ``σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}``, with ``α + β = 0.95`` — persistent enough that
    the scored span has regime-like quiet and violent stretches, comfortably below one so
    the series is stationary and the module's own ADF guard lets it through.

    Volume rises with the conditional volatility, so the hybrid legs' volume feature carries
    information rather than being a constant the learner has to discover it cannot use.
    """
    state = seed
    price = 100.0
    variance = omega / (1.0 - alpha - beta)
    previous = 0.0
    rows = ["open_time,open,high,low,close,volume"]
    for index in range(count):
        z, state = _gaussian(state)
        variance = omega + alpha * previous * previous + beta * variance
        previous = math.sqrt(variance) * z
        open_price = price
        price *= math.exp(previous)
        close = price
        high = max(open_price, close) * 1.0008
        low = min(open_price, close) * 0.9992
        stamp = datetime(2023, 1, 1) + timedelta(hours=index)
        volume = 1000.0 + 200.0 * math.sin(index / 7.0) + 20000.0 * math.sqrt(variance)
        rows.append(
            f"{stamp.strftime('%Y-%m-%dT%H:%M:%SZ')},{open_price:.6f},{high:.6f},"
            f"{low:.6f},{close:.6f},{volume:.6f}"
        )
    return "\n".join(rows) + "\n"


def random_walk_csv(count: int, *, seed: int = 777) -> str:
    """A well-formed OHLCV frame carrying one column that is deliberately not stationary.

    ``cumulative_volume`` is a running total, which is the case the stationarity guard
    exists for: a user pointing ``price_source`` at a column that is not a return series at
    all. A drift is added so the unit-root null is not rejected for the right reason *or*
    the wrong one — the test is run with a constant and no trend term, so a drifting random
    walk is refused by a margin no draw size makes marginal.
    """
    state = seed
    price = 100.0
    cumulative = 0.0
    rows = ["open_time,open,high,low,close,volume,cumulative_volume"]
    for index in range(count):
        z, state = _gaussian(state)
        open_price = price
        price *= math.exp(0.0015 * z)
        close = price
        high = max(open_price, close) * 1.0008
        low = min(open_price, close) * 0.9992
        stamp = datetime(2023, 1, 1) + timedelta(hours=index)
        volume = 1000.0 + 100.0 * math.sin(index / 7.0)
        cumulative += volume + 0.5
        rows.append(
            f"{stamp.strftime('%Y-%m-%dT%H:%M:%SZ')},{open_price:.6f},{high:.6f},"
            f"{low:.6f},{close:.6f},{volume:.6f},{cumulative:.6f}"
        )
    return "\n".join(rows) + "\n"


#: 1200 hourly bars, resampled to 300 four-hour bars. The scored span is the last 80 bars of
#: the return series, so the run has 299-80=219 bars of history behind its first scored bar
#: against a 100-bar training window — room to spare, which is what makes the schedule checks
#: about the arithmetic rather than about a dataset that only just fits.
GARCH_CSV = garch_hourly_csv(1200)
WALK_CSV = random_walk_csv(200)


# ------------------------------------------------------------------- the harness


def forecast(dataset_id: str, **overrides: Any) -> tuple[int, Any]:
    body: dict[str, Any] = {
        "dataset_id": dataset_id,
        "price_source": "close",
        "models": list(MODELS),
        "distribution": "students_t",
        "mean_model": "constant",
        "p": 1,
        "o": 1,
        "q": 1,
        "lags": 3,
        "walk_forward": {"train_window": TRAIN, "test_window": TEST, "step_size": STEP},
        "forecast_horizon": HORIZON,
        "conditioning_dataset_id": None,
        "name": None,
        "write": True,
    }
    for key, value in overrides.items():
        if key == "walk_forward" and isinstance(value, dict):
            body["walk_forward"] = {**body["walk_forward"], **value}
        else:
            body[key] = value
    return call("POST", "/volatility/forecast", body)


def poll(job_id: str, *, timeout: float = 900.0) -> dict[str, Any]:
    """Poll a job to a terminal state, as the browser does."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, body = call("GET", f"/jobs/{job_id}")
        if status != 200 or not isinstance(body, dict):
            print(f"polling {job_id} failed: {status} {json.dumps(body)[:200]}")
            sys.exit(1)
        if body["status"] != "running":
            return body
        time.sleep(0.2)
    print(f"job {job_id} did not finish within {timeout:.0f}s")
    sys.exit(1)


def run(dataset_id: str, **overrides: Any) -> dict[str, Any]:
    """Accept a run and wait for it, returning the terminal job record."""
    status, accepted = forecast(dataset_id, **overrides)
    if status != 202 or not isinstance(accepted, dict):
        print(f"forecast was not accepted: {status} {json.dumps(accepted)[:300]}")
        sys.exit(1)
    return poll(str(accepted["job_id"]))


def report_of(job: dict[str, Any]) -> dict[str, Any]:
    return ((job or {}).get("result") or {}).get("report") or {}


def dataset_of(job: dict[str, Any]) -> dict[str, Any]:
    return ((job or {}).get("result") or {}).get("dataset") or {}


def preview_of(job: dict[str, Any]) -> dict[str, Any]:
    return ((job or {}).get("result") or {}).get("preview") or {}


def columns_of(preview: dict[str, Any]) -> list[str]:
    return [str(column["name"]) for column in preview.get("columns", [])]


def score_of(report: dict[str, Any], model: str) -> dict[str, Any]:
    for entry in report.get("scores") or []:
        if entry.get("model") == model:
            return entry
    return {}


def path_of(report: dict[str, Any], model: str) -> dict[str, Any]:
    for entry in report.get("forward_paths") or []:
        if entry.get("model") == model:
            return entry
    return {}


def page_rows(dataset_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
    """Every row of a saved dataset, in one page, in the order it was written.

    Unsorted, deliberately: the stored frame is supposed to be chronological, and asking
    the page endpoint to sort it would hide a regression in how it was written — which is
    exactly the property a reader of the file (rather than of the API) depends on.
    """
    status, body = call("GET", f"/datasets/{dataset_id}/page?offset=0&limit={limit}")
    if status != 200 or not isinstance(body, dict):
        print(f"paging {dataset_id} failed: {status} {json.dumps(body)[:200]}")
        sys.exit(1)
    if body.get("total", 0) > len(body.get("rows", [])):
        print(f"{dataset_id} has {body['total']} rows — more than one page; widen the driver")
        sys.exit(1)
    return list(body.get("rows", []))


def as_time(value: Any) -> datetime:
    """Parse an ISO stamp, tolerating both the ``Z`` and ``+00:00`` spellings."""
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def close_enough(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    """Whether two JSON numbers agree, treating null as "not comparable" rather than zero."""
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= tolerance * max(1.0, abs(float(right)))


# --------------------------------------------------------------------- seeding

print("== seeding ==")
raw_id = upload("e2e_volatility_hourly.csv", GARCH_CSV)
print(f"  uploaded={raw_id}")

# A declared-timeframe frame, produced by the merge endpoint's resample. Module E does not
# refuse a dataset that declares no timeframe — it falls back to the spacing it measures —
# but the *forward path's* timestamps are projected with the declared bar length, so the two
# cases project on different rules and both are checked below.
status, merged = call(
    "POST",
    "/datasets/merge",
    {
        "base_dataset_id": raw_id,
        "name": "E2E volatility 4h base",
        "timeframe": "4h",
        "joins": [],
        "write": True,
    },
)
source_id = ((merged or {}).get("dataset") or {}).get("dataset_id") if status == 201 else None
check(
    "a declared-timeframe dataset is produced by resampling",
    status == 201 and bool(source_id),
    f"status={status} id={source_id}" + ("" if source_id else f" {json.dumps(merged)[:140]}"),
)
if source_id is None:
    print("cannot continue without a source dataset")
    sys.exit(1)

status, source_meta = call("GET", f"/datasets/{source_id}")
source_meta = source_meta or {}
check("the source dataset reports a checksum", bool(source_meta.get("checksum")),
      f"status={status}")
check(
    "the source dataset declares the timeframe it was resampled to",
    source_meta.get("timeframe") == "4h",
    str(source_meta.get("timeframe")),
)
source_end = as_time(source_meta.get("end_timestamp"))
print(f"  source={source_id} rows={source_meta.get('row_count')} end={source_end}")

walk_id = upload("e2e_volatility_walk.csv", WALK_CSV)
print(f"  uploaded={walk_id}")

# --------------------------------------------------------------------- regime

print("== a Module D output to condition on ==")

# The conditioning path is only exercised against a real regime dataset. A synthetic
# probability frame would test the join and not the interface: the dropdown offers a regime
# output only when it carries an observation-time column, and the run refuses one that does
# not. Building the conditioning input through the module that produces it in production is
# what makes the check end to end.
status, accepted = call(
    "POST",
    "/regime/detect",
    {
        "dataset_id": source_id,
        "model_type": "hmm",
        "n_states": 4,
        "min_state_duration": 5,
        "crisis_threshold": 0.35,
        "switching_cost_penalty": 0.005,
        "min_train_bars": 60,
        "refit_every": 60,
        "volatility_window": 24,
        "write": True,
    },
)
regime_job = poll(str(accepted["job_id"])) if status == 202 else {}
regime_id = dataset_of(regime_job).get("dataset_id")
check(
    "a regime dataset is produced to condition on",
    regime_job.get("status") == "succeeded" and bool(regime_id),
    f"status={status} job={regime_job.get('status')} id={regime_id} "
    f"{json.dumps(regime_job.get('error'))[:160]}",
)
if not regime_id:
    print("cannot check conditioning without a regime dataset")
    sys.exit(1)

status, regime_meta = call("GET", f"/datasets/{regime_id}")
regime_columns = [str(column["name"]) for column in (regime_meta or {}).get("columns", [])]
check(
    "the regime output carries the observation-time column the join is keyed on",
    "open_time" in regime_columns and "close_time" not in regime_columns,
    str(regime_columns),
)
check(
    "the regime output carries the posterior columns conditioning expects",
    {"prob_crisis", "confidence"} <= set(regime_columns),
    str(regime_columns),
)

# --------------------------------------------------------------------- options

print("== options ==")

status, raw_options = call("GET", f"/volatility/options?dataset_id={raw_id}")
check("options are served for an uploaded dataset", status == 200, f"status={status}")
raw_options = raw_options or {}
check(
    "an uploaded dataset has no timeframe, so the forward path falls back to measured spacing",
    raw_options.get("timeframe") is None
    and "no_timeframe" in [warning.get("code") for warning in raw_options.get("warnings", [])],
    json.dumps(raw_options.get("warnings"))[:200],
)
check(
    "no declared timeframe is a caution rather than a refusal",
    [warning["code"] for warning in raw_options.get("warnings", [])
     if warning["level"] == "error"] == [],
    json.dumps(raw_options.get("warnings"))[:200],
)

status, options = call("GET", f"/volatility/options?dataset_id={source_id}")
check("options are served for the source dataset", status == 200, f"status={status}")
options = options or {}

check(
    "the options report the row count and the shorter return series a walk-forward runs over",
    options.get("rows") == source_meta.get("row_count")
    and options.get("usable_returns") == source_meta.get("row_count") - 1,
    f"rows={options.get('rows')} usable={options.get('usable_returns')} "
    f"frame={source_meta.get('row_count')}",
)

# The guardrails the request schema enforces, sent so the page's live arithmetic uses the
# same numbers a refusal would. Asserted against literals: a silently relaxed minimum is a
# change nobody reviewed, and it would be invisible in a form that read its bounds from the
# server it was relaxing.
limits = options.get("limits") or {}
check(
    "the options carry the guardrails the request schema enforces",
    limits.get("min_train_window") == MIN_TRAIN_WINDOW
    and limits.get("min_scored_steps") == MIN_SCORED_STEPS
    and limits.get("min_step_size") == 1
    and limits.get("max_forecast_horizon") == 500
    and limits.get("max_order") == 5
    and limits.get("max_lags") == 100,
    json.dumps(limits),
)
check(
    "the derived default schedule is one this dataset can actually run",
    options.get("feasible") is True
    and (options.get("defaults") or {}).get("train_window", 0)
    + (options.get("defaults") or {}).get("test_window", 0)
    <= options.get("usable_returns", 0),
    f"feasible={options.get('feasible')} defaults={json.dumps(options.get('defaults'))[:160]}",
)

sources = {entry["id"]: entry for entry in options.get("sources", [])}
check(
    "close is offered as a return source because the frame has a close column",
    sources.get("close", {}).get("available") is True,
    json.dumps(sources.get("close"))[:160],
)
check(
    "the derived log_return column is documented as absent rather than offered",
    sources.get("log_return", {}).get("available") is False
    and bool(sources.get("log_return", {}).get("reason")),
    json.dumps(sources.get("log_return"))[:160],
)

models = {entry["id"]: entry for entry in options.get("models", [])}
check(
    "all five models are offered, grouped into the two families",
    set(models) == {"garch", "egarch", "gjr_garch", "arma_egarch_xgboost", "tdnn_gjr_garch"}
    and models["garch"]["family"] == "simple"
    and models["gjr_garch"]["family"] == "simple"
    and models["arma_egarch_xgboost"]["family"] == "hybrid"
    and models["tdnn_gjr_garch"]["family"] == "hybrid",
    json.dumps({k: v.get("family") for k, v in models.items()}),
)
check(
    "the parameters a model consumes are declared, so a control that does nothing is visible",
    models["garch"]["needs"] == ["p", "q"]
    and sorted(models["egarch"]["needs"]) == ["o", "p", "q"]
    and "lags" in models["arma_egarch_xgboost"]["needs"]
    and "lags" not in models["garch"]["needs"],
    json.dumps({k: v.get("needs") for k, v in models.items()}),
)

metrics = {entry["id"]: entry for entry in options.get("metrics", [])}
check(
    "the metrics table's columns are sent with the direction that counts as better",
    set(metrics) == {"mqlike", "rmse", "mae", "mz_r2"}
    and metrics["mqlike"]["lower_is_better"] is True
    and metrics["mz_r2"]["lower_is_better"] is False,
    json.dumps({k: v.get("lower_is_better") for k, v in metrics.items()}),
)
check(
    "MZ R² is marked as a diagnostic rather than a ranking, so nothing crowns its maximum",
    metrics["mz_r2"]["is_ranking"] is False
    and all(metrics[key]["is_ranking"] is True for key in ("mqlike", "rmse", "mae")),
    json.dumps({k: v.get("is_ranking") for k, v in metrics.items()}),
)

conditioning = {entry["dataset_id"]: entry for entry in options.get("conditioning", [])}
check(
    "the regime output is offered as a conditioning dataset, and is usable",
    conditioning.get(regime_id, {}).get("usable") is True,
    json.dumps(conditioning.get(regime_id))[:200],
)
check(
    "the conditioning entry names the posterior columns it contributes",
    set(conditioning.get(regime_id, {}).get("columns") or [])
    >= {"prob_crisis", "prob_trending_up", "confidence"},
    json.dumps(conditioning.get(regime_id, {}).get("columns"))[:200],
)

# --------------------------------------------------------------- the handshake

print("== the job handshake ==")

status, accepted = forecast(source_id)
check(
    "the forecast endpoint accepts the work with a 202 rather than a result",
    status == 202
    and isinstance(accepted, dict)
    and accepted.get("status") == "running"
    and accepted.get("poll") == f"/api/jobs/{accepted.get('job_id')}"
    and "result" not in accepted
    and "report" not in accepted,
    f"status={status} {json.dumps(accepted)[:200]}",
)
if status != 202:
    print("cannot continue without an accepted run")
    sys.exit(1)

job = poll(str(accepted["job_id"]))
check(
    "the run reaches succeeded with an output dataset",
    job.get("status") == "succeeded" and bool(job.get("output_dataset_id")),
    f"status={job.get('status')} output={job.get('output_dataset_id')} "
    f"{json.dumps(job.get('error'))[:200]}",
)
check(
    "the job's one-line message says what happened rather than that something did",
    "MQLIKE" in str(job.get("message") or ""),
    str(job.get("message"))[:160],
)
if job.get("status") != "succeeded":
    print(f"cannot continue: {json.dumps(job.get('error'))[:400]}")
    sys.exit(1)

report = report_of(job)
output_id = str(job["output_dataset_id"])

# ------------------------------------------------------------- the saved artifact

print("== the saved artifact ==")

status, output_meta = call("GET", f"/datasets/{output_id}")
output_meta = output_meta or {}
output_columns = [str(column["name"]) for column in output_meta.get("columns", [])]
check(
    "the stored dataset carries the specified schema in the specified order",
    tuple(output_columns) == VOL_FORECAST_COLUMNS,
    str(output_columns),
)
check(
    "no close_time column is carried, so a mistaken join cannot look correct",
    "close_time" not in output_columns,
    str(output_columns),
)
check(
    "no is_forward_path column is carried, so no projected row can be filtered wrongly",
    "is_forward_path" not in output_columns,
    str(output_columns),
)
check(
    "the preview of the in-memory frame carries the same schema as the stored file",
    tuple(columns_of(preview_of(job))) == VOL_FORECAST_COLUMNS,
    str(columns_of(preview_of(job))),
)
check(
    "the forecast lands in outputs, as the product of an analysis rather than an input",
    output_meta.get("category") == "outputs" and output_meta.get("operation") == "volatility",
    f"category={output_meta.get('category')} operation={output_meta.get('operation')}",
)
check(
    "the saved forecast records the price dataset as its parent",
    output_meta.get("parent_dataset_id") == source_id,
    f"parent={output_meta.get('parent_dataset_id')} expected={source_id}",
)

provenance = output_meta.get("provenance") or {}
check(
    "the provenance records what would be needed to reproduce the run",
    provenance.get("module_source") == "volatility"
    and provenance.get("distribution") == "students_t"
    and provenance.get("mean_model") == "constant"
    and provenance.get("models") == ",".join(MODELS)
    and provenance.get("train_window") == str(TRAIN)
    and provenance.get("test_window") == str(TEST)
    and provenance.get("step_size") == str(STEP),
    json.dumps({k: provenance.get(k) for k in
                ("module_source", "distribution", "models", "train_window")})[:240],
)

rows = page_rows(output_id)
check(
    "the stored frame holds one row per model per scored step, and nothing else",
    len(rows) == N_SCORED * len(MODELS) and output_meta.get("row_count") == len(rows),
    f"rows={len(rows)} expected={N_SCORED * len(MODELS)} "
    f"meta={output_meta.get('row_count')}",
)

times = [as_time(row["open_time"]) for row in rows]
check(
    "the stored frame is chronological, so any later module can read it as a series",
    all(times[index] <= times[index + 1] for index in range(len(times) - 1)),
    f"{len(times)} timestamps",
)
check(
    "no stored row is dated after the last observation, so no projection reached the store",
    max(times) <= source_end,
    f"last row {max(times)} vs parent end {source_end}",
)

per_model: dict[str, list[dict[str, Any]]] = {name: [] for name in MODELS}
for row in rows:
    per_model.setdefault(str(row["model"]), []).append(row)
check(
    "every selected model appears, and no other model does",
    set(per_model) == set(MODELS),
    str(sorted(per_model)),
)
check(
    "each model is scored on exactly the same bars",
    len({frozenset(str(row["open_time"]) for row in group) for group in per_model.values()})
    == 1
    and all(len(group) == N_SCORED for group in per_model.values()),
    json.dumps({name: len(group) for name, group in per_model.items()}),
)
check(
    "each model's own timeline is strictly ascending with no repeats",
    all(
        all(
            as_time(group[index]["open_time"]) < as_time(group[index + 1]["open_time"])
            for index in range(len(group) - 1)
        )
        for group in per_model.values()
    ),
    "per-model timestamps",
)

# The identities the frame is supposed to satisfy. A variance and its square root written by
# two separate code paths is exactly where a stale column comes from, and a reader who trusts
# the wrong one sees a chart that is wrong by a square.
#
# A refused row is not exempt from having a shape. It must carry *no* forecast at all: half a
# row — a null variance beside a stale volatility, or a ratio formed against the null — is
# worse than a blank, because a reader takes it for a number. So the loop asserts two things:
# every finite row satisfies all four identities, and every refused row has all four
# forecast-derived columns null while still reporting the realised variance it can compute.
# Written this way rather than by skipping the non-finite rows, because a skip would also
# skip the question of what was written in their place.
identity_ok = True
refusal_shape_ok = True
non_finite_rows = 0
for row in rows:
    variance = row["forecast_variance"]
    volatility = row["forecast_volatility"]
    actual_variance = row["actual_variance"]
    actual_volatility = row["actual_volatility"]
    if actual_variance is None or not close_enough(
        actual_volatility, math.sqrt(float(actual_variance)), 1e-9
    ):
        identity_ok = False
        break
    if row["finite"] is not True:
        non_finite_rows += 1
        if any(
            row[column] is not None
            for column in (
                "forecast_variance",
                "forecast_volatility",
                "error_variance",
                "ratio",
            )
        ):
            refusal_shape_ok = False
            break
        continue
    if variance is None or float(variance) <= 0.0:
        identity_ok = False
        break
    if not close_enough(volatility, math.sqrt(float(variance)), 1e-9):
        identity_ok = False
        break
    if not close_enough(row["error_variance"], float(actual_variance) - float(variance), 1e-9):
        identity_ok = False
        break
    if not close_enough(row["ratio"], float(actual_variance) / float(variance), 1e-9):
        identity_ok = False
        break
check(
    "every finite row satisfies volatility² = variance, error = actual − forecast, and "
    "ratio = actual/forecast",
    identity_ok,
    f"{len(rows)} rows checked",
)
check(
    "a refused row carries no forecast column at all, rather than a half-written one",
    refusal_shape_ok,
    f"{non_finite_rows} refused rows of {len(rows)}",
)
check(
    "no scored step came back non-finite, so the metrics below are over the whole span",
    all(row["finite"] is True for row in rows),
    f"{non_finite_rows} non-finite",
)

# ---------------------------------------------------------------- the scored region

print("== the scored region ==")

first = report.get("first_scored_index")
last = report.get("last_scored_index")
return_rows = report.get("return_rows")
check(
    "the return series is one bar shorter than the frame, both of them reported",
    report.get("rows") == source_meta.get("row_count")
    and return_rows == source_meta.get("row_count") - 1,
    f"rows={report.get('rows')} returns={return_rows} frame={source_meta.get('row_count')}",
)
check(
    "returns were derived from close and the report says which column they are",
    report.get("derived_returns") is True
    and report.get("return_column") == "log_return"
    and report.get("price_source") == "close",
    f"derived={report.get('derived_returns')} column={report.get('return_column')} "
    f"source={report.get('price_source')}",
)
check(
    "the scored region is the last test_window bars of the return series",
    first == return_rows - TEST
    and last == first + (N_SCORED - 1) * STEP,
    f"first={first} last={last} expected_first={return_rows - TEST}",
)
check(
    "the scored span's own timestamps match the frame that was written",
    as_time(report.get("first_scored_time")) == min(times)
    and as_time(report.get("last_scored_time")) == max(times),
    f"{report.get('first_scored_time')} .. {report.get('last_scored_time')}",
)
check(
    "the scored span ends before the last bar, because the last bar has no forecast to score",
    as_time(report.get("last_scored_time")) < source_end,
    f"last scored {report.get('last_scored_time')} vs parent end {source_end}",
)
check(
    "the fitting scale is reported, so the ×100 conditioning transform is not invisible",
    report.get("scale_factor") == 100.0,
    str(report.get("scale_factor")),
)
check(
    "stationarity was tested and reported before anything was fitted",
    report.get("stationary") is True
    and isinstance(report.get("adf_pvalue"), (int, float))
    and float(report["adf_pvalue"]) <= 0.05,
    f"stationary={report.get('stationary')} p={report.get('adf_pvalue')} "
    f"stat={report.get('adf_statistic')}",
)
check(
    "the realised-variance proxy is named, because the metrics depend on which it is",
    report.get("actual_proxy") == "squared log return",
    str(report.get("actual_proxy")),
)

# ------------------------------------------------------------------- the metrics

print("== the metrics ==")

check(
    "the report carries one score per model and one DM test per pair",
    len(report.get("scores") or []) == len(MODELS)
    and len(report.get("comparisons") or []) == len(MODELS) * (len(MODELS) - 1) // 2,
    f"{len(report.get('scores') or [])} scores, "
    f"{len(report.get('comparisons') or [])} comparisons",
)

scored_ok = True
detail = []
for name in MODELS:
    entry = score_of(report, name)
    detail.append(
        f"{name}: n={entry.get('n_steps')}/{entry.get('n_finite')} mqlike={entry.get('mqlike')}"
    )
    if (
        entry.get("n_steps") != N_SCORED
        or entry.get("n_finite") != N_SCORED
        or entry.get("n_nan") != 0
        or entry.get("n_inf") != 0
        or entry.get("mqlike") is None
        or entry.get("unscored_reason") is not None
    ):
        scored_ok = False
check(
    "every model is scored over every step, with the non-finite counts reported as zero",
    scored_ok,
    " | ".join(detail)[:300],
)
# The frame and the report are two renderings of one walk-forward, produced by two different
# functions, and the thing they can silently disagree about is *which* steps produced a
# number. Tying them together is two lines here and is a check nothing else in this file
# makes: the report's own off-by-one (§122) lived in this gap, invisible to every assertion
# that read one of the two alone.
check(
    "the frame's non-finite rows and the report's counts are the same number, per model",
    all(
        sum(1 for row in per_model[name] if row["finite"] is not True)
        == int(score_of(report, name).get("n_nan") or 0)
        + int(score_of(report, name).get("n_inf") or 0)
        and sum(1 for row in per_model[name] if row["finite"] is True)
        == int(score_of(report, name).get("n_finite") or 0)
        for name in MODELS
    ),
    " | ".join(
        f"{name}: frame={sum(1 for row in per_model[name] if row['finite'] is not True)} "
        f"report={score_of(report, name).get('n_nan')}+{score_of(report, name).get('n_inf')}"
        for name in MODELS
    )[:300],
)
check(
    "MQLIKE is non-negative, as the ratio form of the loss is by construction",
    all(
        isinstance(score_of(report, name).get("mqlike"), (int, float))
        and float(score_of(report, name)["mqlike"]) >= 0.0
        for name in MODELS
    ),
    " | ".join(f"{name}={score_of(report, name).get('mqlike')}" for name in MODELS),
)
check(
    "the Mincer–Zarnowitz regression is reported with its slope and intercept beside the R²",
    all(
        score_of(report, name).get("mz_slope") is not None
        and score_of(report, name).get("mz_intercept") is not None
        and score_of(report, name).get("mz_r2") is not None
        for name in MODELS
    ),
    " | ".join(f"{name} r2={score_of(report, name).get('mz_r2')}" for name in MODELS),
)
check(
    "the near-zero MZ R² is explained in the report rather than left to be misread",
    any("Mincer" in warning and "proxy" in warning for warning in report.get("warnings") or []),
    json.dumps(report.get("warnings"))[:240],
)

comparison_ok = True
comparison_detail = []
for entry in report.get("comparisons") or []:
    # Read as ``None`` first and reject it explicitly. ``float(None)`` raises, and a crash
    # here would take the remaining thirty checks in this file down with it — turning a
    # reportable failure into a traceback that says nothing about which model was at fault.
    raw_statistic = entry.get("statistic")
    raw_p_value = entry.get("p_value")
    if not isinstance(raw_statistic, (int, float)) or not isinstance(raw_p_value, (int, float)):
        comparison_ok = False
        comparison_detail.append(
            f"{entry.get('model_a')} vs {entry.get('model_b')}: no statistic reported"
        )
        continue
    statistic = float(raw_statistic)
    p_value = float(raw_p_value)
    better = entry.get("better")
    expected = (
        entry.get("model_a") if statistic < 0.0
        else (entry.get("model_b") if statistic > 0.0 else None)
    )
    comparison_detail.append(
        f"{entry.get('model_a')} vs {entry.get('model_b')} t={statistic:.3f} "
        f"p={p_value:.3f} better={better}"
    )
    if (
        not 0.0 <= p_value <= 1.0
        or not 0.0 < int(entry.get("n_effective", 0)) <= N_SCORED
        or better != expected
        or bool(entry.get("significant")) != (p_value < 0.05)
    ):
        comparison_ok = False
check(
    "each DM test reports a p-value, a sample size and the better model, consistently",
    comparison_ok,
    " | ".join(comparison_detail)[:320],
)
check(
    "the DM verdict is written as a boolean beside the p-value rather than left to the reader",
    all(isinstance(entry.get("significant"), bool) for entry in report.get("comparisons") or []),
    f"{len(report.get('comparisons') or [])} comparisons",
)

# ------------------------------------------------------------------- the fan chart

print("== the forward path ==")

check(
    "the forward step is the declared four-hour bar, and the report says which rule produced it",
    close_enough(report.get("forward_step_seconds"), FOUR_HOURS, 1e-12)
    and report.get("forward_step_source") == "declared timeframe",
    f"step={report.get('forward_step_seconds')} source={report.get('forward_step_source')}",
)

quantiles = report.get("error_quantiles") or {}
check(
    "the error quantiles are carried per model, on the ratio the bands are drawn from",
    set(quantiles) == set(MODELS)
    and all(
        set(quantiles[name]) == {"0.05", "0.25", "0.50", "0.75", "0.95"} for name in MODELS
    ),
    json.dumps({name: sorted(quantiles.get(name, {})) for name in MODELS})[:240],
)
check(
    "the error quantiles are ordered, so the bands they draw cannot cross",
    all(
        list(quantiles[name][key] for key in ("0.05", "0.25", "0.50", "0.75", "0.95"))
        == sorted(quantiles[name][key] for key in ("0.05", "0.25", "0.50", "0.75", "0.95"))
        for name in MODELS
    ),
    json.dumps(quantiles.get(MODELS[0]))[:200],
)

paths = report.get("forward_paths") or []
check(
    "every model carries a forward path of exactly the requested horizon",
    len(paths) == len(MODELS)
    and all(len(path.get("variance") or []) == HORIZON for path in paths)
    and all(len(path.get("volatility") or []) == HORIZON for path in paths)
    and all(len(path.get("times") or []) == HORIZON for path in paths),
    json.dumps({path.get("model"): len(path.get("variance") or []) for path in paths}),
)
check(
    "every forward timestamp is strictly after the last observation",
    all(
        as_time(path["times"][0]) > source_end
        and all(
            as_time(path["times"][index]) < as_time(path["times"][index + 1])
            for index in range(len(path["times"]) - 1)
        )
        for path in paths
    ),
    f"first={paths[0].get('times', [None])[0] if paths else None} end={source_end}",
)
# A step the engine refused is written as JSON ``null`` in both columns, so the root is taken
# only where there is a variance to take it of. Where there is not, the two columns must be
# null *together*: a null variance beside a surviving volatility is the failure this looks
# for, and it would plot as a spike to zero in the fan chart rather than as the gap it is.
forward_identity_ok = True
for path in paths:
    # ``strict=False`` and not ``True``: the two arrays are asserted to be ``HORIZON`` long a
    # few checks above, and an assertion that *raises* mid-driver takes every later check down
    # with it. A length regression is reported by that check, with both numbers in it.
    for path_variance, path_volatility in zip(
        path["variance"], path["volatility"], strict=False
    ):
        if path_variance is None or path_volatility is None:
            if path_variance is not None or path_volatility is not None:
                forward_identity_ok = False
                break
            continue
        if not close_enough(path_volatility, math.sqrt(float(path_variance)), 1e-9):
            forward_identity_ok = False
            break
    if not forward_identity_ok:
        break
check(
    "the forward path's volatility is the square root of its variance, and the two are null "
    "together on a refused step",
    forward_identity_ok,
    f"{len(paths)} paths",
)
# Same rule for the bands: a refused step nulls all four of its endpoints, because a band
# drawn around a variance that does not exist is a picture of a number that does not exist.
# Where the four are present, the outer interval must contain the inner one — not merely be
# ordered, which is a weaker statement and would pass two intervals that had been swapped.
bands_ok = True
for path in paths:
    bands = path.get("bands") or []
    if [band.get("label") for band in bands] != ["5–95%", "25–75%"]:
        bands_ok = False
        break
    outer, inner = bands
    for index in range(len(outer["low"])):
        endpoints = [
            outer["low"][index],
            outer["high"][index],
            inner["low"][index],
            inner["high"][index],
        ]
        if any(value is None for value in endpoints):
            if not all(value is None for value in endpoints):
                bands_ok = False
                break
            continue
        if not (
            float(outer["low"][index]) <= float(outer["high"][index])
            and float(inner["low"][index]) <= float(inner["high"][index])
            and float(outer["low"][index]) <= float(inner["low"][index])
            and float(outer["high"][index]) >= float(inner["high"][index])
        ):
            bands_ok = False
            break
    if not bands_ok:
        break
check(
    "each forward path carries the two empirical bands, ordered and non-crossing",
    bands_ok,
    json.dumps([band.get("label") for band in (paths[0].get("bands") or [])]) if paths else "none",
)
check(
    "the forward path is not in the stored dataset, which holds observations only",
    all(row["model"] != "forward_path" for row in rows) and max(times) < source_end,
    f"last stored {max(times)} < parent end {source_end}",
)

# ---------------------------------------------------------------- determinism

print("== determinism ==")

# Run on the *hourly* upload rather than the resampled frame, which checks the second
# half of the forward-step rule at the same time: with no declared timeframe the bar length
# is measured from the data. The schedule is short on purpose — this pair of runs exists to
# compare two payloads, not to score anything, and ten steps is enough to make the two
# comparable.
#
# **Why 400 and not 100.** A one-hour bar carries four times less information than the
# four-hour bar the main run uses, so the same 100 observations identify far less. Measured
# on this upload at the ten scored positions: a 100-, 180- and 220-bar window all leave the
# EGARCH leg collapsed (its one-step forecast falls to ``1e-46``–``1e-76`` times the window's
# mean squared return), and it is identified again at 400 (ratios 0.75–1.33 across the ten
# steps) and at 800. The run is unscored either way — ten steps is under the thirty-bar
# floor — so a collapsed recursion would not fail the metrics; it would quietly gut the two
# determinism checks below, because comparing two runs of a model that is returning
# floating-point dust is not a determinism test. 400 is the shortest measured window that
# leaves both models estimating, and the fit cost at that length is immaterial for ten steps.
walk_body: dict[str, Any] = {
    "models": ["egarch", "arma_egarch_xgboost"],
    "walk_forward": {"train_window": 400, "test_window": 40, "step_size": 4},
    "forecast_horizon": 8,
    "write": False,
}
first_run = report_of(run(raw_id, **walk_body))
second_run = report_of(run(raw_id, **walk_body))

check(
    "two identical requests return bit-identical forward paths",
    json.dumps(first_run.get("forward_paths")) == json.dumps(second_run.get("forward_paths"))
    and bool(first_run.get("forward_paths")),
    f"{len(first_run.get('forward_paths') or [])} paths compared",
)
check(
    "two identical requests return bit-identical walk-forward error quantiles",
    json.dumps(first_run.get("error_quantiles")) == json.dumps(second_run.get("error_quantiles"))
    and bool(first_run.get("error_quantiles")),
    json.dumps(first_run.get("error_quantiles"))[:200],
)
check(
    "an unscored run still fits and projects, and says why the metrics are absent",
    all(
        entry.get("mqlike") is None and bool(entry.get("unscored_reason"))
        for entry in first_run.get("scores") or []
    )
    and not first_run.get("comparisons"),
    json.dumps([entry.get("unscored_reason") for entry in first_run.get("scores") or []])[:280],
)
check(
    "with no declared timeframe the forward step is measured from the data",
    close_enough(first_run.get("forward_step_seconds"), 3600.0, 1e-9)
    and "median observed spacing" in str(first_run.get("forward_step_source")),
    f"step={first_run.get('forward_step_seconds')} "
    f"source={first_run.get('forward_step_source')}",
)

# A separate run for the short-protocol message, on the most robust estimator available: the
# assertion is about which *sentence* the report reaches for when a protocol scores too few
# bars, and a step that happened to raise would send it down the other branch and test
# nothing.
short = report_of(
    run(
        source_id,
        models=["garch"],
        distribution="normal",
        walk_forward={"train_window": TRAIN, "test_window": 60, "step_size": 6},
        forecast_horizon=2,
        write=False,
    )
)
check(
    "a protocol that scores too few bars blames the sample rather than the optimiser",
    all(
        "All 10 steps produced a finite forecast" in str(entry.get("unscored_reason"))
        and str(MIN_SCORED_STEPS) in str(entry.get("unscored_reason"))
        for entry in short.get("scores") or []
    ),
    json.dumps([entry.get("unscored_reason") for entry in short.get("scores") or []])[:300],
)

# ---------------------------------------------------------------- conditioning

print("== conditioning on a regime output ==")

conditioned_job = run(
    source_id,
    models=["arma_egarch_xgboost"],
    conditioning_dataset_id=regime_id,
    forecast_horizon=6,
)
conditioned = report_of(conditioned_job)
check(
    "a conditioned run succeeds against a real regime dataset",
    conditioned_job.get("status") == "succeeded",
    f"status={conditioned_job.get('status')} "
    f"{json.dumps(conditioned_job.get('error'))[:200]}",
)
check(
    "the report names the conditioning dataset and the columns it contributed",
    conditioned.get("conditioning_dataset_id") == regime_id
    and conditioned.get("conditioning_dataset_name") == regime_meta.get("name")
    and set(conditioned.get("conditioning_columns") or [])
    >= {"prob_crisis", "prob_trending_up", "confidence"},
    f"id={conditioned.get('conditioning_dataset_id')} "
    f"columns={conditioned.get('conditioning_columns')}",
)
check(
    "the report states that the conditioning columns are lagged before use",
    any("lagged" in warning for warning in conditioned.get("warnings") or []),
    json.dumps(conditioned.get("warnings"))[:240],
)
check(
    "the warm-up bars with no posterior are counted and reported, not quietly filled",
    any("no regime posterior" in warning for warning in conditioned.get("warnings") or []),
    json.dumps(conditioned.get("warnings"))[:240],
)
check(
    "the conditioned run is scored over the same region as the unconditioned one",
    conditioned.get("first_scored_index") == report.get("first_scored_index")
    and conditioned.get("last_scored_index") == report.get("last_scored_index"),
    f"{conditioned.get('first_scored_index')}..{conditioned.get('last_scored_index')} vs "
    f"{report.get('first_scored_index')}..{report.get('last_scored_index')}",
)

status, conditioned_meta = call(
    "GET", f"/datasets/{conditioned_job.get('output_dataset_id')}"
)
check(
    "the conditioned output is saved with a readable provenance",
    status == 200
    and ((conditioned_meta or {}).get("provenance") or {}).get("module_source") == "volatility",
    f"status={status}",
)

# ------------------------------------------------------------------- refusals

print("== refusals ==")

# A request the schema can reject is rejected at the door, with a 422 and no job created.
#
# The *envelope* is asserted, not only the message. Four of this module's request models
# refuse inside a ``model_validator``, and a pydantic refusal is answered by FastAPI itself
# unless something says otherwise — which it does, in ``app/core/errors.py``
# (``DECISIONS §117``). That handler is what stands between a specific refusal and a page
# that shows a generic failure, because ``src/api/client.ts`` reads ``error.code``. Checking
# the message alone would still pass if the handler were removed and the client fell back to
# ``json.dumps(payload)``, which contains every field name the fragments look for.
for label, body, fragment in [
    ("an empty model list is refused", {"models": []}, "model"),
    ("selecting a model twice is refused", {"models": ["garch", "garch"]}, "more than once"),
    ("an asymmetry order of zero for an asymmetric model is refused",
     {"models": ["egarch"], "o": 0}, "asymmetry"),
    ("a blank price source is refused", {"price_source": "  "}, "price_source"),
    ("a train window below the minimum is refused",
     {"walk_forward": {"train_window": MIN_TRAIN_WINDOW - 1}}, "train_window"),
    ("a test window below the minimum is refused",
     {"walk_forward": {"test_window": MIN_SCORED_STEPS - 1}}, "test_window"),
    ("a step size of zero is refused", {"walk_forward": {"step_size": 0}}, "step_size"),
    ("a step larger than the region it walks is refused",
     {"walk_forward": {"test_window": 40, "step_size": 41}}, "step_size"),
    ("a forecast horizon beyond the maximum is refused",
     {"forecast_horizon": 501}, "forecast_horizon"),
    ("a lag order beyond the maximum is refused", {"lags": 101}, "lags"),
]:
    status, payload = forecast(source_id, **body)
    message = rejection_message(status, payload)
    envelope = (payload.get("error") or {}) if isinstance(payload, dict) else {}
    check(
        label,
        status == 422
        and envelope.get("code") == "validation_failure"
        and fragment.lower() in message.lower(),
        f"status={status} code={envelope.get('code')} {message[:140]}",
    )

status, payload = forecast("NOPE")
check(
    "an unknown dataset is refused with a 404 before any job is created",
    status == 404,
    f"status={status} {error_message(payload)[:120]}",
)

# The refusals below need the file read, so they cannot be an HTTP status: the response was
# a 202 before the frame was opened. They arrive on the job record instead, and the check is
# on the *code* as well as the message — an internal error would carry a message too.
status, refused = failed_job(
    source_id, walk_forward={"train_window": 250, "test_window": 100, "step_size": 1}
)
message = job_error(refused)
check(
    "a walk-forward window that cannot fit is refused inside the job, naming both numbers",
    status == 202
    and refused.get("status") == "failed"
    and (refused.get("error") or {}).get("code") == "validation_failure"
    and "350" in message
    and str(source_meta.get("row_count", 0) - 1) in message,
    f"status={status} job={refused.get('status')} {message[:160]}",
)

status, refused = failed_job(
    walk_id,
    price_source="cumulative_volume",
    models=["garch"],
    walk_forward={"train_window": 100, "test_window": 80, "step_size": 20},
    forecast_horizon=2,
)
message = job_error(refused)
check(
    "a column that is not a return series is refused with the ADF numbers",
    status == 202
    and refused.get("status") == "failed"
    and (refused.get("error") or {}).get("code") == "validation_failure"
    and "stationary" in message
    and "Dickey" in message
    and "cumulative_volume" in message,
    f"status={status} job={refused.get('status')} {message[:200]}",
)

status, refused = failed_job(source_id, price_source="nope")
message = job_error(refused)
check(
    "a price source that names no column is refused, naming the column",
    status == 202
    and refused.get("status") == "failed"
    and (refused.get("error") or {}).get("code") == "validation_failure"
    and "'nope'" in message,
    f"status={status} job={refused.get('status')} {message[:160]}",
)

# A refused run must not have written anything. The store is append-only and a stray
# half-written dataset is the artifact equivalent of a failed fit that still reported a
# metric.
status, payload = forecast(source_id, walk_forward={"train_window": 250, "test_window": 100,
                                                   "step_size": 1})
refused_job = poll(str(payload["job_id"])) if status == 202 else {}
check(
    "a refused run leaves no output dataset behind",
    refused_job.get("status") == "failed" and refused_job.get("output_dataset_id") is None,
    f"job={refused_job.get('status')} output={refused_job.get('output_dataset_id')}",
)

# write=false is the preview path: the computation runs in full and the store is untouched.
status, payload = forecast(source_id, models=["garch"], write=False, name="E2E volatility preview")
preview_only = poll(str(payload["job_id"])) if status == 202 else {}
preview_report = report_of(preview_only)
check(
    "the preview-only run computes a full result but stores nothing",
    preview_only.get("status") == "succeeded"
    and dataset_of(preview_only) == {}
    and preview_report.get("scores")
    and preview_only.get("output_dataset_id") is None,
    f"dataset={dataset_of(preview_only)} output={preview_only.get('output_dataset_id')} "
    f"scores={len(preview_report.get('scores') or [])}",
)
check(
    "a run that writes nothing still leaves a readable preview",
    len(preview_of(preview_only).get("tail", [])) > 0
    and tuple(columns_of(preview_of(preview_only))) == VOL_FORECAST_COLUMNS,
    f"{len(preview_of(preview_only).get('tail', []))} tail rows",
)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for item in FAILURES:
        print("  -", item)
    sys.exit(1)
print("all volatility-endpoint checks passed")
