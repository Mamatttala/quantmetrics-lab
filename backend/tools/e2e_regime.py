"""End-to-end exercise of the Part D regime endpoints over HTTP.

Not a unit test: it drives the running server exactly as the browser does, so it catches
contract drift the in-process TestClient cannot see — the 202-then-poll handshake, the
`error` envelope arriving through a job record rather than through a request, and the
provenance a saved run actually leaves on the dataset a user keeps.

Four things make this driver worth its runtime beyond re-checking the unit tests:

**The job handshake is a protocol, not a function call.** ``POST /api/regime/detect`` has to
return *before* the work is done. An in-process test can call the service and await it,
which would pass even if the endpoint had been written to block — the very thing the design
exists to prevent. Only a real HTTP client can observe that the response arrives with no
result in it.

**The dataset has to declare a timeframe.** Module D refuses one that does not, because the
crisis threshold is annualised and there is no factor to annualise by. A CSV upload declares
nothing, so the driver uploads and then *resamples* through the merge endpoint — the only
route to a declared-timeframe frame without a network fetch. That gives both halves of the
guardrail for free: the upload refuses, the resampled frame runs, and the two are asserted
against each other so the refusal cannot quietly become a blanket one.

**The crisis override is checked against a real volatility spike**, not a synthetic
probability vector. The unit tests inject the mask directly; here the spike is in the
prices, so the check covers the observation matrix, the trailing window, the annualisation
factor and the override in one pass. A units error anywhere along that chain moves the
realized volatility away from the threshold and the spike stops being detected — which is
exactly the failure a unit test with an injected mask cannot see.

**The evidence is read back off disk, not out of the preview.** A run's preview is five head
rows and five tail rows; the first five are inside the warm-up and carry no label at all, and
the last five sit fifty bars past the volatility spike. Asserting the crisis behaviour
against a ten-row window would pass by accident on a frame that never detected crisis. Every
probability check below therefore pages the *saved output dataset*, which is also the only
way to prove the artifact a user keeps is the one that was checked.

Run it against a live server (``python tools/serve.py`` in another shell):

    python tools/e2e_regime.py
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

#: The exact output schema Part D specifies. Asserted as an ordered tuple: a consumer
#: reading these by position is a thing that will eventually happen.
#:
#: The time column is `open_time` — the instant the bar was observed, which is the column
#: the engine actually reads (`quality.find_time_column`). It was named `close_time` while
#: holding `open_time` values, which is the one mistake a timestamp column must not make.
OUTPUT_COLUMNS = (
    "open_time",
    "dominant_state",
    "prob_trending_up",
    "prob_trending_down",
    "prob_ranging",
    "prob_crisis",
    "confidence",
)

PROBABILITY_COLUMNS = OUTPUT_COLUMNS[2:6]

STATES = {"trending_up", "trending_down", "ranging", "crisis", "transition"}

#: The four states that name a regime, as opposed to Transition, which names the absence of
#: a decision. Only these have a committed state for `confidence` to be the probability of.
DECIDED_STATES = {"trending_up", "trending_down", "ranging", "crisis"}

#: Walk-forward settings for the driver. Much shorter than the shipped defaults, because
#: the driver's frame is a couple of hundred bars and the defaults would label none of it.
#: The schedule is still a genuine expanding window — the property under test is the shape
#: of the walk, not its length.
MIN_TRAIN = 60
REFIT_EVERY = 60
VOL_WINDOW = 24

#: The crisis probability the override writes in. Repeated from the request default rather
#: than imported, because the check below is precisely that the server honours the number
#: the request asked for — importing the constant would let a changed default pass.
OVERRIDE_PROBABILITY = 0.90


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
        with urllib.request.urlopen(request, timeout=180) as response:
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


def upload(name: str, body: str) -> str:
    """POST a CSV as multipart/form-data, built by hand to keep the driver stdlib-only."""
    boundary = "----quantmetrics-e2e-regime-boundary"
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

    Deliberately not ``random``: the driver has to be reproducible across Python versions
    and machines, because "the same request gives the same labels" is one of the things it
    asserts, and a generator whose stream changes between releases would make that check
    fail for a reason that has nothing to do with the module.
    """
    state = (1103515245 * state + 12345) % (2**31)
    u1 = max(state / (2**31), 1e-12)
    state = (1103515245 * state + 12345) % (2**31)
    u2 = state / (2**31)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2), state


def hourly_csv(
    count: int,
    *,
    seed: int = 20240101,
    vol: float = 0.0015,
    spike: tuple[int, int] | None = None,
) -> str:
    """A deterministic hourly OHLCV frame, with an optional volatility spike.

    ``vol`` is the per-bar log-return scale, chosen so the annualised realized volatility
    sits near 0.14 — comfortably under the 0.35 crisis threshold, so the run is not a crisis
    from end to end and the override check means something. Inside ``spike`` the scale is
    multiplied by 12, which on a 4h resample annualises to roughly 0.9 and breaches by a wide
    margin: a threshold that is only just crossed would make the check depend on the exact
    random draw.
    """
    state = seed
    price = 100.0
    rows = ["open_time,open,high,low,close,volume"]
    for index in range(count):
        z, state = _gaussian(state)
        scale = vol * 12.0 if spike is not None and spike[0] <= index < spike[1] else vol
        open_price = price
        price *= math.exp(scale * z)
        close = price
        high = max(open_price, close) * 1.0008
        low = min(open_price, close) * 0.9992
        stamp = datetime(2023, 1, 1) + timedelta(hours=index)
        volume = 1000.0 + 100.0 * math.sin(index / 7.0)
        rows.append(
            f"{stamp.strftime('%Y-%m-%dT%H:%M:%SZ')},{open_price:.4f},{high:.4f},"
            f"{low:.4f},{close:.4f},{volume:.4f}"
        )
    return "\n".join(rows) + "\n"


#: 800 hourly bars, resampled to 200 four-hour bars. The spike sits at hourly bars 560-590,
#: i.e. four-hour bars 140-148, so the crisis lands two thirds of the way through with enough
#: room after it for the trailing volatility to decay and the state to leave.
HOURLY_CSV = hourly_csv(800, spike=(560, 590))


# ------------------------------------------------------------------- the harness


def detect(dataset_id: str, **overrides: Any) -> tuple[int, Any]:
    body: dict[str, Any] = {
        "dataset_id": dataset_id,
        "model_type": "hmm",
        "n_states": 4,
        "min_state_duration": 5,
        "crisis_threshold": 0.35,
        "switching_cost_penalty": 0.005,
        "min_train_bars": MIN_TRAIN,
        "refit_every": REFIT_EVERY,
        "volatility_window": VOL_WINDOW,
        "write": True,
    }
    body.update(overrides)
    return call("POST", "/regime/detect", body)


def poll(job_id: str, *, timeout: float = 300.0) -> dict[str, Any]:
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
    status, accepted = detect(dataset_id, **overrides)
    if status != 202 or not isinstance(accepted, dict):
        print(f"detect was not accepted: {status} {json.dumps(accepted)[:300]}")
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


def net_value_add_of(job: dict[str, Any]) -> dict[str, Any]:
    return report_of(job).get("net_value_add") or {}


def page_rows(dataset_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
    """Every row of a saved dataset, in one page.

    One page is enough at a couple of hundred rows and the caller asserts the count, so a
    frame that outgrew the page cap would be a visible failure rather than a check that
    quietly examined only the first page.
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


def time_column_of(meta: dict[str, Any]) -> str | None:
    """The dataset's timeline column, found by name — the same heuristic the backend uses."""
    for column in meta.get("columns", []):
        if "time" in str(column["name"]).lower():
            return str(column["name"])
    return None


# --------------------------------------------------------------------- seeding

print("== seeding ==")
raw_id = upload("e2e_regime_hourly.csv", HOURLY_CSV)
print(f"  uploaded={raw_id}")

# A declared-timeframe frame, produced by the merge endpoint's resample. Module D refuses a
# dataset that declares no timeframe, so without this the driver could only ever exercise
# the refusals.
status, merged = call(
    "POST",
    "/datasets/merge",
    {
        "base_dataset_id": raw_id,
        "name": "E2E regime 4h base",
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
source_checksum = source_meta.get("checksum")
check("the source dataset reports a checksum", bool(source_checksum), f"status={status}")

# --------------------------------------------------------------------- options

print("== options ==")

status, raw_options = call("GET", f"/regime/options?dataset_id={raw_id}")
check("options are served for an uploaded dataset", status == 200, f"status={status}")
raw_options = raw_options or {}
check(
    "an uploaded dataset declares no timeframe, so annualisation is refused",
    raw_options.get("annualisation", {}).get("available") is False,
    json.dumps(raw_options.get("annualisation"))[:160],
)
check(
    "the refusal is an error-level warning, not a caution",
    [warning["code"] for warning in raw_options.get("warnings", [])
     if warning["level"] == "error"] == ["no_timeframe"],
    json.dumps(raw_options.get("warnings"))[:200],
)
check(
    "the reason names annualisation rather than a missing column",
    "annualised" in str(raw_options.get("annualisation", {}).get("reason", "")),
    str(raw_options.get("annualisation", {}).get("reason"))[:120],
)

status, options = call("GET", f"/regime/options?dataset_id={source_id}")
check("options are served for the source dataset", status == 200, f"status={status}")
options = options or {}

annualisation = options.get("annualisation", {})
# The factor is checked against the asset class the server *reports* rather than a guess at
# it: an uploaded CSV carries a placeholder asset class (DECISIONS §25), and the property
# under test here is the arithmetic — bars per day times days per year — not which family the
# placeholder landed in.
asset_class = str(options.get("asset_class", ""))
days = 365 if asset_class.startswith("crypto") else 252
check(
    "the resampled frame annualises, by the factor its timeframe and asset class imply",
    annualisation.get("available") is True
    and annualisation.get("periods_per_year") == 6 * days,
    f"asset_class={asset_class} periods_per_year={annualisation.get('periods_per_year')} "
    f"(4h = 6 bars/day x {days} days)",
)
check(
    "the timeframe is reported so the factor can be checked by hand",
    options.get("timeframe") == "4h",
    str(options.get("timeframe")),
)

feature_ids = [entry["id"] for entry in options.get("features", [])]
check(
    "every observation the specification names is offered or explained",
    set(feature_ids) == {"log_return", "realized_volatility", "volume_change", "adx", "rsi"},
    str(feature_ids),
)
by_id = {entry["id"]: entry for entry in options.get("features", [])}
check(
    "the computed observations are available on a grid with a close and a volume",
    all(by_id[name]["available"] for name in ("log_return", "realized_volatility",
                                              "volume_change")),
    json.dumps({name: by_id[name]["available"] for name in by_id}),
)
check(
    "adx is carried, never computed, and unavailable without the column",
    by_id["adx"]["available"] is False
    and by_id["adx"]["kind"] == "carried"
    and "adx" in str(by_id["adx"]["reason"]),
    json.dumps(by_id["adx"])[:180],
)

models = {entry["id"]: entry for entry in options.get("models", [])}
check(
    "the hybrid model is refused with a reason naming the missing indicator",
    models["hybrid"]["available"] is False and "adx" in str(models["hybrid"]["reason"]),
    json.dumps(models["hybrid"])[:180],
)
check(
    "the three models that need no carried column are available",
    all(models[name]["available"] for name in ("hmm", "gmm", "rules")),
    json.dumps({name: models[name]["available"] for name in models}),
)

status, _ = call("GET", "/regime/options?dataset_id=NOPE")
check("options for an unknown dataset is a 404", status == 404, f"status={status}")

# ------------------------------------------------------------------ the handshake

print("== the job handshake ==")

status, accepted = detect(source_id)
check(
    "detect is accepted with 202, not answered with 200",
    status == 202,
    f"status={status} body={json.dumps(accepted)[:140]}",
)
accepted = accepted or {}
check(
    "the handle carries a job id and a poll path, and no result",
    bool(accepted.get("job_id"))
    and accepted.get("poll") == f"/api/jobs/{accepted.get('job_id')}"
    and "report" not in accepted
    and "result" not in accepted,
    json.dumps(accepted)[:200],
)
first = poll(str(accepted["job_id"]))
check(
    "the job reaches a terminal state by polling",
    first.get("status") == "succeeded",
    json.dumps(first.get("error"))[:200] if first.get("error") else str(first.get("status")),
)

status, unknown = call("GET", "/jobs/does-not-exist")
check(
    "a job that never existed is a 404, distinguishable from one still running",
    status == 404 and ((unknown or {}).get("error") or {}).get("code") == "job_not_found",
    f"status={status} {json.dumps(unknown)[:140]}",
)

status, listed = call("GET", "/jobs?limit=50")
listed_ids = [row["job_id"] for row in (listed or [])] if status == 200 else []
check("the run appears in the job registry", str(accepted["job_id"]) in listed_ids,
      f"status={status} jobs={len(listed_ids)}")

# --------------------------------------------------------------------- the output

print("== the output ==")

report = report_of(first)
preview = preview_of(first)

check(
    "the output carries the specified schema, in order",
    tuple(columns_of(preview)) == OUTPUT_COLUMNS,
    str(columns_of(preview)),
)
check(
    "some bars were labelled",
    report.get("labelled_rows", 0) > 0,
    f"labelled={report.get('labelled_rows')} of {report.get('rows')}",
)
check(
    "every row is accounted for across the states plus the warm-up prefix",
    sum(report.get("state_counts", {}).values()) == report.get("rows"),
    json.dumps(report.get("state_counts"))[:200],
)
check(
    "the two unlabelled gaps are reported separately, not conflated",
    report.get("warmup_bars") == VOL_WINDOW
    and report.get("unlabelled_bars") == report.get("warmup_bars") + MIN_TRAIN
    and report.get("labelled_rows") == report.get("rows") - report.get("unlabelled_bars"),
    f"warmup={report.get('warmup_bars')} unlabelled={report.get('unlabelled_bars')} "
    f"labelled={report.get('labelled_rows')} of {report.get('rows')}",
)

# The walk-forward schedule, which DECISIONS §74 pins to bars from the start of the series
# rather than to a fraction of its length. The two give the same refit points on a fixed
# frame and different ones on any truncation, which is what the causality test in the unit
# suite is built on; here the schedule is read back and checked against the request. The
# first anchor sits one observation warm-up in, because the model cannot train on rows that
# have no feature vector — anchoring to raw bar 60 would train on NaN.
refits = report.get("refit_points") or []
anchors = [point.get("trained_through_bar") for point in refits]
check(
    "the refit schedule is anchored in bars from the start of the series",
    len(anchors) >= 2
    and anchors[0] == VOL_WINDOW + MIN_TRAIN
    and all(b - a == REFIT_EVERY for a, b in zip(anchors, anchors[1:], strict=False)),
    f"refit_count={report.get('refit_count')} anchors={anchors[:4]} "
    f"(expected {VOL_WINDOW}+{MIN_TRAIN}, then every {REFIT_EVERY})",
)
check(
    "the schedule stops before running off the end of the frame",
    anchors == [] or anchors[-1] <= report.get("rows", 0),
    f"last anchor={anchors[-1] if anchors else None} rows={report.get('rows')}",
)
check(
    "each refit reports the diagnostics part D asks for",
    bool(refits)
    and all(
        key in refits[0]
        for key in ("bic", "aic", "n_parameters", "log_likelihood", "training_bars",
                    "labels_bars", "state_naming")
    ),
    str(sorted(refits[0].keys()))[:220] if refits else "no refit points",
)

# The saved artifact, read back off disk. Everything below is asserted against these rows
# rather than the preview, for the reason in the module docstring: a ten-row head/tail window
# contains no crisis bar at all, so a preview-based crisis check passes on a run that never
# detected one.
child_meta = dataset_of(first)
child_id = child_meta.get("dataset_id")
if not child_id:
    print("the run wrote no dataset; the output checks cannot run")
    sys.exit(1)
saved = page_rows(str(child_id))
check(
    "the saved frame has one row per input bar",
    len(saved) == report.get("rows"),
    f"saved={len(saved)} report_rows={report.get('rows')}",
)

labelled = [row for row in saved if row.get("dominant_state")]
check(
    "the labelled rows are the ones the report claims",
    len(labelled) == report.get("labelled_rows"),
    f"saved={len(labelled)} reported={report.get('labelled_rows')}",
)
check(
    "the unlabelled prefix is present in the frame, not dropped",
    sum(1 for row in saved if not row.get("dominant_state")) == report.get("unlabelled_bars"),
    f"{sum(1 for row in saved if not row.get('dominant_state'))} of {len(saved)} unlabelled, "
    f"report says {report.get('unlabelled_bars')}",
)
check(
    "the state counts agree with the report about how many rows carry no label",
    report.get("state_counts", {}).get("warmup") == report.get("unlabelled_bars"),
    f"state_counts.warmup={report.get('state_counts', {}).get('warmup')} "
    f"unlabelled_bars={report.get('unlabelled_bars')}",
)

timeline = [as_time(row["open_time"]) for row in saved]
check(
    "the timeline is strictly increasing",
    all(later > earlier for earlier, later in zip(timeline, timeline[1:], strict=False)),
    f"{len(timeline)} bars",
)

source_time_column = time_column_of(source_meta)
if source_time_column:
    source_timeline = [as_time(row[source_time_column]) for row in page_rows(source_id)]
    mismatch = next(
        (i for i, (a, b) in enumerate(zip(source_timeline, timeline, strict=False)) if a != b),
        None,
    )
    check(
        "the output timeline is the input timeline, bar for bar",
        len(source_timeline) == len(timeline) and mismatch is None,
        f"{len(source_timeline)} vs {len(timeline)} bars; first mismatch at {mismatch}",
    )
else:
    check("the source exposes a time column to compare against", False, "none found")

soft = [
    [float(row[name]) for name in PROBABILITY_COLUMNS]
    for row in labelled
    if all(isinstance(row.get(name), (int, float)) for name in PROBABILITY_COLUMNS)
]
check(
    "the labels are soft: every probability is non-negative and each row sums to one",
    len(soft) == len(labelled)
    and all(value >= 0.0 for row in soft for value in row)
    and all(abs(sum(row) - 1.0) < 1e-6 for row in soft),
    f"{len(soft)} of {len(labelled)} rows checked, worst sum error "
    f"{max((abs(sum(row) - 1.0) for row in soft), default=float('nan')):.2e}",
)
spread = sum(1 for row in soft if any(1e-6 < value < 1 - 1e-6 for value in row))
check(
    "no bar carries a hard 0/1 vector — the model reports genuine uncertainty somewhere",
    spread > 0,
    f"{spread} of {len(soft)} bars carry a probability strictly between 0 and 1",
)
check(
    "the dominant state is always one of the five named states",
    all(row["dominant_state"] in STATES for row in labelled),
    str(sorted({str(row["dominant_state"]) for row in labelled})),
)
check(
    "confidence is the probability of the state the bar was committed to",
    all(
        abs(float(row["confidence"]) - float(row[f"prob_{row['dominant_state']}"])) < 1e-9
        for row in labelled
        if row["dominant_state"] in DECIDED_STATES
    ),
    "checked on every non-Transition labelled bar",
)
transitions = [row for row in labelled if row["dominant_state"] == "transition"]
check(
    "a Transition bar is one no state holds a clear majority of",
    all(
        max(float(row[name]) for name in PROBABILITY_COLUMNS) < 0.55 + 1e-9
        for row in transitions
    ),
    f"{len(transitions)} transition bars",
)

# ---------------------------------------------------------------- crisis override

print("== crisis override ==")

check(
    "the volatility spike fired the crisis override",
    report.get("crisis_override_bars", 0) > 0,
    f"override_bars={report.get('crisis_override_bars')} "
    f"threshold={report.get('crisis_threshold')} "
    f"periods_per_year={report.get('periods_per_year')}",
)

crisis_rows = [row for row in labelled if row["dominant_state"] == "crisis"]
check(
    "crisis appears in the output",
    len(crisis_rows) > 0,
    f"{len(crisis_rows)} crisis-labelled bars of {len(labelled)} labelled",
)
check(
    "the crisis count in the output matches the count the report claims",
    len(crisis_rows) == report.get("state_counts", {}).get("crisis", 0),
    f"rows={len(crisis_rows)} report={report.get('state_counts', {}).get('crisis')}",
)

# The override writes the configured probability *exactly*, and it is applied inside
# `commit_labels` before the label is read off that same array — so a bar carrying exactly
# that number is a bar the override fired on, and it must be labelled crisis. Bars the model
# reached crisis on by itself are deliberately not asserted here: anti-thrashing is allowed to
# hold those, which is the point of the rule. Matching the exact value is what separates the
# two cases without re-deriving the volatility series in the driver.
overridden = [
    row for row in labelled
    if abs(float(row.get("prob_crisis", 0.0)) - OVERRIDE_PROBABILITY) < 1e-9
]
check(
    "every bar carrying the override probability is labelled crisis",
    len(overridden) > 0 and all(row["dominant_state"] == "crisis" for row in overridden),
    f"{len(overridden)} bars at exactly {OVERRIDE_PROBABILITY}; states="
    f"{sorted({str(row['dominant_state']) for row in overridden})}",
)
check(
    "crisis is the argmax of every bar the override fired on",
    len(overridden) > 0
    and all(
        max(PROBABILITY_COLUMNS, key=lambda name: float(row[name])) == "prob_crisis"
        for row in overridden
    ),
    f"{len(overridden)} bars checked",
)
check(
    "the override count in the report matches the bars actually carrying it",
    len(overridden) == report.get("crisis_override_bars"),
    f"rows={len(overridden)} report={report.get('crisis_override_bars')}",
)
check(
    "the override bypasses anti-thrashing, so it holds at least as many bars as it fired on",
    len(crisis_rows) >= len(overridden),
    f"held={len(crisis_rows)} fired={len(overridden)}",
)
check(
    "the state leaves crisis once the trailing volatility decays",
    labelled[-1]["dominant_state"] != "crisis" and len(crisis_rows) < len(labelled),
    f"last bar is {labelled[-1]['dominant_state']}; crisis on {len(crisis_rows)} of "
    f"{len(labelled)}",
)

# ------------------------------------------------------------------ anti-thrashing

print("== anti-thrashing and the switching cost ==")

# Run on the fitted model, not the heuristic. `min_state_duration` can only bite where a
# model produces competing states, and on this frame — realized volatility around 0.14
# against a 0.35 crisis threshold — the heuristic scales every state's probability down by
# the same factor until nothing clears the commit threshold, so its labels are decided by
# the crisis override and both settings agree by construction. Comparing them there would
# test nothing. The heuristic's behaviour on such a frame is asserted on its own below.
impatient = report_of(run(source_id, min_state_duration=1))
patient = report_of(run(source_id, min_state_duration=10))

check(
    "a longer minimum duration never increases the number of switches",
    patient.get("switch_count", 0) <= impatient.get("switch_count", 0),
    f"min=1 -> {impatient.get('switch_count')} switches, "
    f"min=10 -> {patient.get('switch_count')}",
)
check(
    "the rule actually bites — the two settings do not produce the same series",
    patient.get("switch_count", 0) < impatient.get("switch_count", 0),
    f"min=1 -> {impatient.get('switch_count')}, min=10 -> {patient.get('switch_count')}",
)

nva = patient.get("net_value_add", {})
check(
    "the strategy took a position, so Net Value-Add is about the model",
    nva.get("bars_long", 0) > 0 and nva.get("strategy_log_return") not in (None, 0.0),
    f"bars_long={nva.get('bars_long')} strategy={nva.get('strategy_log_return')}",
)
check(
    "Net Value-Add is reported with its definition and an in-sample caveat",
    "net_value_add" in nva
    and "long in trending_up" in str(nva.get("definition", ""))
    and nva.get("in_sample") is True,
    json.dumps(nva)[:200],
)
check(
    "Net Value-Add is the return improvement less the switching cost",
    nva.get("return_improvement") is not None
    and nva.get("switching_cost") is not None
    and abs(
        float(nva["net_value_add"])
        - (float(nva["return_improvement"]) - float(nva["switching_cost"]))
    )
    < 1e-9,
    json.dumps(nva)[:240],
)
check(
    "the switching cost is charged on position changes, not on state changes",
    abs(
        float(nva.get("switching_cost", 0.0))
        - int(nva.get("position_changes", 0)) * float(nva.get("switching_cost_penalty", 0.0))
    )
    < 1e-9
    and nva.get("position_changes", 0) <= patient.get("switch_count", 0),
    f"position_changes={nva.get('position_changes')} "
    f"state switches={patient.get('switch_count')} cost={nva.get('switching_cost')}",
)

free = net_value_add_of(run(source_id, min_state_duration=10, switching_cost_penalty=0.0))
check(
    "a zero penalty leaves the raw improvement as the Net Value-Add",
    float(free.get("switching_cost", -1.0)) == 0.0
    and abs(
        float(free.get("net_value_add", 0.0)) - float(free.get("return_improvement", 0.0))
    )
    < 1e-12,
    json.dumps(free)[:200],
)
check(
    "raising the penalty lowers the Net Value-Add without moving the labels",
    float(free.get("net_value_add", 0.0)) > float(nva.get("net_value_add", 0.0))
    and free.get("return_improvement") == nva.get("return_improvement")
    and free.get("bars_long") == nva.get("bars_long"),
    f"penalty 0 -> {free.get('net_value_add')}, penalty 0.005 -> {nva.get('net_value_add')}",
)

# The heuristic path, on a frame where it is deliberately indecisive. Every state's
# probability is multiplied by `quiet = 1 - volatility / crisis_threshold` — 0.6 here — so
# no calm bar can reach the 0.60 commit threshold and the only committed states come from
# the crisis override. The engine reports this through `bars_long`, and the check exists so
# that a change making the heuristic decisive on such a frame is a visible change rather
# than a quietly different Net Value-Add.
heuristic = report_of(run(source_id, model_type="rules"))
heuristic_nva = heuristic.get("net_value_add", {})
check(
    "the heuristic commits only through the crisis override at mid-range volatility",
    heuristic_nva.get("bars_long") == 0
    and heuristic.get("state_counts", {}).get("trending_up", 0) == 0
    and heuristic.get("state_counts", {}).get("crisis", 0) > 0
    and heuristic.get("state_counts", {}).get("transition", 0) > 0,
    f"state_counts={json.dumps(heuristic.get('state_counts'))}",
)
check(
    "a strategy that never went long reports an improvement of exactly minus the benchmark",
    heuristic_nva.get("strategy_log_return") == 0.0
    and heuristic_nva.get("position_changes") == 0
    and abs(
        float(heuristic_nva.get("return_improvement", 0.0))
        + float(heuristic_nva.get("benchmark_log_return", 0.0))
    )
    < 1e-12,
    json.dumps(heuristic_nva)[:220],
)

# ------------------------------------------------------------------ determinism

print("== determinism ==")

# Repeating the *fitted* request, not the heuristic one. The heuristic path has no
# parameters to estimate, so it is deterministic by construction and repeating it proves
# nothing; the HMM runs Baum-Welch from an initialisation that has to be deterministic for
# the labels to be reproducible, and that is the property worth asserting. EM finds a local
# optimum, so a random start — or a fit that stops on a floating-point tie — would show up
# here as a different switch count on the same input.
again = report_of(run(source_id, min_state_duration=10))
check(
    "the same request produces the same labels",
    again.get("switch_count") == patient.get("switch_count")
    and again.get("state_counts") == patient.get("state_counts")
    and again.get("transition_matrix") == patient.get("transition_matrix")
    and again.get("crisis_override_bars") == patient.get("crisis_override_bars"),
    f"switches {patient.get('switch_count')} vs {again.get('switch_count')}; "
    f"counts {json.dumps(again.get('state_counts'))}",
)
check(
    "the same request converges to the same fit, not merely the same labels",
    [point.get("log_likelihood") for point in again.get("refit_points", [])]
    == [point.get("log_likelihood") for point in patient.get("refit_points", [])]
    and [point.get("n_iter") for point in again.get("refit_points", [])]
    == [point.get("n_iter") for point in patient.get("refit_points", [])],
    f"log-likelihoods "
    f"{[point.get('log_likelihood') for point in again.get('refit_points', [])]}",
)
check(
    "the same request produces the same scored evaluation",
    again.get("net_value_add") == patient.get("net_value_add"),
    json.dumps(again.get("net_value_add"))[:160],
)

# --------------------------------------------------------------------- lineage

print("== lineage and immutability ==")

status, source_after = call("GET", f"/datasets/{source_id}")
check(
    "the source dataset is byte-identical after every run against it",
    (source_after or {}).get("checksum") == source_checksum,
    f"{source_checksum} -> {(source_after or {}).get('checksum')}",
)

check(
    "the output lands in the outputs category, parented to the source",
    child_meta.get("category") == "outputs"
    and child_meta.get("parent_dataset_id") == source_id,
    json.dumps({k: child_meta.get(k) for k in ("category", "parent_dataset_id")}),
)
check(
    "the job names the dataset it produced, and the one it ran against",
    first.get("output_dataset_id") == child_id and first.get("dataset_id") == source_id,
    f"output={first.get('output_dataset_id')} input={first.get('dataset_id')}",
)

# The lineage *row* — the whole request plus the diagnostics — is asserted in-process by
# `tests/test_regime_api.py`, which reads the SQLite table directly. There is no HTTP
# endpoint that exposes it, so what is checked here is everything the lineage is surfaced
# through: the category, the parent link, and the provenance a user can read off the dataset.
provenance = child_meta.get("provenance") or {}
check(
    "the output's provenance names the module and the thresholds that produced it",
    provenance.get("module_source") == "regime"
    and provenance.get("regime_model") == "hmm"
    and provenance.get("crisis_threshold") == "0.35",
    json.dumps(provenance)[:240],
)
check(
    "the provenance records the Net Value-Add, so a saved frame can be judged later",
    str(provenance.get("net_value_add")) == str(report["net_value_add"]["net_value_add"]),
    f"provenance={provenance.get('net_value_add')} "
    f"report={report['net_value_add']['net_value_add']}",
)

status, child_frame = call("GET", f"/datasets/{child_id}")
child_columns = [str(column["name"]) for column in (child_frame or {}).get("columns", [])]
check(
    "the stored dataset carries the specified schema",
    tuple(child_columns) == OUTPUT_COLUMNS,
    str(child_columns),
)
check(
    "the stored dataset has the same number of rows as its source",
    (child_frame or {}).get("row_count") == source_meta.get("row_count"),
    f"{(child_frame or {}).get('row_count')} vs {source_meta.get('row_count')}",
)

# -------------------------------------------------------------------- refusals

print("== refusals ==")

status, payload = detect(source_id, model_type="hybrid", features=["log_return"])
failed = poll(str(payload["job_id"])) if status == 202 else {}
check(
    "a hybrid run without a carried indicator fails inside the job, with the reason",
    status == 202
    and failed.get("status") == "failed"
    and "indicator" in str((failed.get("error") or {}).get("message", "")),
    f"status={status} job={failed.get('status')} {json.dumps(failed.get('error'))[:180]}",
)

status, payload = detect(raw_id)
failed = poll(str(payload["job_id"])) if status == 202 else {}
check(
    "a dataset with no declared timeframe is refused, naming annualisation",
    status == 202
    and failed.get("status") == "failed"
    and "annualis" in str((failed.get("error") or {}).get("message", "")),
    f"status={status} job={failed.get('status')} {json.dumps(failed.get('error'))[:180]}",
)

for label, body, fragment in [
    ("n_states outside 2..5 is refused", {"n_states": 9}, "n_states"),
    ("a crisis threshold of zero is refused", {"crisis_threshold": 0.0}, "crisis_threshold"),
    ("a negative switching cost is refused", {"switching_cost_penalty": -1.0},
     "switching_cost_penalty"),
    ("an empty feature list is refused", {"features": []}, "feature"),
    ("a duplicated feature is refused", {"features": ["log_return", "log_return"]},
     "more than once"),
    ("a commit confidence below the transition confidence is refused",
     {"transition_confidence": 0.7, "commit_confidence": 0.6}, "confidence"),
]:
    status, payload = detect(source_id, **body)
    message = error_message(payload) or json.dumps(payload)[:160]
    check(
        label,
        status == 422 and fragment.lower() in message.lower(),
        f"status={status} {message[:140]}",
    )

status, payload = detect("NOPE")
check(
    "an unknown dataset is refused with a 404 before any job is created",
    status == 404,
    f"status={status} {error_message(payload)[:120]}",
)

status, payload = detect(source_id, name="E2E regime names this", write=False)
check("write=false is accepted, so a preview is available without saving", status == 202,
      f"status={status}")
preview_only = poll(str(payload["job_id"])) if status == 202 else {}
preview_report = report_of(preview_only)
check(
    "the preview-only run computes a full result but stores nothing",
    preview_only.get("status") == "succeeded"
    and dataset_of(preview_only) == {}
    and preview_report.get("labelled_rows", 0) > 0
    and preview_only.get("output_dataset_id") is None,
    f"dataset={dataset_of(preview_only)} labelled={preview_report.get('labelled_rows')} "
    f"output={preview_only.get('output_dataset_id')}",
)
check(
    "a run that writes nothing still leaves a readable preview",
    len(preview_of(preview_only).get("tail", [])) > 0,
    f"{len(preview_of(preview_only).get('tail', []))} tail rows",
)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for item in FAILURES:
        print("  -", item)
    sys.exit(1)
print("all regime-endpoint checks passed")
