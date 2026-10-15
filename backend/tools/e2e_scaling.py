"""End-to-end exercise of the C7 scaling endpoints over HTTP.

Not a unit test. ``tests/test_scaling.py`` and the guard tests in
``tests/test_leakage_guards.py`` assert the arithmetic in process; this driver asserts the
things that only exist once a real request has been served, a real Parquet file written, and
a real dataset read back.

**The leak is checked against the stored artifact.** The frame is built so its first 70% sits
at one level and the rest at another, a 70/15/15 plan is made, and a standard scaler is fitted
on the training span. If the fit had seen the whole series the training rows would still be
centred at zero; because it did not, they are, and the test rows are not. Both numbers are
read out of the saved dataset rather than the preview — a ten-row head/tail window contains
no test row at all on this frame.

**Two frames that differ only after the cut produce identical parameters.** That is the
property in its strongest form: not "the test rows do not influence the *obvious* statistic"
but "no byte of the test span reaches the fit at all". The driver uploads a second frame
lifted by 500 from row 280 onward, scales it the same way, and requires the fitted center and
scale to be exactly equal and the first 280 scaled rows to be identical.

**The rolling z-score is checked by truncation over HTTP.** The same request is run against
the full 400-row frame and against its own first 300 rows, and the shared prefix of the two
stored columns must be identical. A centred window, or one that used a full-series statistic,
would differ there.

**The refusals are checked in the two places they occur.** A request the model layer rejects
never becomes a job (422, no job id); a request that can only be rejected after the plan file
is read — a plan belonging to another dataset, a walk-forward plan with no fold — becomes a
job that *fails*, and the reason arrives through the job record because there is no request
left to attach a status to.

Run it against a live server (``python tools/serve.py`` in another shell):

    python tools/e2e_scaling.py
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

ROWS = 400
#: floor(400 x 0.70) — where a 70/15/15 plan puts the train/validation cut, and therefore
#: where the frame's level changes so that the training span is a clean population.
TEST_FROM = 280
WINDOW = 60

#: Columns that must never be offered as scalable, whatever their dtype. `open_time` is a
#: string here, but the rule is not about dtype: a timestamp divided by a standard deviation
#: is a number with no meaning, and it is the axis every downstream module joins on.
NEVER_SCALED = ("open_time", "close_time", "timestamp")

PARAMS_COLUMNS = (
    "feature",
    "scaler_type",
    "center_",
    "scale_",
    "constant",
    "fitted_rows",
    "formula",
)


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
        with urllib.request.urlopen(request, timeout=300) as response:
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


def message_of(payload: Any) -> str:
    """The reason a request was refused, from either error shape.

    A refusal this application raises carries ``error.message``. A pydantic constraint
    violation — ``window: 1`` — is rejected by FastAPI before any handler runs and carries
    ``detail[].loc`` instead. Reading only the first shape would make every field-level check
    below compare an empty string and fail on a server that refused the request correctly.
    """
    message = str(((payload or {}).get("error") or {}).get("message", ""))
    return message or json.dumps(payload)[:200]


def upload(name: str, body: str) -> str:
    """POST a CSV as multipart/form-data, built by hand to keep the driver stdlib-only."""
    boundary = "----quantmetrics-e2e-scaling-boundary"
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


def hourly_csv(count: int, *, step_at: int, tail_lift: float = 0.0) -> str:
    """A deterministic hourly OHLCV frame whose level steps at ``step_at``.

    The generator is index-based and closed-form, so ``hourly_csv(300, ...)`` is byte for byte
    the first 300 rows of ``hourly_csv(400, ...)`` with the same arguments. The causality
    check below depends on exactly that: it compares a full-series run against a truncated one
    and requires the shared prefix to agree, which is only a meaningful comparison if the
    inputs agree there too.

    ``tail_lift`` raises the level from ``step_at`` onward without touching a row before it —
    the second frame in the leakage check, identical to the first inside the training span.
    """
    rows = ["open_time,open,high,low,close,volume"]
    for index in range(count):
        after = index >= step_at
        level = 100.0 + (100.0 if after else 0.0) + (tail_lift if after else 0.0)
        open_price = level + 10.0 * math.sin(index / 23.0) + 0.01 * index
        close = level + 10.0 * math.sin((index + 1) / 23.0) + 0.01 * (index + 1)
        high = max(open_price, close) + 0.5
        low = min(open_price, close) - 0.5
        volume = 1000.0 + 50.0 * math.cos(index / 11.0)
        stamp = datetime(2023, 1, 1) + timedelta(hours=index)
        rows.append(
            f"{stamp.strftime('%Y-%m-%dT%H:%M:%SZ')},{open_price:.4f},{high:.4f},"
            f"{low:.4f},{close:.4f},{volume:.4f}"
        )
    return "\n".join(rows) + "\n"


def page_rows(dataset_id: str, *, limit: int = 2000) -> list[dict[str, Any]]:
    status, body = call("GET", f"/datasets/{dataset_id}/page?offset=0&limit={limit}")
    if status != 200 or not isinstance(body, dict):
        print(f"paging {dataset_id} failed: {status} {json.dumps(body)[:200]}")
        sys.exit(1)
    if body.get("total", 0) > len(body.get("rows", [])):
        print(f"{dataset_id} has {body['total']} rows — more than one page; widen the driver")
        sys.exit(1)
    return list(body.get("rows", []))


def page_row(dataset_id: str, offset: int) -> dict[str, Any]:
    """One row of a stored dataset, by position.

    Separate from ``page_rows`` because that helper refuses a partial page — its guard is
    there to catch a driver that silently compared the first thousand rows of a longer
    frame, and it would fire on a caller that asked for a single row on purpose.
    """
    status, body = call("GET", f"/datasets/{dataset_id}/page?offset={offset}&limit=1")
    rows = list((body or {}).get("rows") or []) if status == 200 else []
    if not rows:
        print(f"no row {offset} in {dataset_id}: {status} {json.dumps(body)[:200]}")
        sys.exit(1)
    return dict(rows[0])


def as_time(value: Any) -> datetime:
    """Parse an ISO stamp, tolerating both the ``Z`` and ``+00:00`` spellings.

    The plan's timestamps are serialised from pandas and the page's from the Parquet
    reader; both are UTC but neither promises the other's spelling, so the comparison is
    between instants rather than between strings.
    """
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def mean_of(values: list[float]) -> float:
    return sum(values) / len(values)


def pstdev_of(values: list[float]) -> float:
    """Population standard deviation, matching ``StandardScaler``'s ddof=0 scale."""
    centre = mean_of(values)
    return math.sqrt(sum((value - centre) ** 2 for value in values) / len(values))


# ------------------------------------------------------------------- job helpers


def run_scaling(dataset_id: str, scaler_type: str, **overrides: Any) -> tuple[int, Any]:
    body: dict[str, Any] = {
        "dataset_id": dataset_id,
        "scaler_type": scaler_type,
        "features": [],
        "write": True,
    }
    body.update(overrides)
    return call("POST", "/scaling/fit_transform", body)


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


def scale(dataset_id: str, scaler_type: str, **overrides: Any) -> dict[str, Any]:
    """Accept a run and wait for it, returning the terminal job record."""
    status, accepted = run_scaling(dataset_id, scaler_type, **overrides)
    if status != 202 or not isinstance(accepted, dict):
        print(f"fit_transform was not accepted: {status} {json.dumps(accepted)[:300]}")
        sys.exit(1)
    return poll(str(accepted["job_id"]))


def result_of(job: dict[str, Any]) -> dict[str, Any]:
    return dict((job or {}).get("result") or {})


def report_of(job: dict[str, Any]) -> dict[str, Any]:
    return dict(result_of(job).get("report") or {})


def parameter_of(job: dict[str, Any], feature: str) -> dict[str, Any]:
    for parameter in result_of(job).get("parameters") or []:
        if parameter["feature"] == feature:
            return dict(parameter)
    print(f"no parameter for {feature}")
    sys.exit(1)


def plan_for(dataset_id: str) -> str:
    """Make a 70/15/15 plan and return its id."""
    status, created = call(
        "POST",
        "/splits/create",
        {
            "dataset_id": dataset_id,
            "split_type": "percentage_70_15_15",
            "parameters": {},
            "name": None,
        },
    )
    if status != 201:
        print(f"could not make a plan: {status} {json.dumps(created)[:300]}")
        sys.exit(1)
    return str(created["dataset"]["dataset_id"])


# --------------------------------------------------------------------- seeding

print("== seeding ==")
source_id = upload("e2e_scaling_hourly.csv", hourly_csv(ROWS, step_at=TEST_FROM))
status, source_meta = call("GET", f"/datasets/{source_id}")
source_meta = source_meta or {}
source_checksum = source_meta.get("checksum")
check(
    "the source dataset is uploaded and reports a checksum",
    status == 200 and bool(source_checksum) and source_meta.get("row_count") == ROWS,
    f"status={status} rows={source_meta.get('row_count')}",
)
if not source_checksum:
    print("cannot continue without a source checksum")
    sys.exit(1)

# --------------------------------------------------------------------- options

print("== options, before any plan exists ==")

status, options = call("GET", f"/scaling/options?dataset_id={source_id}")
options = options or {}
check(
    "options are served for the dataset",
    status == 200 and options.get("rows") == ROWS,
    f"status={status} rows={options.get('rows')}",
)
features = list(options.get("features") or [])
check(
    "the numeric columns are offered as features and the time columns are not",
    set(features) == {"open", "high", "low", "close", "volume"}
    and not any(name in features for name in NEVER_SCALED),
    str(features),
)

by_id = {entry["id"]: entry for entry in options.get("scalers") or []}
check(
    "the three fitted scalers are unavailable, and say why",
    all(by_id[name]["available"] is False for name in ("minmax", "standard", "robust"))
    and all(
        "split plan" in str(by_id[name]["reason"]) for name in ("minmax", "standard", "robust")
    ),
    json.dumps({name: by_id[name]["reason"] for name in ("minmax", "standard", "robust")})[:260],
)
check(
    "the rolling z-score is available without one, because it fits nothing",
    by_id["rolling_zscore"]["available"] is True
    and by_id["rolling_zscore"]["needs_split_plan"] is False
    and by_id["rolling_zscore"]["reason"] is None,
    json.dumps(by_id["rolling_zscore"]),
)
check(
    "the defaults are the server's own, so the page does not invent them",
    options.get("defaults", {}).get("scaler_type") == "standard"
    and options.get("defaults", {}).get("window") == WINDOW
    and list(options.get("defaults", {}).get("features") or []) == features,
    json.dumps(options.get("defaults"))[:200],
)
check(
    "no plan is listed yet, which is what the page greys the scalers from",
    options.get("plans") == [],
    json.dumps(options.get("plans"))[:120],
)

# ------------------------------------------------------------------ the plan

print("== a plan, and what it changes ==")

plan_id = plan_for(source_id)
status, options = call("GET", f"/scaling/options?dataset_id={source_id}")
options = options or {}
by_id = {entry["id"]: entry for entry in options.get("scalers") or []}
check(
    "every scaler becomes available once a plan exists",
    all(by_id[name]["available"] is True for name in by_id),
    json.dumps({name: by_id[name]["available"] for name in by_id}),
)
check(
    "and the plan is offered for the picker, with the type C7 has to reason about",
    [plan.get("split_plan_id") for plan in options.get("plans") or []] == [plan_id]
    and options["plans"][0].get("split_type") == "percentage_70_15_15",
    json.dumps(options.get("plans"))[:200],
)

# -------------------------------------------------------------- the handshake

print("== the job handshake ==")

status, accepted = run_scaling(
    source_id, "standard", split_plan_id=plan_id, features=["close"]
)
accepted = accepted or {}
check(
    "fit_transform is accepted with 202, not answered with 200",
    status == 202,
    f"status={status} body={json.dumps(accepted)[:160]}",
)
check(
    "the handle carries a job id and a poll path, and no result",
    bool(accepted.get("job_id"))
    and accepted.get("poll") == f"/api/jobs/{accepted.get('job_id')}"
    and "report" not in accepted
    and "parameters" not in accepted,
    json.dumps(accepted)[:200],
)
standard_job = poll(str(accepted["job_id"]))
check(
    "the job reaches a terminal state by polling",
    standard_job.get("status") == "succeeded",
    json.dumps(standard_job.get("error"))[:200]
    if standard_job.get("error")
    else str(standard_job.get("status")),
)
if standard_job.get("status") != "succeeded":
    print("cannot continue without a scaled dataset")
    sys.exit(1)

report = report_of(standard_job)
check(
    "the report says which rows the parameters were estimated from",
    report.get("fit_role") == "train"
    and report.get("fit_rows") == TEST_FROM
    and report.get("split_plan_id") == plan_id,
    f"fit_role={report.get('fit_role')} fit_rows={report.get('fit_rows')} "
    f"plan={report.get('split_plan_id')}",
)
check(
    "the fit span is the training span, stated in time as well as in rows",
    as_time(report.get("fit_start_time")) == as_time(page_row(source_id, 0)["open_time"])
    and as_time(report.get("fit_end_time"))
    == as_time(page_row(source_id, TEST_FROM - 1)["open_time"]),
    f"{report.get('fit_start_time')} -> {report.get('fit_end_time')}",
)

# ----------------------------------------------------------------- the artifact

print("== the stored artifacts ==")

scaled_meta = result_of(standard_job).get("dataset") or {}
params_meta = result_of(standard_job).get("params_dataset") or {}
scaled_id = scaled_meta.get("dataset_id")
params_id = params_meta.get("dataset_id")
check(
    "the scaled frame lands in processed_data, parented to its source",
    scaled_meta.get("category") == "processed_data"
    and scaled_meta.get("parent_dataset_id") == source_id
    and scaled_meta.get("operation") == "scaling",
    json.dumps(
        {
            key: scaled_meta.get(key)
            for key in ("category", "parent_dataset_id", "operation")
        }
    ),
)
check(
    "the parameters land in outputs, as a dataset of their own",
    params_meta.get("category") == "outputs"
    and params_meta.get("operation") == "scaler_params",
    json.dumps({key: params_meta.get(key) for key in ("category", "operation")}),
)
if not scaled_id or not params_id:
    print("the run wrote no artifacts; the leakage checks cannot run")
    sys.exit(1)

status, scaled_frame = call("GET", f"/datasets/{scaled_id}")
check(
    "the frame keeps every column and every row of its source",
    (scaled_frame or {}).get("row_count") == ROWS
    and [column["name"] for column in (scaled_frame or {}).get("columns", [])]
    == [column["name"] for column in source_meta.get("columns", [])],
    f"rows={(scaled_frame or {}).get('row_count')} "
    f"columns={[c['name'] for c in (scaled_frame or {}).get('columns', [])]}",
)

status, params_frame = call("GET", f"/datasets/{params_id}")
params_columns = tuple(
    str(column["name"]) for column in (params_frame or {}).get("columns", [])
)
check(
    "the parameter table carries the schema, in order",
    params_columns == PARAMS_COLUMNS,
    str(params_columns),
)
check(
    "it has one row per scaled feature",
    (params_frame or {}).get("row_count") == len(report.get("features") or []),
    f"{(params_frame or {}).get('row_count')} vs {report.get('features')}",
)

status, exported = call("GET", f"/datasets/{params_id}/export?format=json")
check(
    "the parameters are exportable as JSON, the format the specification names",
    status == 200 and isinstance(exported, list) and bool(exported),
    f"status={status} rows={len(exported) if isinstance(exported, list) else 'n/a'}",
)
if isinstance(exported, list) and exported:
    check(
        "the exported row carries the affine pair under the specification's own names",
        all(
            key in exported[0] for key in ("feature", "scaler_type", "center_", "scale_")
        ),
        json.dumps(exported[0])[:200],
    )

# -------------------------------------------------------------------- leakage

print("== no look-ahead: the fit never saw the test span ==")

source_rows = page_rows(source_id)
train_close = [float(row["close"]) for row in source_rows[:TEST_FROM]]
scaled_rows = page_rows(scaled_id)
scaled_close = [float(row["close"]) for row in scaled_rows]

parameter = parameter_of(standard_job, "close")
check(
    "center_ is the training span's own mean, not the series mean",
    abs(parameter["center_"] - mean_of(train_close)) < 1e-9,
    f"center_={parameter['center_']:.6f} train_mean={mean_of(train_close):.6f} "
    f"series_mean={mean_of([float(r['close']) for r in source_rows]):.6f}",
)
check(
    "scale_ is the training span's own population standard deviation",
    abs(parameter["scale_"] - pstdev_of(train_close)) < 1e-9,
    f"scale_={parameter['scale_']:.6f} train_pstdev={pstdev_of(train_close):.6f}",
)
check(
    "the training rows come out centred at zero, as any correct scaler would",
    abs(mean_of(scaled_close[:TEST_FROM])) < 1e-9,
    f"train mean after scaling={mean_of(scaled_close[:TEST_FROM]):.3e}",
)
test_mean = mean_of(scaled_close[TEST_FROM:])
check(
    "the test rows do not — they are expressed in training units, which is the point",
    abs(test_mean) > 1.0,
    f"test mean after scaling={test_mean:.3f} (a whole-series fit would put it near 0)",
)
check(
    "so a whole-series fit is distinguishable from this one by the artifact alone",
    abs(mean_of(scaled_close)) > 1.0,
    f"whole-frame mean={mean_of(scaled_close):.3f}",
)

# The strongest form of the same property: two frames that differ only after the cut must
# produce byte-identical parameters and byte-identical scaled training rows. Not "the obvious
# statistic did not leak" but "nothing from the test span reached the fit at all".
lifted_id = upload(
    "e2e_scaling_lifted.csv", hourly_csv(ROWS, step_at=TEST_FROM, tail_lift=500.0)
)
lifted_plan = plan_for(lifted_id)
lifted_job = scale(lifted_id, "standard", split_plan_id=lifted_plan, features=["close"])
lifted_meta = result_of(lifted_job).get("dataset") or {}
lifted_parameter = parameter_of(lifted_job, "close")

check(
    "a test span lifted by 500 leaves every fitted parameter exactly where it was",
    lifted_parameter["center_"] == parameter["center_"]
    and lifted_parameter["scale_"] == parameter["scale_"],
    f"center {parameter['center_']:.12f} vs {lifted_parameter['center_']:.12f}; "
    f"scale {parameter['scale_']:.12f} vs {lifted_parameter['scale_']:.12f}",
)
if lifted_meta.get("dataset_id"):
    lifted_close = [
        float(row["close"]) for row in page_rows(str(lifted_meta["dataset_id"]))
    ]
    worst = max(
        abs(a - b)
        for a, b in zip(scaled_close[:TEST_FROM], lifted_close[:TEST_FROM], strict=False)
    )
    check(
        "and leaves every transformed training row bit-identical",
        worst == 0.0,
        f"worst difference over {TEST_FROM} rows = {worst:.3e}",
    )
    check(
        "while the test rows do move, because the data there does",
        abs(mean_of(lifted_close[TEST_FROM:]) - test_mean) > 1.0,
        f"test mean {test_mean:.3f} -> {mean_of(lifted_close[TEST_FROM:]):.3f}",
    )
else:
    check("the lifted run stored a dataset to compare against", False, "none written")

# --------------------------------------------------------- the rolling z-score

print("== the rolling z-score is causal ==")

rolling_job = scale(source_id, "rolling_zscore", features=["close"], window=WINDOW)
rolling_report = report_of(rolling_job)
rolling_meta = result_of(rolling_job).get("dataset") or {}
check(
    "a rolling run needs no plan and says so",
    rolling_job.get("status") == "succeeded"
    and rolling_report.get("split_plan_id") is None
    and rolling_report.get("window") == WINDOW,
    f"status={rolling_job.get('status')} plan={rolling_report.get('split_plan_id')} "
    f"window={rolling_report.get('window')}",
)
rolling_id = str(rolling_meta.get("dataset_id") or "")
if not rolling_id:
    print("the rolling run wrote no dataset; the causality check cannot run")
    sys.exit(1)
rolling_close = [row["close"] for row in page_rows(rolling_id)]
check(
    "the first window - 1 rows are null, because there is no window behind them",
    all(value is None for value in rolling_close[: WINDOW - 1])
    and rolling_close[WINDOW - 1] is not None,
    f"rows 0..{WINDOW - 2} null, row {WINDOW - 1}="
    f"{rolling_close[WINDOW - 1] if len(rolling_close) > WINDOW - 1 else 'missing'}",
)

# The same request against the first 300 rows of the same frame. The generator is
# closed-form, so the truncated frame's rows are byte-identical to the full frame's prefix —
# which is what makes the comparison below a statement about causality rather than about
# two different datasets.
short_id = upload("e2e_scaling_short.csv", hourly_csv(300, step_at=TEST_FROM))
short_job = scale(short_id, "rolling_zscore", features=["close"], window=WINDOW)
short_meta = result_of(short_job).get("dataset") or {}
short_close = [
    row["close"] for row in page_rows(str(short_meta.get("dataset_id") or ""))
] if short_meta.get("dataset_id") else []
shared = list(zip(rolling_close[:300], short_close, strict=False))
# NaN is the warm-up: the first `window - 1` rows have no full trailing window, so both
# runs must agree that they are missing rather than agree on a number.
agrees = all(
    (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-12)
    for a, b in shared
)
worst = max(
    (abs(a - b) for a, b in shared if a is not None and b is not None), default=float("nan")
)
check(
    "truncating the frame to its first 300 rows leaves the shared prefix unchanged",
    len(short_close) == 300 and agrees,
    f"{len(short_close)} rows compared; worst={worst:.3e}",
)

# ------------------------------------------------------------------- refusals

print("== refusals ==")

for scaler in ("minmax", "standard", "robust"):
    status, payload = run_scaling(source_id, scaler, features=["close"])
    check(
        f"{scaler} without a split plan is refused before any job exists",
        status == 422 and "split_plan_id" in message_of(payload),
        f"status={status} {message_of(payload)[:140]}",
    )

status, payload = run_scaling(
    source_id, "rolling_zscore", split_plan_id=plan_id, features=["close"]
)
check(
    "a rolling run with a plan is refused, rather than accepting it and doing nothing",
    status == 422 and "rolling_zscore" in message_of(payload),
    f"status={status} {message_of(payload)[:160]}",
)

status, payload = run_scaling(
    source_id, "standard", split_plan_id=plan_id, features=["close"], window=WINDOW
)
check(
    "a window on a fitted scaler is refused, not quietly dropped",
    status == 422 and "window" in message_of(payload),
    f"status={status} {message_of(payload)[:160]}",
)

status, payload = run_scaling(
    source_id, "rolling_zscore", features=["close"], window=1
)
check(
    "a one-bar window is refused — it has no dispersion to divide by",
    status == 422,
    f"status={status} {message_of(payload)[:140]}",
)

status, payload = run_scaling(source_id, "rolling_zscore", features=["close"])
# No `window` in the request: the point is that the server supplies one. Reading it back out
# of the run's report is what makes this a check rather than a repeat of the run above, which
# named the window explicitly.
defaulted: dict[str, Any] = poll(str(payload["job_id"])) if status == 202 else {}
check(
    "a rolling run without a window takes the documented default",
    status == 202
    and defaulted.get("status") == "succeeded"
    and report_of(defaulted).get("window") == WINDOW,
    f"status={status} window={report_of(defaulted).get('window')} expected={WINDOW}",
)

status, payload = run_scaling("NOPE", "rolling_zscore", features=["close"])
check(
    "an unknown dataset is refused with a 404 before any job is created",
    status == 404,
    f"status={status} {message_of(payload)}",
)

status, payload = run_scaling(source_id, "standard", split_plan_id="NOPE", features=["close"])
failed = poll(str(payload["job_id"])) if status == 202 else {}
check(
    "a plan that does not exist fails inside the job, where there is a channel for it",
    status == 202
    and failed.get("status") == "failed"
    and "plan" in str((failed.get("error") or {}).get("message", "")).lower(),
    f"status={status} job={failed.get('status')} "
    f"{json.dumps(failed.get('error'))[:180]}",
)

# A plan made from a different frame. Its row indices would apply cleanly to the wrong rows
# and the spans would look plausible, which is why this is a refusal and not a warning.
foreign_plan = plan_for(lifted_id)
status, payload = run_scaling(
    source_id, "standard", split_plan_id=foreign_plan, features=["close"]
)
failed = poll(str(payload["job_id"])) if status == 202 else {}
check(
    "a plan belonging to another dataset is refused, naming the mismatch",
    failed.get("status") == "failed"
    and "different dataset" in str((failed.get("error") or {}).get("message", "")),
    f"status={status} {json.dumps(failed.get('error'))[:200]}",
)

status, payload = run_scaling(
    source_id, "standard", split_plan_id=plan_id, features=["not_a_column"]
)
failed = poll(str(payload["job_id"])) if status == 202 else {}
check(
    "a feature that is not in the frame is refused, naming the column",
    failed.get("status") == "failed"
    and "not_a_column" in str((failed.get("error") or {}).get("message", "")),
    f"status={status} {json.dumps(failed.get('error'))[:200]}",
)

# A walk-forward plan holds one train/test pair per fold, and pooling them would compute the
# first fold's statistics from rows lying after its test span.
status, tscv_plan = call(
    "POST",
    "/splits/create",
    {
        "dataset_id": source_id,
        "split_type": "tscv_expanding",
        "parameters": {"n_folds": 4},
        "name": None,
    },
)
tscv_plan_id = str((tscv_plan or {}).get("dataset", {}).get("dataset_id") or "")
status, payload = run_scaling(
    source_id, "standard", split_plan_id=tscv_plan_id, features=["close"]
)
failed = poll(str(payload["job_id"])) if status == 202 else {}
check(
    "a walk-forward plan without a fold is refused, explaining what fits what",
    tscv_plan_id != ""
    and failed.get("status") == "failed"
    and "fold" in str((failed.get("error") or {}).get("message", "")),
    f"plan={tscv_plan_id} {json.dumps(failed.get('error'))[:200]}",
)

status, payload = run_scaling(
    source_id, "standard", split_plan_id=tscv_plan_id, fold=1, features=["close"]
)
with_fold = poll(str(payload["job_id"])) if status == 202 else {}
check(
    "and with a fold it runs, fitting on that fold's training span alone",
    with_fold.get("status") == "succeeded"
    and report_of(with_fold).get("fold") == 1
    and report_of(with_fold).get("fit_rows") == ROWS // 5,
    f"status={with_fold.get('status')} fold={report_of(with_fold).get('fold')} "
    f"fit_rows={report_of(with_fold).get('fit_rows')}",
)

status, payload = run_scaling(
    source_id, "standard", split_plan_id=tscv_plan_id, fold=99, features=["close"]
)
failed = poll(str(payload["job_id"])) if status == 202 else {}
check(
    "a fold outside the plan is refused rather than clamped",
    failed.get("status") == "failed"
    and "99" in str((failed.get("error") or {}).get("message", "")),
    f"status={status} {json.dumps(failed.get('error'))[:200]}",
)

# -------------------------------------------------- preview-only and determinism

print("== preview only, and repeats ==")

preview_job = scale(source_id, "robust", split_plan_id=plan_id, features=["close"], write=False)
check(
    "write=false computes the full result and stores nothing",
    preview_job.get("status") == "succeeded"
    and result_of(preview_job).get("dataset") is None
    and result_of(preview_job).get("params_dataset") is None
    and preview_job.get("output_dataset_id") is None,
    f"dataset={result_of(preview_job).get('dataset')} "
    f"output={preview_job.get('output_dataset_id')}",
)
check(
    "and still returns the parameters and a readable preview",
    bool(result_of(preview_job).get("parameters"))
    and len(result_of(preview_job).get("preview", {}).get("tail") or []) > 0,
    f"{len(result_of(preview_job).get('parameters') or [])} parameters, "
    f"{len(result_of(preview_job).get('preview', {}).get('tail') or [])} tail rows",
)

repeat_job = scale(source_id, "standard", split_plan_id=plan_id, features=["close"])
check(
    "the same request produces the same parameters",
    result_of(repeat_job).get("parameters") == result_of(standard_job).get("parameters"),
    json.dumps(result_of(repeat_job).get("parameters"))[:200],
)
check(
    "and the same preview, so a stored frame is reproducible from its request",
    result_of(repeat_job).get("preview") == result_of(standard_job).get("preview"),
    "head and tail compared",
)

# ------------------------------------------------- immutability and provenance

print("== immutability and provenance ==")

status, source_after = call("GET", f"/datasets/{source_id}")
check(
    "the source dataset is byte-identical after every run against it",
    (source_after or {}).get("checksum") == source_checksum,
    f"{str(source_checksum)[:12]} -> {str((source_after or {}).get('checksum'))[:12]}",
)

provenance = scaled_meta.get("provenance") or {}
check(
    "the scaled frame's provenance records the scaler and the plan it was fitted against",
    provenance.get("module_source") == "scaling"
    and provenance.get("scaler_type") == "standard"
    and provenance.get("split_plan_id") == plan_id,
    json.dumps(provenance)[:240],
)
check(
    "and names the features it replaced, so the artifact says what changed",
    provenance.get("scaled_features") == "close",
    str(provenance.get("scaled_features")),
)

status, scaled_again = call("GET", f"/datasets/{scaled_id}")
check(
    "the artifact itself is unchanged by the repeat run, which wrote a new dataset",
    (scaled_again or {}).get("checksum") == scaled_meta.get("checksum")
    and result_of(repeat_job).get("dataset", {}).get("dataset_id") != scaled_id,
    f"first={scaled_id} repeat="
    f"{result_of(repeat_job).get('dataset', {}).get('dataset_id')}",
)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for item in FAILURES:
        print("  -", item)
    sys.exit(1)
print("all scaling-endpoint checks passed")
