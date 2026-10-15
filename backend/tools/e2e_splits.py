"""End-to-end exercise of the C6 split endpoints over HTTP.

Not a unit test: it drives the running server as the browser does, and it is here for the
three things an in-process test cannot show.

**The plan is written to Parquet and read back.** ``tests/test_splits.py`` asserts the
geometry the service computed in memory. What it cannot show is that the geometry survives
the round trip — that ``start_index`` lands as an integer column rather than ``object``,
that ``start_time`` comes back tz-aware UTC rather than reinterpreted as local time, and
that the stored table has the columns in the order ``PLAN_COLUMNS`` names. C7 reads that
file and slices the source frame with those integers; a dtype that drifted would not fail
here, it would fail three modules later as a wrong row.

**The plan's timestamps are the observation column, not the bar close.** The driver takes a
span's ``start_index``, pages that row of the *source* dataset, and requires the plan's
``start_time`` to equal the source's ``open_time`` at that index — and to differ from its
``close_time``, which is one bar later. Splitting on bar-close instants dates every span one
bar late with nothing in the table to show it, and this is the check that catches it. It is
only meaningful against the stored artifacts, which is why it lives here.

**The source is untouched.** The mandate for C6 is that a plan describes a frame rather than
slicing one. The source's checksum is read before and after every create, so a module that
quietly rewrote its input would fail here.

Run it against a live server (``python tools/serve.py`` in another shell):

    python tools/e2e_splits.py
"""

from __future__ import annotations

import json
import math
import sys
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

#: The specification's plan schema plus the two columns this implementation adds — the index
#: pair, which makes the plan unambiguous without a timestamp search, and the checksum, which
#: makes it refuse a frame it was not computed on. Asserted as an ordered tuple: C7 reads
#: these by name, and a positional reader is a thing that will eventually exist.
PLAN_COLUMNS = (
    "split_id",
    "fold",
    "role",
    "start_index",
    "end_index",
    "start_time",
    "end_time",
    "row_count",
    "dataset_checksum",
)

#: The reference frame: 400 hourly bars. Large enough that floor() rounding is exercised
#: rather than approximated, and that a walk-forward fold has a block worth training on.
ROWS = 400

FOLDS = 4
BLOCK = ROWS // (FOLDS + 1)


def call(method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    """One request, via urllib, for a JSON response.

    Not httpx: on this machine a pooled httpx connection to uvicorn was reset before the
    request reached the server, which made a transport quirk look like a failing endpoint.
    The stdlib client opens a fresh connection per call, so a failure here is a failure of
    the API.

    ``path`` is relative to ``BASE``. The prefix is already in ``BASE``, and repeating it
    turns ``/splits/plans`` into ``/api/api/splits/plans`` — which 404s in a way
    indistinguishable from a route that does not exist, so the mistake would read as a
    missing feature.
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


def message_of(payload: Any) -> str:
    """The reason a request was refused, from either error shape.

    A refusal this application raises carries ``error.message``. A pydantic constraint
    violation — ``n_folds: 11``, ``gap_size: -1`` — is rejected by FastAPI before any handler
    runs and carries ``detail[].loc`` instead, with no ``error`` key at all. Reading only the
    first shape would make every field-level check below compare an empty string and fail on
    a server that refused the request perfectly correctly, so the raw body is the fallback.
    """
    message = str(((payload or {}).get("error") or {}).get("message", ""))
    return message or json.dumps(payload)[:200]


def upload(name: str, body: str) -> str:
    """POST a CSV as multipart/form-data, built by hand to keep the driver stdlib-only."""
    boundary = "----quantmetrics-e2e-splits-boundary"
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


def hourly_csv(count: int, *, step_at: int | None = None, base: float = 100.0) -> str:
    """A deterministic hourly OHLCV frame on an exact grid.

    Sine plus a small drift rather than a random walk: the driver has to be reproducible
    across machines, and a generator whose stream could change between releases would make
    "the same request gives the same plan" fail for a reason that has nothing to do with the
    module. The bars are strictly increasing and unique, which is what C6 requires of a
    splittable frame.

    ``close_time`` is carried for one check and one only: the plan's timestamps have to be
    the *observation* instants, and the way to show that is to have the other instant in the
    frame to compare against. On this grid the two differ by one bar, so a plan dated by bar
    close is one row late on every span and a comparison finds it. C6 never offers the column
    as a timeline — ``quality.TIME_COLUMN_CANDIDATES`` does not list it — so carrying it
    costs the module nothing.

    ``step_at`` lifts every bar from that index onward by 100. C6 does not care, but the C7
    driver reuses this generator and does: a frame whose two halves sit at different levels
    is what makes a whole-series fit visibly different from a training-span fit.
    """
    rows = ["open_time,close_time,open,high,low,close,volume"]
    for index in range(count):
        level = base + (100.0 if step_at is not None and index >= step_at else 0.0)
        open_price = level + 10.0 * math.sin(index / 23.0) + 0.01 * index
        close = level + 10.0 * math.sin((index + 1) / 23.0) + 0.01 * (index + 1)
        high = max(open_price, close) + 0.5
        low = min(open_price, close) - 0.5
        volume = 1000.0 + 50.0 * math.cos(index / 11.0)
        stamp = datetime(2023, 1, 1) + timedelta(hours=index)
        closes = stamp + timedelta(hours=1) - timedelta(milliseconds=1)
        rows.append(
            f"{stamp.strftime('%Y-%m-%dT%H:%M:%SZ')},{closes.strftime('%Y-%m-%dT%H:%M:%S.%f')[:23]}Z,"
            f"{open_price:.4f},{high:.4f},{low:.4f},{close:.4f},{volume:.4f}"
        )
    return "\n".join(rows) + "\n"


def as_time(value: Any) -> datetime:
    """Parse an ISO stamp, tolerating both the ``Z`` and ``+00:00`` spellings."""
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def create(dataset_id: str, split_type: str, **parameters: Any) -> tuple[int, Any]:
    return call(
        "POST",
        "/splits/create",
        {
            "dataset_id": dataset_id,
            "split_type": split_type,
            "parameters": parameters,
            "name": None,
        },
    )


def spans_of(payload: Any) -> list[dict[str, Any]]:
    return list((payload or {}).get("plan") or [])


def report_of(payload: Any) -> dict[str, Any]:
    return dict((payload or {}).get("report") or {})


def plan_dataset_of(payload: Any) -> dict[str, Any]:
    return dict((payload or {}).get("dataset") or {})


def fold_pairs(plan: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every fold's (train span, test span), in fold order."""
    folds = sorted({int(span["fold"]) for span in plan})
    pairs = []
    for fold in folds:
        train = next(s for s in plan if s["fold"] == fold and s["role"] == "train")
        test = next(s for s in plan if s["fold"] == fold and s["role"] == "test")
        pairs.append((train, test))
    return pairs


def page_row(dataset_id: str, offset: int) -> dict[str, Any]:
    """One row of a stored dataset, by position."""
    status, body = call("GET", f"/datasets/{dataset_id}/page?offset={offset}&limit=1")
    rows = list((body or {}).get("rows") or []) if status == 200 else []
    if not rows:
        print(f"no row {offset} in {dataset_id}: {status} {json.dumps(body)[:200]}")
        sys.exit(1)
    return dict(rows[0])


def page_rows(dataset_id: str, *, limit: int = 1000) -> list[dict[str, Any]]:
    status, body = call("GET", f"/datasets/{dataset_id}/page?offset=0&limit={limit}")
    if status != 200 or not isinstance(body, dict):
        print(f"paging {dataset_id} failed: {status} {json.dumps(body)[:200]}")
        sys.exit(1)
    if body.get("total", 0) > len(body.get("rows", [])):
        print(f"{dataset_id} has {body['total']} rows — more than one page; widen the driver")
        sys.exit(1)
    return list(body.get("rows", []))


# --------------------------------------------------------------------- seeding

print("== seeding ==")
source_id = upload("e2e_splits_hourly.csv", hourly_csv(ROWS, step_at=280))
status, source_meta = call("GET", f"/datasets/{source_id}")
source_meta = source_meta or {}
source_checksum = source_meta.get("checksum")
check(
    "the source dataset is uploaded and reports a checksum",
    status == 200 and bool(source_checksum) and source_meta.get("row_count") == ROWS,
    f"status={status} rows={source_meta.get('row_count')} checksum={str(source_checksum)[:12]}",
)
if not source_checksum:
    print("cannot continue without a source checksum")
    sys.exit(1)

# ------------------------------------------------------------------- percentage

print("== percentage split ==")

status, created = create(source_id, "percentage_70_15_15")
plan = spans_of(created)
report = report_of(created)
check(
    "a percentage plan is answered with 201 and the finished object, not a job handle",
    status == 201 and bool(plan) and "report" in (created or {}),
    f"status={status} spans={len(plan)}" + ("" if status == 201 else f" {message_of(created)}"),
)
if status != 201:
    print("cannot continue without a plan")
    sys.exit(1)

check(
    "the plan has one span per role, in train/val/test order",
    [span["role"] for span in plan] == ["train", "val", "test"],
    str([span["role"] for span in plan]),
)
expected_cuts = (math.floor(ROWS * 0.70), math.floor(ROWS * 0.85), ROWS)
check(
    "the cuts are floor(N x p), so the three spans sum to exactly N",
    [span["start_index"] for span in plan] == [0, expected_cuts[0], expected_cuts[1]]
    and [span["end_index"] for span in plan] == list(expected_cuts),
    f"bounds={[(s['start_index'], s['end_index']) for s in plan]} expected={expected_cuts}",
)
check(
    "each span's row_count is its own index range, and the total is the frame",
    all(span["row_count"] == span["end_index"] - span["start_index"] for span in plan)
    and sum(span["row_count"] for span in plan) == ROWS,
    f"counts={[s['row_count'] for s in plan]} sum={sum(s['row_count'] for s in plan)}",
)
check(
    "rows_by_role agrees with the spans the plan carries",
    report.get("rows_by_role", {}).get("test") == plan[2]["row_count"]
    and report.get("rows_by_role", {}).get("train") == plan[0]["row_count"],
    json.dumps(report.get("rows_by_role")),
)
check(
    "every row is in exactly one test span — nothing is left unscored",
    report.get("rows_tested") == plan[2]["row_count"]
    and report.get("rows_never_tested") == ROWS - plan[2]["row_count"],
    f"tested={report.get('rows_tested')} never={report.get('rows_never_tested')}",
)
check(
    "every span carries the checksum of the frame it was computed on",
    all(span["dataset_checksum"] == source_checksum for span in plan),
    str({str(span["dataset_checksum"])[:12] for span in plan}),
)

# ------------------------------------------------------------ the stored table

print("== the stored plan table ==")

plan_id = plan_dataset_of(created).get("dataset_id")
check(
    "the plan is stored as an outputs dataset, produced by the split operation",
    plan_dataset_of(created).get("category") == "outputs"
    and plan_dataset_of(created).get("operation") == "split",
    json.dumps(
        {
            key: plan_dataset_of(created).get(key)
            for key in ("category", "operation", "parent_dataset_id")
        }
    ),
)
check(
    "the plan is parented to the dataset it describes",
    plan_dataset_of(created).get("parent_dataset_id") == source_id,
    str(plan_dataset_of(created).get("parent_dataset_id")),
)
if not plan_id:
    print("the plan wrote no dataset; the round-trip checks cannot run")
    sys.exit(1)

status, plan_meta = call("GET", f"/datasets/{plan_id}")
plan_columns = tuple(str(column["name"]) for column in (plan_meta or {}).get("columns", []))
check(
    "the stored table carries the plan schema, in order",
    plan_columns == PLAN_COLUMNS,
    str(plan_columns),
)
check(
    "the stored table has one row per span",
    (plan_meta or {}).get("row_count") == len(plan),
    f"{(plan_meta or {}).get('row_count')} vs {len(plan)}",
)

stored = page_rows(plan_id)
check(
    "the stored rows are the plan, span for span",
    [(int(row["start_index"]), int(row["end_index"])) for row in stored]
    == [(span["start_index"], span["end_index"]) for span in plan],
    str([(row["start_index"], row["end_index"]) for row in stored]),
)

# The load-bearing check of this driver. The plan's boundaries must be the *observation*
# instant — the bar's open — and not its close. Both columns exist on this frame, they differ
# by exactly one bar, and dating a span by close moves its first row one bar late.
first_test_start = plan[2]["start_index"]
source_at_cut = page_row(source_id, first_test_start)
check(
    "a span's start_time is the source's open_time at start_index, bar for bar",
    as_time(plan[2]["start_time"]) == as_time(source_at_cut["open_time"]),
    f"plan={plan[2]['start_time']} source.open_time={source_at_cut['open_time']}",
)
check(
    "and it is deliberately not the bar's close, which sits one bar later",
    as_time(plan[2]["start_time"]) != as_time(source_at_cut["close_time"]),
    f"plan={plan[2]['start_time']} source.close_time={source_at_cut['close_time']}",
)
last_train_index = plan[0]["end_index"] - 1
source_at_end = page_row(source_id, last_train_index)
check(
    "a span's end_time is the last bar it contains, inclusive",
    as_time(plan[0]["end_time"]) == as_time(source_at_end["open_time"]),
    f"plan={plan[0]['end_time']} source[{last_train_index}].open_time="
    f"{source_at_end['open_time']}",
)

status, plans = call("GET", f"/splits/plans?dataset_id={source_id}")
listed = [row for row in (plans or []) if row.get("split_plan_id") == plan_id]
check(
    "the plan is listed against its own dataset",
    status == 200 and len(listed) == 1,
    f"status={status} matches={len(listed)} of {len(plans or [])}",
)
check(
    "the listing is enough to build a picker from, without opening the table",
    bool(listed)
    and listed[0].get("split_type") == "percentage_70_15_15"
    # The plan's own row count — one row per span — not the source frame's. A picker that
    # said 400 here would be quoting a number the plan table does not contain, and the
    # three spans the driver just read from the 201 are what it must agree with.
    and listed[0].get("rows") == len(plan)
    and listed[0].get("folds") == 1,
    json.dumps(listed[0])[:200] if listed else "not listed",
)

# ------------------------------------------------------------------- date splits

print("== date splits ==")

rows = page_rows(source_id)
train_end = rows[199]["open_time"]
val_start = rows[200]["open_time"]
val_end = rows[299]["open_time"]
test_start = rows[300]["open_time"]
test_end = rows[ROWS - 1]["open_time"]

status, dated = create(
    source_id,
    "date_3way",
    train_start=rows[0]["open_time"],
    train_end=train_end,
    val_start=val_start,
    val_end=val_end,
    test_start=test_start,
    test_end=test_end,
)
dated_plan = spans_of(dated)
check(
    "a three-way date split is accepted and resolves each boundary to its bar",
    status == 201
    and [(span["start_index"], span["end_index"]) for span in dated_plan]
    == [(0, 200), (200, 300), (300, ROWS)],
    f"status={status} bounds={[(s['start_index'], s['end_index']) for s in dated_plan]}"
    + ("" if status == 201 else f" {message_of(dated)}"),
)
check(
    "the named last bar of a span is inside it, not outside",
    bool(dated_plan)
    and as_time(dated_plan[0]["end_time"]) == as_time(train_end)
    and dated_plan[1]["start_index"] == 200,
    f"train ends {dated_plan[0]['end_time'] if dated_plan else None} at row "
    f"{dated_plan[0]['end_index'] - 1 if dated_plan else None}",
)

status, two_way = create(
    source_id,
    "date_2way",
    train_start=rows[0]["open_time"],
    train_end=rows[199]["open_time"],
    test_start=val_start,
    test_end=test_end,
)
check(
    "a two-way date split produces no validation span",
    status == 201
    and [span["role"] for span in spans_of(two_way)] == ["train", "test"]
    and "val" not in report_of(two_way).get("rows_by_role", {}),
    f"status={status} roles={[s['role'] for s in spans_of(two_way)]} "
    f"by_role={json.dumps(report_of(two_way).get('rows_by_role'))}",
)

status, payload = create(
    source_id,
    "date_3way",
    train_start=rows[0]["open_time"],
    train_end=rows[399]["open_time"],
    val_start=rows[100]["open_time"],
    val_end=rows[199]["open_time"],
    test_start=rows[200]["open_time"],
    test_end=rows[399]["open_time"],
)
check(
    "boundaries that run backwards are refused with 422 before any plan is written",
    status == 422,
    f"status={status} {message_of(payload)}",
)

status, payload = create(
    source_id,
    "date_3way",
    train_start=rows[0]["open_time"],
    train_end=rows[100]["open_time"],
    val_start=rows[101]["open_time"],
    val_end=rows[200]["open_time"],
    test_start=rows[201]["open_time"],
    test_end=rows[399]["open_time"],
    gap_size=5,
)
check(
    "a date split refuses gap_size, where the user names the boundary directly",
    status == 422 and "date" in message_of(payload).lower(),
    f"status={status} {message_of(payload)[:160]}",
)

# ------------------------------------------------------- walk-forward and purge

print("== walk-forward, embargo and purge ==")

status, tscv = create(source_id, "tscv_expanding", n_folds=FOLDS)
tscv_plan = spans_of(tscv)
pairs = fold_pairs(tscv_plan) if tscv_plan else []
first_train, first_test = pairs[0] if pairs else ({}, {})
check(
    "an expanding walk-forward cuts the series into folds + 1 equal blocks",
    status == 201
    and len(pairs) == FOLDS
    and (first_train.get("start_index"), first_train.get("end_index")) == (0, BLOCK)
    and (first_test.get("start_index"), first_test.get("end_index")) == (BLOCK, 2 * BLOCK),
    f"status={status} folds={len(pairs)} block={BLOCK} "
    f"train={first_train.get('start_index')}:{first_train.get('end_index')} "
    f"test={first_test.get('start_index')}:{first_test.get('end_index')}",
)
check(
    "every fold's training span ends at or before its test span begins",
    bool(pairs) and all(t["end_index"] <= s["start_index"] for t, s in pairs),
    f"{len(pairs)} folds checked",
)
check(
    "the first block is never tested — it is the data every fold starts from",
    bool(pairs) and all(s["start_index"] >= BLOCK for _, s in pairs),
    f"earliest test start={min((s['start_index'] for _, s in pairs), default=None)}",
)
check(
    "the last fold absorbs the remainder, so no tail row is left untested",
    bool(pairs) and any(s["end_index"] == ROWS for _, s in pairs),
    f"last test end={max((s['end_index'] for _, s in pairs), default=None)} rows={ROWS}",
)

status, purged = create(
    source_id, "tscv_expanding", n_folds=FOLDS, purge_flag=True, label_horizon_bars=5
)
purged_plan = spans_of(purged)
purged_report = report_of(purged)
boundary = (purged_report.get("boundaries") or [{}])[0]
p_train = next(r for r in purged_plan if r["fold"] == 1 and r["role"] == "train")
p_test = next(r for r in purged_plan if r["fold"] == 1 and r["role"] == "test")
check(
    "the purge removes exactly label_horizon_bars rows before the cut",
    status == 201
    and boundary.get("cut_index") == BLOCK
    and boundary.get("purge_rows_removed") == 5
    and boundary.get("effective_end_index") == BLOCK - 5,
    f"status={status} boundary={json.dumps(boundary)}",
)
check(
    "so the training span ends five rows before the test span begins",
    p_train["end_index"] == p_test["start_index"] - 5,
    f"train ends at {p_train['end_index']}, test starts at {p_test['start_index']}",
)
# Scoped to fold 1, and deliberately. An expanding fold re-trains from row zero every time,
# so a row purged at fold 1's boundary is legitimately inside fold 2's training span — fold
# 2's test span starts 85 rows later, and a row 85 bars upstream of a test span is not
# look-ahead. What must not happen is the purged row staying where it was, in the span it was
# cut from, or being handed to the test span instead.
fold_1 = [span for span in purged_plan if span["fold"] == 1]
dropped = range(p_train["end_index"], p_test["start_index"])
check(
    "the purged rows are in neither the training span they were cut from nor the test span",
    dropped.start < dropped.stop
    and all(
        not (span["start_index"] <= row < span["end_index"])
        for row in dropped
        for span in fold_1
    ),
    f"rows {dropped.start}..{dropped.stop - 1} against fold 1 spans "
    f"{[(s['role'], s['start_index'], s['end_index']) for s in fold_1]}",
)

status, both = create(
    source_id,
    "tscv_expanding",
    n_folds=FOLDS,
    gap_size=5,
    purge_flag=True,
    label_horizon_bars=5,
)
both_report = report_of(both)
both_boundary = (both_report.get("boundaries") or [{}])[0]
check(
    "the embargo and the purge overlap rather than adding up",
    both_boundary.get("gap_rows_removed") == 5
    and both_boundary.get("purge_rows_removed") == 0
    and both_boundary.get("effective_end_index") == BLOCK - 5,
    json.dumps(both_boundary),
)
check(
    "and the report says so, rather than presenting two knobs as twice the safety",
    any("purge_flag removed no rows" in note for note in both_report.get("warnings", [])),
    json.dumps(both_report.get("warnings"))[:220],
)

status, rolling = create(source_id, "tscv_rolling", n_folds=FOLDS)
rolling_plan = spans_of(rolling)
fold3_train = next(r for r in rolling_plan if r["fold"] == 3 and r["role"] == "train")
fold3_test = next(r for r in rolling_plan if r["fold"] == 3 and r["role"] == "test")
check(
    "a rolling window trains on the single block before the cut, not on all history",
    status == 201
    and fold3_train["end_index"] - fold3_train["start_index"] == BLOCK
    and fold3_train["end_index"] == fold3_test["start_index"],
    f"status={status} train={fold3_train['start_index']}:{fold3_train['end_index']} "
    f"test starts {fold3_test['start_index']} (block={BLOCK})",
)
expanding_fold3 = next(
    (r for r in tscv_plan if r["fold"] == 3 and r["role"] == "train"), None
)
check(
    "and an expanding window trains on everything before it, so the two differ by design",
    expanding_fold3 is not None and expanding_fold3["start_index"] == 0,
    f"expanding fold 3 trains from {expanding_fold3 and expanding_fold3['start_index']}",
)

# ------------------------------------------------------------------- refusals

print("== refusals ==")

tiny_id = upload("e2e_splits_tiny.csv", hourly_csv(2))
status, payload = create(tiny_id, "percentage_70_15_15")
check(
    "a split that would leave a one-row training span is refused, naming the span",
    status == 422 and "training span" in message_of(payload),
    f"status={status} {message_of(payload)[:160]}",
)

short_id = upload("e2e_splits_short.csv", hourly_csv(15))
status, payload = create(short_id, "tscv_expanding", n_folds=10)
check(
    "more folds than the frame can seat are refused, with the arithmetic",
    status == 422 and "block" in message_of(payload),
    f"status={status} {message_of(payload)[:160]}",
)

status, payload = create(source_id, "tscv_expanding", n_folds=FOLDS, purge_flag=True)
check(
    "purge_flag without a label horizon is refused — the flag means nothing alone",
    status == 422 and "label_horizon" in message_of(payload),
    f"status={status} {message_of(payload)[:160]}",
)

for label, body, fragment, expected in [
    ("n_folds outside 2..10 is refused",
     {"split_type": "tscv_expanding", "n_folds": 11}, "n_folds", 422),
    ("a negative gap_size is refused",
     {"split_type": "percentage_70_15_15", "gap_size": -1}, "gap_size", 422),
    ("a gap_size beyond the cap is refused",
     {"split_type": "percentage_70_15_15", "gap_size": 21}, "gap_size", 422),
    ("a percentage split with no parameters at all takes the documented defaults",
     {"split_type": "percentage_60_20_20"}, "", 201),
]:
    parameters = {key: value for key, value in body.items() if key != "split_type"}
    status, payload = call(
        "POST",
        "/splits/create",
        {
            "dataset_id": source_id,
            "split_type": body["split_type"],
            "parameters": parameters,
            "name": None,
        },
    )
    if expected == 201:
        check(label, status == 201, f"status={status} {message_of(payload)}")
    else:
        check(
            label,
            status == 422 and fragment.lower() in message_of(payload).lower(),
            f"status={status} {message_of(payload)[:140]}",
        )

status, payload = create(
    source_id,
    "date_3way",
    train_start=rows[0]["open_time"],
    train_end=train_end,
    val_start=val_start,
    val_end=val_end,
)
check(
    "a date split missing a boundary pair is refused rather than defaulted",
    status == 422,
    f"status={status} {message_of(payload)[:160]}",
)

status, payload = create(
    source_id,
    "date_3way",
    train_start="2023-01-01T00:00",
    train_end=train_end,
    val_start=val_start,
    val_end=val_end,
    test_start=test_start,
    test_end=test_end,
)
check(
    "a zone-less boundary is refused — it would be read in the server's local zone",
    status == 422,
    f"status={status} {message_of(payload)[:160]}",
)

status, payload = create("NOPE", "percentage_70_15_15")
check("an unknown dataset is refused with a 404", status == 404,
      f"status={status} {message_of(payload)}")

status, payload = call("GET", "/splits/plans?dataset_id=NOPE")
check(
    "listing the plans of a dataset that does not exist is a 404, not an empty list",
    status == 404,
    f"status={status} {message_of(payload)}",
)

status, payload = call("POST", "/splits/create", {"dataset_id": source_id})
check(
    "a request with no split_type at all is a 422, not a defaulted plan",
    status == 422,
    f"status={status} {message_of(payload)[:120]}",
)

# ---------------------------------------------------- immutability and repeats

print("== immutability and repeats ==")

status, source_after = call("GET", f"/datasets/{source_id}")
check(
    "the source dataset is byte-identical after every plan made from it",
    (source_after or {}).get("checksum") == source_checksum,
    f"{str(source_checksum)[:12]} -> {str((source_after or {}).get('checksum'))[:12]}",
)

status, again = create(source_id, "percentage_70_15_15")
again_plan = spans_of(again)
check(
    "the same request produces the same geometry",
    [(s["start_index"], s["end_index"], s["role"]) for s in again_plan]
    == [(s["start_index"], s["end_index"], s["role"]) for s in plan],
    str([(s["start_index"], s["end_index"]) for s in again_plan]),
)
second_id = plan_dataset_of(again).get("dataset_id")
check(
    "but it is a new plan with its own id, so both are addressable",
    second_id is not None and second_id != plan_id,
    f"{second_id} vs {plan_id}",
)

status, listed_again = call("GET", f"/splits/plans?dataset_id={source_id}")
listed_ids = [row["split_plan_id"] for row in (listed_again or [])]
check(
    "both plans are listed, and only from this dataset",
    status == 200 and plan_id in listed_ids and second_id in listed_ids,
    f"{len(listed_ids)} plans: {listed_ids}",
)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for item in FAILURES:
        print("  -", item)
    sys.exit(1)
print("all split-endpoint checks passed")
