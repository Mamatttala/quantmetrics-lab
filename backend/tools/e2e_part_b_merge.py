"""End-to-end exercise of the Resample & Merge endpoint over HTTP.

Not a unit test: it drives the running server exactly as the browser does. Datasets are
discovered from `GET /datasets` rather than hard-coded, so the script does not depend on
ids from an earlier session's scratch data.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any

# A driver run by hand inherits its console's encoding — cp1256 on this machine, which
# cannot encode the `→` printed for a two-dataset join, so a run whose every check had
# passed ended in UnicodeEncodeError instead of a verdict. `verify_all.py` sets
# PYTHONIOENCODING for the children it starts; this covers the direct invocation.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8000/api"
FAILURES: list[str] = []


def call(method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode()
            return response.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as error:
        raw = error.read().decode()
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, {"raw": raw[:400]}


def call_bytes(url: str) -> tuple[int, bytes, str]:
    """A GET whose body may be binary (the parquet export)."""
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers.get("Content-Type", "")


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(label)


#: Bar length in seconds, shortest first. The resample target is looked up on this ladder
#: rather than hard-coded, because it must be *strictly coarser* than the base:
#: `resample_frame` refuses the other direction, since building four 15-minute bars from
#: one hourly bar fabricates detail no market produced. A hard-coded "4h" is refused
#: outright against a daily base, and the driver would report a correct server as broken.
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


def coarser_than(timeframe: str | None) -> str | None:
    """The next rung above `timeframe`, or None when there is none."""
    seconds = TIMEFRAME_SECONDS.get(timeframe or "")
    if seconds is None:
        return None
    return next((name for name, value in TIMEFRAME_SECONDS.items() if value > seconds), None)


def resample_target(dataset: dict[str, Any]) -> str | None:
    """A legal resample target for this dataset, or None if nothing is coarser.

    A dataset declaring no timeframe is treated as resamplable to anything: the server
    permits it and only warns, having no source interval to check the request against.
    """
    if not dataset.get("timeframe"):
        return "4h"
    return coarser_than(dataset["timeframe"])


def parse_stamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def merge(payload: dict[str, Any]) -> tuple[int, Any]:
    return call("POST", "/datasets/merge", payload)


def column_names(body: Any) -> list[str]:
    preview = (body or {}).get("preview") or {}
    return [str(c.get("name")) for c in preview.get("columns", [])]


# ------------------------------------------------------------------ discovery

status, listing = call("GET", "/datasets")
if status != 200 or not isinstance(listing, list):
    print(f"cannot list datasets: {status} {listing}")
    sys.exit(1)
rows = [d for d in listing if d.get("row_count", 0) > 0]


def column_names_of(dataset_id: str) -> list[str]:
    status, detail = call("GET", f"/datasets/{dataset_id}")
    if status != 200:
        return []
    return [str(c.get("name")) for c in (detail or {}).get("columns", [])]


candidates = {d["dataset_id"]: column_names_of(d["dataset_id"]) for d in rows}
bases = [
    d for d in rows
    if {"open", "high", "low", "close"}.issubset(set(candidates[d["dataset_id"]]))
]
low_frequency = [
    d for d in rows
    if candidates[d["dataset_id"]]
    and ("calc_time" in candidates[d["dataset_id"]] or "create_time" in candidates[d["dataset_id"]])
]

if not bases:
    print("no OHLC dataset available to use as a base; nothing to exercise")
    sys.exit(1)

# The join checks are only meaningful when the two datasets actually overlap in time:
# a base from a different period would still "pass" an inner-join-is-smaller assertion,
# vacuously, with zero matched rows. So the partner is chosen first and the base is then
# the OHLC dataset whose range covers it.
partner = low_frequency[0] if low_frequency else None


def overlaps(base_row: dict[str, Any], partner_row: dict[str, Any]) -> bool:
    a_start, a_end = base_row.get("start_timestamp"), base_row.get("end_timestamp")
    b_start, b_end = partner_row.get("start_timestamp"), partner_row.get("end_timestamp")
    if None in (a_start, a_end, b_start, b_end):
        return True  # an unknown range is not a reason to exclude it
    return a_start <= b_end and b_start <= a_end


# Prefer a partner that some base actually covers; otherwise the join checks run against
# a pair with no shared timestamps and every "inner is smaller" assertion passes on zero
# matched rows without testing anything.
for candidate in low_frequency:
    if any(overlaps(d, candidate) for d in bases):
        partner = candidate
        break

# A base already at the top of the ladder has no legal resample target, and resampling is
# the first thing this driver exercises. The unfiltered list is kept only to report why.
targets = {d["dataset_id"]: resample_target(d) for d in bases}
resamplable = [d for d in bases if targets[d["dataset_id"]]]
if not resamplable:
    print(f"none of the {len(bases)} OHLC datasets can be resampled to a coarser timeframe")
    sys.exit(1)

overlapping = [d for d in resamplable if overlaps(d, partner)] if partner is not None else []
base = (overlapping or resamplable)[0]
base_id = base["dataset_id"]
base_timeframe = base.get("timeframe")
agg_timeframe = targets[base_id] or "4h"
target_seconds = TIMEFRAME_SECONDS[agg_timeframe]
print(
    f"base: {base['name']} ({base_id}) {base['row_count']} rows @ "
    f"{base_timeframe or 'an undeclared timeframe'}, resampling to {agg_timeframe}"
)
if partner is not None:
    print(f"join: {partner['name']} ({partner['dataset_id']}) {partner['row_count']} rows")
    if base not in overlapping:
        print("  (no overlapping base — the join checks will be vacuous)")
else:
    print("join: no funding/metrics dataset available")

# --------------------------------------------------------------- resampling

print("== resample ==")
status, body = merge({"base_dataset_id": base_id, "timeframe": agg_timeframe, "write": False})
report = (body or {}).get("report") or {}
check(f"{agg_timeframe} resample previews", status == 200,
      f"status={status} {json.dumps(body)[:160]}")
check("resample shrinks the frame",
      report.get("rows_after", 0) < report.get("rows_before", 0),
      f"{report.get('rows_before')} -> {report.get('rows_after')}")
# The aggregation table lives in the resample step's own report, which the merge report
# nests under "resample" — the top level carries that step's row counts, not its table.
# This check read `report["aggregations"]`, a key that has never existed, and so printed
# `null` against a server that had sent the whole mapping.
aggregations = (report.get("resample") or {}).get("aggregations") or {}
# `open`→first and `close`→last are the pair that decides whether a bar's edges are honest
# — a `mean` open reports a price the bar never opened at. Both names are present whatever
# else the base carries, because `bases` selects on open/high/low/close.
check("resample reports the statistic it applied to each column",
      aggregations.get("open") == "first"
      and aggregations.get("close") == "last"
      and len(aggregations) >= 4,
      json.dumps(aggregations)[:160])
head = ((body or {}).get("preview") or {}).get("head") or []
stamps = [parse_stamp(row.get("open_time")) for row in head]
first_stamp = next((s for s in stamps if s is not None), None)
source_start = parse_stamp(base.get("start_timestamp"))
# `label="left", closed="left"` labels a bar with its opening instant, so the first bar
# begins at or before the first source observation and less than one interval before it.
# Labelling with the closing edge would put it strictly after, and both bounds are needed:
# `<= source` alone also passes a frame whose first bin merely happens to start exactly at
# the source. An endswith(":00:00") test would be wrong in the other direction — it fails
# on a 15-minute target, whose bars legitimately sit off the hour.
if first_stamp is not None and source_start is not None:
    lead = (source_start - first_stamp).total_seconds()
    check("the first bar is labelled by interval open",
          0 <= lead < target_seconds,
          f"source {source_start:%Y-%m-%d %H:%M}, first bar {first_stamp:%Y-%m-%d %H:%M}")
else:
    print("  (no comparable start timestamp; the bar-label check did not run)")

if base_timeframe:
    status, body = merge(
        {"base_dataset_id": base_id, "timeframe": base_timeframe, "write": False}
    )
    same = (body or {}).get("report") or {}
    check("resampling to the base's own timeframe is a no-op",
          status == 200 and same.get("rows_before") == same.get("rows_after"),
          f"{same.get('rows_before')} -> {same.get('rows_after')}")
else:
    print("  (base declares no timeframe; the no-op check does not apply)")

# ------------------------------------------------------------------- joins

if partner is not None:
    pid = partner["dataset_id"]
    prefix = "fund_"
    print("== join: backward alignment with a prefix ==")
    status, body = merge(
        {
            "base_dataset_id": base_id,
            "timeframe": agg_timeframe,
            "joins": [
                {"dataset_id": pid, "how": "left", "method": "backward", "prefix": prefix}
            ],
            "write": False,
        }
    )
    names = column_names(body)
    check("backward join previews", status == 200, f"status={status} {json.dumps(body)[:200]}")
    check("joined columns carry the prefix",
          any(n.startswith(prefix) for n in names),
          f"{[n for n in names if n.startswith(prefix)][:6]}")
    check("the base's own columns survive",
          {"open", "high", "low", "close"}.issubset(set(names)), f"{len(names)} columns")
    left_rows = ((body or {}).get("report") or {}).get("rows", 0)

    print("== join: exact alignment and inner vs left ==")
    status, exact = merge(
        {
            "base_dataset_id": base_id,
            "timeframe": agg_timeframe,
            "joins": [{"dataset_id": pid, "how": "left", "method": "exact"}],
            "write": False,
        }
    )
    exact_rows = ((exact or {}).get("report") or {}).get("rows", 0)
    check("exact alignment is not larger than backward",
          status == 200 and exact_rows <= left_rows,
          f"exact={exact_rows} backward={left_rows}")

    status, inner = merge(
        {
            "base_dataset_id": base_id,
            "timeframe": agg_timeframe,
            "joins": [{"dataset_id": pid, "how": "inner", "method": "backward"}],
            "write": False,
        }
    )
    inner_rows = ((inner or {}).get("report") or {}).get("rows", 0)
    check("inner join drops unmatched bars",
          status == 200 and inner_rows <= left_rows,
          f"inner={inner_rows} left={left_rows}")

    print("== join: a tolerance narrows the match ==")
    status, tight = merge(
        {
            "base_dataset_id": base_id,
            "timeframe": agg_timeframe,
            "joins": [
                {
                    "dataset_id": pid,
                    "how": "left",
                    "method": "backward",
                    "tolerance_seconds": 60,
                }
            ],
            "write": False,
        }
    )
    tight_rows = ((tight or {}).get("report") or {}).get("rows", 0)
    check("a 60s tolerance never adds rows",
          status == 200 and tight_rows <= left_rows,
          f"tolerance={tight_rows} unbounded={left_rows}")

    print("== join: refusals ==")
    status, body = merge(
        {
            "base_dataset_id": base_id,
            "timeframe": agg_timeframe,
            "joins": [{"dataset_id": base_id, "how": "left", "method": "backward"}],
            "write": False,
        }
    )
    check("joining a dataset to itself is refused", status == 422,
          f"status={status} {json.dumps(body)[:140]}")

    status, body = merge(
        {
            "base_dataset_id": base_id,
            "timeframe": agg_timeframe,
            # A ULID that cannot be in the catalog: the generator stamps the top 48 bits
            # with the current millisecond, so an all-zero timestamp means 1970. The id
            # this check used to carry was a real dataset from an earlier session, and a
            # store that still holds it turns "unknown id is a 404" into a false failure.
            "joins": [{"dataset_id": "0" * 26, "how": "left"}],
            "write": False,
        }
    )
    check("an unknown join dataset is a 404", status == 404,
          f"status={status} {json.dumps(body)[:140]}")

# ------------------------------------------------------------- column edits
#
# Every request here carries a timeframe, so a 422 can only mean what the check says it
# means. Without one the request would be refused for there being nothing to do at all,
# and "renaming an absent column is refused" would pass against a server that had stopped
# validating column operations entirely.

print("== column ops ==")
status, body = merge(
    {
        "base_dataset_id": base_id,
        "timeframe": agg_timeframe,
        "column_ops": [
            {"op": "rename", "column": "close", "new_name": "settle"},
            {"op": "drop", "column": "volume"},
        ],
        "write": False,
    }
)
names = column_names(body)
check("a rename reaches the preview", "settle" in names and "close" not in names, f"{names}")
check("a drop reaches the preview", "volume" not in names, f"{names}")

# The subject has to be a column the base does *not* have. This request named `close`,
# which every base carries by construction (`bases` selects on open/high/low/close), so it
# was an ordinary legal rename: the server answered 200 and this check — whose label says
# "absent" — reported a correct server as broken. The refusal is asserted by its message
# and not only by its status, because "the column ops were refused" is not the claim: the
# claim is that the *missing name* is what was refused.
status, body = merge(
    {
        "base_dataset_id": base_id,
        "timeframe": agg_timeframe,
        "column_ops": [{"op": "rename", "column": "not_a_column", "new_name": "nope"}],
        "write": False,
    }
)
check("renaming an absent column is refused",
      status == 422 and "missing column" in json.dumps(body),
      f"status={status} {json.dumps(body)[:140]}")

# ------------------------------------------------------------------ writing

print("== write ==")
joins: list[dict[str, Any]] = []
if partner is not None:
    joins = [
        {
            "dataset_id": partner["dataset_id"],
            "how": "left",
            "method": "backward",
            "prefix": "fund_",
        }
    ]
status, body = merge(
    {
        "base_dataset_id": base_id,
        "name": "E2E merged",
        "timeframe": agg_timeframe,
        "joins": joins,
        "column_ops": [{"op": "rename", "column": "close", "new_name": "settle"}],
        "write": True,
    }
)
saved = (body or {}).get("dataset") or {}
check("a writing merge returns 201", status == 201, f"status={status} {json.dumps(body)[:200]}")
check("the child is processed_data", saved.get("category") == "processed_data",
      str(saved.get("category")))
check("the child points at its parent", saved.get("parent_dataset_id") == base_id,
      str(saved.get("parent_dataset_id")))
check("the preview survives the write",
      bool(((body or {}).get("preview") or {}).get("columns")),
      f"{len(column_names(body))} columns")

child_id = saved.get("dataset_id")
if child_id:
    status, _ = call("GET", f"/datasets/{child_id}")
    check("the child is retrievable", status == 200, f"status={status}")
    s, raw, ctype = call_bytes(f"{BASE}/datasets/{child_id}/export?format=parquet")
    # `PAR1` is Parquet's magic, at both ends of the file. A 200 carrying an HTML error
    # page or an empty body would pass a status-code-only check.
    check("the child exports as parquet",
          s == 200 and raw[:4] == b"PAR1" and raw[-4:] == b"PAR1",
          f"status={s} type={ctype} bytes={len(raw)}")
    s, raw, _ = call_bytes(f"{BASE}/datasets/{child_id}/export?format=csv")
    header = raw.split(b"\r\n", 1)[0].split(b"\n", 1)[0].decode(errors="replace")
    check("the child exports as csv", s == 200 and header == ",".join(column_names(body)),
          f"status={s} header={header[:70]!r}")

    # The parent must be untouched: the whole point of forking rather than editing. The
    # label used to say "and still raw_data", which is not what this asserts and is not
    # true of a parent that is itself a derived frame — the check compares the parent's
    # category against the category it had before the write, and a base that was
    # `processed_data` on the way in has to still be `processed_data` on the way out.
    status, parent = call("GET", f"/datasets/{base_id}")
    check("the parent is unchanged, with the category it already had",
          status == 200 and (parent or {}).get("category") == base.get("category"),
          f"category={(parent or {}).get('category')}")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for item in FAILURES:
        print("  -", item)
    sys.exit(1)
print("all merge-endpoint checks passed")
