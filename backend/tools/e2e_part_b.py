"""End-to-end exercise of the Part B endpoints over HTTP.

Not a unit test: it drives the running server exactly as the browser does, so it
catches contract drift the in-process TestClient cannot see (serialisation, status
codes, response envelopes).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any

# A driver run by hand inherits its console's encoding, and this machine's is cp1256 —
# which cannot encode the `—` in a check's detail or the `→` in e2e_ui's labels, so a run
# whose every check had passed ended in UnicodeEncodeError instead of a verdict.
# `verify_all.py` sets PYTHONIOENCODING for the children it starts; this covers a driver
# started on its own, which is how the file's own header says to run it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8000/api"
# The dataset every cleaning check below runs against. Not a hardcoded id: the driver
# uploads its own frame (`DEFECTIVE_CSV`) in the seeding block further down and this is
# bound to the result, so the defects the checks rely on are the defects the fixture has
# rather than whatever a dataset left in the store happens to contain.
DEFECTIVE = ""
FAILURES: list[str] = []

# A frame carrying one of every defect the cleaning checks look for: an exact duplicate
# row, a duplicated timestamp with a different value, an empty cell, a gap in the hourly
# sequence, a non-positive price, an extreme value, and a bar whose high is below its low.
# Held here as text so the driver can seed a store that has never been used, instead of
# depending on a dataset id left behind by an earlier session.
DEFECTIVE_CSV = """\
open_time,open,high,low,close,volume
2024-01-01T00:00:00Z,100,101,99,100.5,10
2024-01-01T01:00:00Z,100.5,102,100,101.5,11
2024-01-01T02:00:00Z,101.5,103,101,102.5,12
2024-01-01T03:00:00Z,102.5,104,102,103.5,13
2024-01-01T04:00:00Z,103.5,105,103,104.5,14
2024-01-01T04:00:00Z,103.5,105,103,104.5,14
2024-01-01T05:00:00Z,104.5,106,104,105.5,
2024-01-01T06:00:00Z,105.5,107,105,-1,16
2024-01-01T06:00:00Z,105.5,107,105,105.6,16.5
2024-01-01T08:00:00Z,107,109,106,108,18
2024-01-01T09:00:00Z,108,110,107,109,19
2024-01-01T10:00:00Z,109,111,108,99999,20
2024-01-01T11:00:00Z,110,112,109,111,21
2024-01-01T12:00:00Z,111,113,110,112,22
2024-01-01T13:00:00Z,112,111,113,113,23
"""


def call(method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    """One request, via urllib, for a JSON response.

    Not httpx: on this machine a pooled httpx connection to uvicorn was reset before
    the request reached the server (nothing in the access log), which made a transport
    quirk look like a failing endpoint. The stdlib client opens a fresh connection per
    call, so a failure here is a failure of the API.
    """
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read().decode()
            return response.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as error:
        raw = error.read().decode()
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, {"raw": raw}


def call_bytes(url: str) -> tuple[int, bytes, str]:
    """One GET whose body is not JSON — the parquet export.

    A binary body cannot go through `call`: decoding it as UTF-8 raised
    `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfa`, which an HTTPError
    handler would then dress up as a failed request. The endpoint was fine; the reader
    was. Returns (status, body, content-type).
    """
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers.get("Content-Type", "")


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(label)


def clean(payload: dict[str, Any]) -> tuple[int, Any]:
    return call("POST", f"/datasets/{DEFECTIVE}/clean", payload)


def upload(name: str, body: str) -> tuple[int, Any]:
    """POST a CSV as multipart/form-data, built by hand to keep the driver stdlib-only."""
    boundary = "----quantmetrics-e2e-boundary"
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
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode() or "{}")


# The frame every check below runs against is uploaded here rather than found in the
# store. A check that depends on a defect must not also depend on *finding* one: the
# earlier version reused a dataset id left behind by another session and verified only
# that its columns looked like OHLCV, so when that dataset turned out to hold a repeated
# timestamp but no exactly-repeated row, "drop_exact removes rows that keep retains"
# failed while both policies were behaving correctly. One upload makes the fixture the
# driver's own — and the row counts below known rather than assumed.
_status, _created = upload("e2e_defective.csv", DEFECTIVE_CSV)
if _status != 201 or not isinstance(_created, dict) or "dataset_id" not in _created:
    print(f"could not seed the defective dataset: {_status} {json.dumps(_created)[:300]}")
    sys.exit(1)
DEFECTIVE = str(_created["dataset_id"])
check("the defective frame is stored whole",
      _created.get("row_count") == 15, f"row_count={_created.get('row_count')}")
print(f"seeded {DEFECTIVE} ({_created.get('row_count')} rows)")


print("== cleaning: every duplicate policy ==")
duplicate_rows: dict[str, int] = {}
for policy in ("keep", "drop_exact", "keep_first_by_time", "keep_last_by_time"):
    status, body = clean({"duplicates": policy, "write": False})
    rows = body.get("report", {}).get("rows_after")
    steps = [s["step"] for s in body.get("report", {}).get("steps", [])]
    if isinstance(rows, int):
        duplicate_rows[policy] = rows
    check(f"duplicates={policy}", status == 200 and rows is not None,
          f"status={status} rows_after={rows} steps={steps}")

# The fixture is 15 rows and holds two different kinds of repetition: one row repeated
# exactly (the 04:00 bar, identical in every column) and one timestamp repeated with
# different values (the 06:00 bar, a provider correction). Those are the two defects the
# two families of policy repair, so the expected row counts are exact arithmetic rather
# than an inequality — which is the difference between a policy that ran and a policy that
# returned the frame it was given.
DUPLICATE_OUTCOMES = {
    "keep": 15,
    "drop_exact": 14,  # the duplicated 04:00 row, and only it
    "keep_first_by_time": 13,  # one row at 04:00 and one at 06:00
    "keep_last_by_time": 13,
}
check("every duplicate policy removes exactly the rows it is defined to remove",
      duplicate_rows == DUPLICATE_OUTCOMES,
      f"got={duplicate_rows} expected={DUPLICATE_OUTCOMES}")

print("== cleaning: every missing-value policy ==")
for policy in ("keep", "ffill", "bfill", "interpolate_time", "zero", "drop_rows"):
    status, body = clean({"missing": policy, "write": False})
    report = body.get("report", {})
    step = next((s for s in report.get("steps", []) if s["step"] == "missing"), None)
    check(f"missing={policy}", status == 200,
          f"status={status} rows {report.get('rows_before')}->{report.get('rows_after')} "
          f"filled={step.get('filled') if step else None}")

print("== cleaning: every outlier policy ==")
for policy in ("keep", "winsorize", "remove", "iqr"):
    status, body = clean(
        {"outliers": policy, "outlier_columns": ["close", "volume"], "write": False}
    )
    report = body.get("report", {})
    step = next((s for s in report.get("steps", []) if s["step"] == "outliers"), None)
    fences = list((step or {}).get("fences", {}).keys())
    check(f"outliers={policy}", status == 200,
          f"status={status} rows {report.get('rows_before')}->{report.get('rows_after')} "
          f"fences={fences} changed={(step or {}).get('cells_changed')}")

print("== cleaning: sort, and combined plan, and write ==")
status, body = clean({"sort_by_time": True, "duplicates": "keep_first_by_time", "write": False})
order = [s["step"] for s in body.get("report", {}).get("steps", [])]
check("sort + dedup (ordering)", status == 200 and order == ["sort", "duplicates"], f"{order}")

status, body = clean(
    {
        "duplicates": "drop_exact",
        "missing": "ffill",
        "outliers": "winsorize",
        "outlier_columns": ["close"],
        "sort_by_time": True,
        "name": "E2E cleaned",
        "write": True,
    }
)
saved = body.get("dataset") or {}
check("combined plan writes a dataset", status == 201 and bool(saved.get("dataset_id")),
      f"status={status} id={saved.get('dataset_id')} rows={saved.get('row_count')} "
      f"parent={saved.get('parent_dataset_id')}")
check("child is processed_data", saved.get("category") == "processed_data",
      str(saved.get("category")))
check("child records the fence in provenance",
      any("outlier" in key or "fence" in key for key in (saved.get("provenance") or {})),
      json.dumps(saved.get("provenance") or {})[:220])
preview = body.get("preview") or {}
head, tail = len(preview.get("head", [])), len(preview.get("tail", []))
check("preview has at most 5 head and 5 tail", head <= 5 and tail <= 5,
      f"head={head} tail={tail} rows={preview.get('row_count')}")

if saved.get("dataset_id"):
    status, _ = call("GET", f"/datasets/{saved['dataset_id']}")
    check("child is retrievable", status == 200, f"status={status}")
    export_status, body, content_type = call_bytes(
        f"{BASE}/datasets/{saved['dataset_id']}/export?format=parquet"
    )
    # `PAR1` is the Parquet magic, at both ends of the file. A 200 carrying an HTML
    # error page or an empty body would pass a status-code-only check.
    check("child exports as parquet",
          export_status == 200 and body[:4] == b"PAR1" and body[-4:] == b"PAR1",
          f"status={export_status} type={content_type} bytes={len(body)}")
    csv_status, csv_body, _ = call_bytes(
        f"{BASE}/datasets/{saved['dataset_id']}/export?format=csv"
    )
    # The header has to name the dataset's own columns, in order. Checking only the
    # status would pass an export of the wrong dataset.
    expected_header = ",".join(column["name"] for column in preview.get("columns", []))
    csv_header = csv_body.split(b"\r\n", 1)[0].split(b"\n", 1)[0].decode(errors="replace")
    check("child exports as csv",
          csv_status == 200 and csv_header == expected_header,
          f"status={csv_status} header={csv_header[:70]!r}")

print("== cleaning: refusals ==")
# 422, not 400: every deliberate refusal in this application is a `ValidationFailure`,
# whose status code is 422 (`app/core/errors.py`). Nothing maps to 400. A check written
# against 400 passes only when the server is broken.
status, body = clean({"outliers": "winsorize", "outlier_columns": [], "write": False})
check("outliers without columns is refused",
      status == 422 and body.get("error", {}).get("code") == "validation_failure",
      f"status={status} hint={(body.get('error', {}).get('details') or {}).get('hint', '')[:70]}")

status, body = clean({"outliers": "iqr", "outlier_columns": ["open_time"], "write": False})
check("outliers on the timestamp column is refused", status == 422,
      f"status={status} msg={body.get('error', {}).get('message', '')[:80]}")

status, body = clean({"outliers": "iqr", "outlier_columns": ["nope"], "write": False})
check("outliers on an unknown column is refused", status == 422,
      f"status={status} msg={body.get('error', {}).get('message', '')[:80]}")

status, body = clean({"lower_quantile": 0.9, "upper_quantile": 0.1, "write": False})
check("inverted quantiles are refused", status == 422, f"status={status}")

status, body = clean({"outliers": "keep", "write": False})
check("an empty plan changes nothing",
      body.get("report", {}).get("steps") == []
      and body.get("report", {}).get("rows_before") == body.get("report", {}).get("rows_after"),
      f"steps={body.get('report', {}).get('steps')}")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for item in FAILURES:
        print("  -", item)
    sys.exit(1)
print("all clean-endpoint checks passed")
