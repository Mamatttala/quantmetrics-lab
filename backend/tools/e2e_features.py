"""End-to-end exercise of the Part C feature endpoints over HTTP.

Not a unit test: it drives the running server exactly as the browser does, so it catches
contract drift the in-process TestClient cannot see (serialisation, status codes, response
envelopes) and — more importantly here — checks the one thing an in-process test with
hand-built contexts cannot: that the catalogue's `available` flag is honest about the data
the server actually holds.

The dataset is seeded by uploading a CSV, so the driver runs against a store that has
never been used. An uploaded file is also the worst case for this page: it declares no
timeframe and carries a placeholder asset class, and a feature page that only worked on
perfectly-labelled data would be a feature page that did not work. A declared-timeframe
dataset is then produced from it through the merge endpoint's resample — the only route
to a real 4h frame without a network fetch — and that is what exercises the duration-window
arithmetic (`4h` → 1 bar, `1d` → 6 bars) rather than refusing every window for the same
reason.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Any

# A driver run by hand inherits its console's encoding, which on this machine is cp1256
# and cannot encode the `—`/`→` it prints. `verify_all.py` sets PYTHONIOENCODING for the
# children it starts; this covers the way the header above says to run it directly, where
# a run whose every check passed ended in UnicodeEncodeError instead of a verdict.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8000/api"
FAILURES: list[str] = []


def _bars(stamps: list[datetime], *, start: float, drift: float, wobble: float) -> str:
    """A deterministic OHLCV frame as CSV, with every bar internally consistent.

    High and low are taken around the open and the close rather than guessed
    independently, so no row has a high below its close. A seeded bar that violates the
    relationship the range estimators assume would make the Parkinson and Garman-Klass
    checks in this driver measure the generator instead of the engine.
    """
    rows = ["open_time,open,high,low,close,volume"]
    for index, stamp in enumerate(stamps):
        open_price = round(start + drift * index + wobble * (index % 7), 4)
        close = round(start + 0.5 + drift * index + wobble * (index % 5), 4)
        high = round(max(open_price, close) + 1.0, 4)
        low = round(min(open_price, close) - 1.0, 4)
        volume = round(10 + (index % 11), 4)
        rows.append(
            f"{stamp.strftime('%Y-%m-%dT%H:%M:%SZ')},{open_price},{high},{low},{close},{volume}"
        )
    return "\n".join(rows) + "\n"


#: 130 hourly bars. Long enough that a 100-bar window has something to compute over,
#: short enough to upload instantly. `quote_volume` is deliberately absent, so the
#: weighted price source resolves to its typical-price branch and the driver can tell the
#: two branches apart by the reported definition rather than by the values.
HOURLY_CSV = _bars(
    [datetime(2024, 1, 1) + timedelta(hours=index) for index in range(130)],
    start=100.0,
    drift=0.37,
    wobble=0.11,
)

#: 40 daily bars: every bar at the same time of day, which is what makes `hour_of_day`
#: refused for a reason that comes from the data rather than from a missing label.
DAILY_CSV = _bars(
    [datetime(2024, 1, 1) + timedelta(days=index) for index in range(40)],
    start=50.0,
    drift=0.9,
    wobble=0.4,
)


def call(method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    """One request, via urllib, for a JSON response.

    Not httpx: on this machine a pooled httpx connection to uvicorn was reset before the
    request reached the server, which made a transport quirk look like a failing endpoint.
    The stdlib client opens a fresh connection per call, so a failure here is a failure of
    the API.
    """
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
            return error.code, {"raw": raw}


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(label)


def upload(name: str, body: str) -> str:
    """POST a CSV as multipart/form-data, built by hand to keep the driver stdlib-only."""
    boundary = "----quantmetrics-e2e-features-boundary"
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
        with urllib.request.urlopen(request, timeout=120) as response:
            created = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        print(f"upload failed: {error.code} {error.read().decode()[:300]}")
        sys.exit(1)
    return str(created["dataset_id"])


def entries(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The catalogue's entries, flattened out of their groups and keyed by id."""
    return {entry["id"]: entry for group in payload["groups"] for entry in group["features"]}


def catalog(dataset_id: str) -> dict[str, Any]:
    # `dataset_id`, not `datasetId`: this driver must send the URL the browser sends, and
    # the browser's was camelCase. Spelling it the backend's way here would have hidden the
    # defect — the point of driving the live server is that it cannot.
    status, body = call("GET", f"/features/catalog?dataset_id={dataset_id}")
    if status != 200 or not isinstance(body, dict):
        print(f"catalogue for {dataset_id} failed: {status} {json.dumps(body)[:300]}")
        sys.exit(1)
    return body


def compute(dataset_id: str, body: dict[str, Any]) -> tuple[int, Any]:
    return call("POST", f"/datasets/{dataset_id}/features", body)


def selection(feature_id: str, window: str | None = None) -> dict[str, Any]:
    return {"feature_id": feature_id, "window": window}


# ----------------------------------------------------------------------- seeding

print("== seeding ==")
hourly_id = upload("e2e_features_hourly.csv", HOURLY_CSV)
daily_id = upload("e2e_features_daily.csv", DAILY_CSV)
print(f"  hourly={hourly_id} daily={daily_id}")

# A declared-timeframe dataset, produced by the merge endpoint's resample. Without one,
# every duration window is refused for there being no bar length to divide by — correct,
# but it leaves that arithmetic untested.
status, merged = call(
    "POST",
    "/datasets/merge",
    {
        "base_dataset_id": hourly_id,
        "name": "E2E features 4h base",
        "timeframe": "4h",
        "joins": [],
        "write": True,
    },
)
four_hour_id = ((merged or {}).get("dataset") or {}).get("dataset_id") if status == 201 else None
check("a declared-timeframe dataset is produced by resampling",
      status == 201 and bool(four_hour_id),
      f"status={status} id={four_hour_id}"
      + ("" if four_hour_id else f" {json.dumps(merged)[:140]}"))

# ---------------------------------------------------------------------- catalogue

print("== catalogue ==")
hourly = catalog(hourly_id)
groups = [group["group"] for group in hourly["groups"]]
check("every group is served, in the specification's order",
      groups == ["price", "volume", "returns", "volatility", "indicators", "time", "exogenous"],
      str(groups))
check("the counts are consistent",
      hourly["total_count"] == hourly["available_count"] + sum(
          1 for group in hourly["groups"] for entry in group["features"] if not entry["available"]
      ),
      f"total={hourly['total_count']} available={hourly['available_count']}")
check("the catalogue is not empty", hourly["total_count"] >= 50, str(hourly["total_count"]))
check("an uploaded file reports no declared timeframe", hourly["timeframe"] is None,
      str(hourly["timeframe"]))

hourly_entries = entries(hourly)
bollinger = hourly_entries["bollinger"]
check("an entry carries the formula the info icon shows",
      bool(bollinger["formula"]) and bool(bollinger["detail"]),
      repr(bollinger["formula"][:70]))
check("baked-in constants are named in the column, not hidden",
      bollinger["columns"] == ["bb_upper_20_2", "bb_lower_20_2", "bb_width_20_2"],
      str(bollinger["columns"]))
check("a parameterless entry reports one warm-up and no windows",
      bollinger["warmup_bars"] == 19 and bollinger["windows"] == [],
      f"warmup={bollinger['warmup_bars']} windows={len(bollinger['windows'])}")

sma = hourly_entries["sma"]
sma_windows = {window["token"]: window for window in sma["windows"]}
check("a parameterised entry reports its warm-up per window, not once",
      sma["warmup_bars"] is None
      and sma_windows["20"]["bars"] == 20
      and sma_windows["20"]["warmup_bars"] == 19,
      f"top={sma['warmup_bars']} window20={json.dumps(sma_windows.get('20'))[:110]}")
check("every window that resolves has a warm-up at least one bar under its length",
      all(
          window["bars"] is None or window["warmup_bars"] >= window["bars"] - 1
          for group in hourly["groups"]
          for entry in group["features"]
          for window in entry["windows"]
      ))
check("an unavailable window says why",
      all(
          window["available"] or window["reason"]
          for group in hourly["groups"]
          for entry in group["features"]
          for window in entry["windows"]
      ))

check("an interval feature is unavailable without a declared timeframe",
      hourly_entries["realized_vol"]["available"] is False
      and "no timeframe" in (hourly_entries["realized_vol"]["reason"] or ""),
      str(hourly_entries["realized_vol"]["reason"])[:90])
check("a bar-count feature survives a missing timeframe",
      hourly_entries["sma"]["available"] is True)
check("time of day is offered where the bars move off one clock time",
      hourly_entries["hour_of_day"]["available"] is True,
      str(hourly_entries["hour_of_day"]["reason"])[:90])

daily_entries = entries(catalog(daily_id))
check("time of day is refused where every bar sits at the same hour",
      daily_entries["hour_of_day"]["available"] is False
      and "same time of day" in (daily_entries["hour_of_day"]["reason"] or ""),
      str(daily_entries["hour_of_day"]["reason"])[:90])
check("calendar features need only a date, not a declared timeframe",
      daily_entries["month_end"]["available"] is True
      and daily_entries["days_to_quarter_end"]["available"] is True)

status, _ = call("GET", "/features/catalog?dataset_id=NOPE")
check("an unknown dataset is a 404", status == 404, f"status={status}")
status, _ = call("GET", "/features/catalog")
check("the catalogue requires a dataset id", status == 422, f"status={status}")

if four_hour_id:
    four = catalog(four_hour_id)
    four_entries = entries(four)
    roc_windows = {window["token"]: window for window in four_entries["roc"]["windows"]}
    check("an interval window resolves against the declared bar length",
          four["timeframe"] == "4h" and roc_windows["4h"]["bars"] == 1,
          f"timeframe={four['timeframe']} 4h={json.dumps(roc_windows.get('4h'))[:110]}")
    vol_windows = {window["token"]: window for window in four_entries["realized_vol"]["windows"]}
    check("a one-bar volatility window is refused rather than computed",
          vol_windows["4h"]["bars"] == 1 and vol_windows["4h"]["available"] is False
          and "at least 2" in (vol_windows["4h"]["reason"] or ""),
          json.dumps(vol_windows.get("4h"))[:130])
    check("a longer volatility window is available on the same dataset",
          vol_windows["1d"]["bars"] == 6 and vol_windows["1d"]["available"] is True,
          json.dumps(vol_windows.get("1d"))[:130])
    check("annualised volatility is available once a bar length is declared",
          four_entries["annualized_vol"]["available"] is True)

check("annualised volatility is refused without a declared timeframe",
      hourly_entries["annualized_vol"]["available"] is False
      and "bar length" in (hourly_entries["annualized_vol"]["reason"] or ""),
      str(hourly_entries["annualized_vol"]["reason"])[:90])

# ------------------------------------------------------- availability is not a lie

print("== every feature the catalogue offers can actually be computed ==")
#
# The strongest check in this driver, and one no in-process test can make. The catalogue
# decides `available` from the dataset's columns, timeframe and asset class; the engine
# decides whether it can compute from the same facts, in a different module. If the two
# disagree, the page offers a checkbox that can only produce an error. Selecting every
# offered feature at once and requiring a 200 is what proves they agree — and it runs
# every computer in the catalogue against real data in a single pass.
offered: list[dict[str, Any]] = []
expected_columns = 0
expected_carried: list[str] = []
for group in hourly["groups"]:
    for entry in group["features"]:
        if not entry["available"]:
            continue
        window: str | None = None
        if entry["param"] is not None:
            usable = next((item for item in entry["windows"] if item["available"]), None)
            if usable is None:
                continue
            window = usable["token"]
        offered.append(selection(entry["id"], window))
        # A carried entry generates nothing — it *is* the dataset's own column, kept. Its
        # name belongs in `carried`, not in `generated_columns`, so counting it here would
        # make the column count disagree with an engine that is doing exactly the right
        # thing. On this dataset `volume` is the one such entry.
        if entry["kind"] == "carried":
            expected_carried.extend(entry["columns"])
        else:
            expected_columns += len(entry["columns"])

status, body = compute(hourly_id, {"selections": offered, "write": False})
report = (body or {}).get("report") if isinstance(body, dict) else None
report = report or {}
check(f"all {len(offered)} offered features compute in one request", status == 200,
      f"status={status}" + (f" {json.dumps(body)[:200]}" if status != 200 else ""))
check("that preview stored nothing", (body or {}).get("dataset") is None)
check("every generated column is reported",
      len(report.get("generated_columns", [])) == expected_columns,
      f"reported={len(report.get('generated_columns', []))} expected={expected_columns}")
check("a carried column is reported as carried, not as generated",
      sorted(report.get("carried", [])) == sorted(expected_carried) and bool(expected_carried),
      f"reported={report.get('carried')} expected={expected_carried}")
check("no row was dropped",
      report.get("rows_before") == report.get("rows_after") == 130,
      f"{report.get('rows_before')} -> {report.get('rows_after')}")

# ------------------------------------------------------------------------ compute

print("== preview and write ==")
before = call("GET", "/datasets")[1]
status, body = compute(hourly_id, {"selections": [selection("sma", "20")], "write": False})
after = call("GET", "/datasets")[1]
check("a preview returns 200 with no dataset", status == 200 and body["dataset"] is None,
      f"status={status}")
check("a preview stores nothing", len(after) == len(before),
      f"{len(before)} -> {len(after)}")
check("the preview carries the new column",
      "sma_20" in [column["name"] for column in body["preview"]["columns"]],
      str([column["name"] for column in body["preview"]["columns"]]))
check("the report resolves the price source and states its definition",
      body["report"]["price_source"]["source"] == "close"
      and bool(body["report"]["price_source"]["definition"]),
      json.dumps(body["report"]["price_source"])[:130])

status, body = compute(hourly_id, {"selections": [selection("sma", "20")]})
child = (body or {}).get("dataset") or {}
report = (body or {}).get("report") or {}
check("a write returns 201 and a new dataset", status == 201 and bool(child.get("dataset_id")),
      f"status={status} id={child.get('dataset_id')}")
check("the child is processed_data with the parent recorded",
      child.get("category") == "processed_data" and child.get("parent_dataset_id") == hourly_id,
      f"category={child.get('category')} parent={child.get('parent_dataset_id')}")
check("the child kept every row", child.get("row_count") == 130, str(child.get("row_count")))
check("the report counts the warm-up and dates the first usable row",
      report.get("warmup_bars") == 19 and report.get("usable_from") is not None,
      f"warmup={report.get('warmup_bars')} from={report.get('usable_from')}")
check("the report carries a selection hash",
      len(report.get("selection_hash") or "") == 64,
      str(report.get("selection_hash"))[:20])

provenance = child.get("provenance") or {}
check("the child records the operation, the source and the choices",
      provenance.get("operation") == "features"
      and provenance.get("derived_from") == hourly_id
      and provenance.get("feature_price_source") == "close"
      and bool(provenance.get("feature_selection_hash")),
      json.dumps(provenance)[:220])
check("the upload's own note is inherited rather than dropped",
      provenance.get("uploaded_filename") == "e2e_features_hourly.csv",
      str(provenance.get("uploaded_filename")))

parent_status, parent = call("GET", f"/datasets/{hourly_id}")
check("the parent is left untouched",
      parent_status == 200
      and "sma_20" not in [column["name"] for column in parent.get("columns", [])],
      str([column["name"] for column in parent.get("columns", [])]))

child_id = child["dataset_id"]
stored_status, stored = call("GET", f"/datasets/{child_id}/preview?head=25")
check("the column is in the stored frame, not only in the preview",
      stored_status == 200 and "sma_20" in [c["name"] for c in stored.get("columns", [])],
      str([c["name"] for c in stored.get("columns", [])]))

page_status, page = call("GET", f"/datasets/{child_id}/page?offset=0&limit=30")
rows = (page or {}).get("rows") or []
check("the warm-up gap is left visible as empty cells, not filled",
      page_status == 200 and len(rows) > 20
      and rows[0]["sma_20"] is None and rows[18]["sma_20"] is None
      and rows[19]["sma_20"] is not None,
      f"row0={rows[0].get('sma_20') if rows else None} "
      f"row18={rows[18].get('sma_20') if len(rows) > 18 else None} "
      f"row19={rows[19].get('sma_20') if len(rows) > 19 else None}")

print("== keeping only the selection ==")
status, body = compute(
    hourly_id, {"selections": [selection("sma", "20")], "keep_only_selected": True}
)
projected = (body or {}).get("dataset") or {}
columns = [column["name"] for column in projected.get("columns", [])]
check("only the timestamp and the new column are kept",
      status == 201 and columns == ["open_time", "sma_20"], str(columns))
check("the report names what was dropped",
      set(((body or {}).get("report") or {}).get("dropped_columns", []))
      == {"open", "high", "low", "close", "volume"},
      str(((body or {}).get("report") or {}).get("dropped_columns")))

print("== many features, a custom window, the other price source, a carried column ==")
status, body = compute(
    hourly_id,
    {
        "selections": [
            selection("sma", "20"),
            selection("ema", "10"),
            selection("rsi", "14"),
            selection("bollinger"),
            selection("macd"),
            selection("parkinson", "5"),
            selection("atr", "14"),
            selection("obv"),
            selection("day_of_week"),
            selection("price_lag", "1"),
        ]
    },
)
report = (body or {}).get("report") or {}
check("ten features compute together", status == 201 and len(report.get("added", [])) == 10,
      f"status={status} added={len(report.get('added', []))}")
check("the reported warm-up is the longest of them, not a sum",
      report.get("warmup_bars") == 33, str(report.get("warmup_bars")))
check("the generated columns are reported in selection order",
      report.get("generated_columns", [])[:3] == ["sma_20", "ema_10", "rsi_14"]
      and "macd_signal_12_26_9" in report.get("generated_columns", []),
      str(report.get("generated_columns", []))[:150])

status, body = compute(hourly_id, {"selections": [selection("sma", "37")]})
check("a window the ticks do not list is accepted",
      status == 201
      and ((body or {}).get("report") or {}).get("generated_columns") == ["sma_37"],
      f"status={status} "
      f"{((body or {}).get('report') or {}).get('generated_columns')}")

status, body = compute(hourly_id, {"selections": [selection("sma", "10"), selection("sma", "11")]})
pairs = ((body or {}).get("report") or {}).get("collinearity", [])
check("near-identical columns are reported rather than left to be discovered",
      status == 201 and any({pair["a"], pair["b"]} == {"sma_10", "sma_11"} for pair in pairs),
      json.dumps(pairs)[:160])

status, body = compute(
    hourly_id, {"selections": [selection("price_lag", "1")], "price_source": "weighted"}
)
report = (body or {}).get("report") or {}
source = report.get("price_source") or {}
check("the weighted price source is resolved and its definition reported",
      status == 201 and source.get("source") == "weighted"
      and "typical price" in source.get("definition", ""),
      json.dumps(source)[:170])
check("which branch was taken is also written as a note",
      any("typical price" in note for note in report.get("notes", [])),
      json.dumps(report.get("notes", []))[:200])

status, body = compute(hourly_id, {"selections": [selection("volume")]})
report = (body or {}).get("report") or {}
check("a carried column passes through untouched",
      status == 201 and report.get("carried") == ["volume"]
      and report.get("generated_columns") == [],
      f"carried={report.get('carried')} generated={report.get('generated_columns')}")

if four_hour_id:
    status, body = compute(
        four_hour_id,
        {"selections": [selection("realized_vol", "1d"), selection("annualized_vol", "1d")]},
    )
    check("a duration window and an annualisation factor compute on a declared timeframe",
          status == 201, f"status={status}"
          + (f" {json.dumps(body)[:200]}" if status != 201 else ""))

# ----------------------------------------------------------------------- refusals

print("== refusals ==")


def refuses(label: str, body: dict[str, Any], fragment: str, dataset: str | None = None) -> None:
    status, payload = compute(dataset or hourly_id, body)
    message = ((payload or {}).get("error") or {}).get("message", "")
    check(label, status == 422 and fragment in message,
          f"status={status} msg={message[:100]}")


refuses("an empty selection is refused", {"selections": []}, "No features")
refuses("an unknown feature is refused",
        {"selections": [selection("not_a_feature")]}, "no feature")
refuses("the same feature and window twice is refused",
        {"selections": [selection("sma", "20"), selection("sma", "20")]}, "twice")
refuses("an interval window is refused without a declared timeframe",
        {"selections": [selection("realized_vol", "4h")]}, "no timeframe")
refuses("annualised volatility is refused without a declared timeframe",
        {"selections": [selection("annualized_vol", "7d")]}, "bar length")
refuses("a window below the minimum is refused",
        {"selections": [selection("sma", "0")]}, "at least")
refuses("a window that is not a number is refused",
        {"selections": [selection("sma", "twenty")]}, "count of candles")
refuses("a generated column that already exists is refused",
        {"selections": [selection("sma", "20")]}, "already exist", dataset=child_id)
refuses("time of day is refused where the bars never move off one clock time",
        {"selections": [selection("hour_of_day")]}, "same time of day", dataset=daily_id)
refuses("a feature needing a column this dataset does not have is refused",
        {"selections": [selection("last_funding_rate")]}, "not available on this dataset")

status, _ = compute(hourly_id, {"selections": [selection("sma", "20")], "price_source": "vwap"})
check("an unknown price source is rejected by the request model", status == 422,
      f"status={status}")
status, _ = compute("NOPE", {"selections": [selection("sma", "20")]})
check("features on an unknown dataset is a 404", status == 404, f"status={status}")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for item in FAILURES:
        print("  -", item)
    sys.exit(1)
print("all feature-endpoint checks passed")
