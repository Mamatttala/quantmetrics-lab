"""Module D's HTTP surface: the job handshake and the lineage it leaves behind.

Two properties are under test here and neither of them is statistical — the statistics are
covered in `test_regime_engine.py`. What is checked here is the pair of promises the
specification makes about *how* a detection runs:

**It does not block the server.** `POST /api/regime/detect` returns 202 and a job id, not a
result. The fit happens in a background task and the caller polls. The point of asserting
the 202 specifically — rather than just "a successful response" — is that a 200 here would
mean the request waited for the fit, which is the failure mode the design exists to avoid.
A test that accepted any 2xx would pass on the implementation the spec forbids.

**It leaves a reproducible trace.** The output lands in `outputs` (never `processed_data`,
never a mutation of the source), carries `parent_dataset_id`, and records the whole request
in lineage as `parameters_used`. Without that, a saved column of regime labels is an
assertion with no argument attached — the thresholds that produced it would be gone.

These go through the real app rather than calling the service, for the same reason
`test_features_api.py` does: the frontend's types are hand-written, so a renamed JSON field
surfaces in the browser as `undefined` rather than as a compile error. Every field the page
reads is therefore asserted by name.
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.models.dataset import DatasetMeta
from app.models.enums import AssetClass, DatasetCategory, Provider, Timeframe
from app.storage import parquet_store
from app.storage.catalog import Catalog

#: The exact output schema the specification states. Asserted as a tuple, in order, because
#: a downstream consumer reading these by position is a thing that will eventually happen.
REGIME_COLUMNS = (
    "open_time",
    "dominant_state",
    "prob_trending_up",
    "prob_trending_down",
    "prob_ranging",
    "prob_crisis",
    "confidence",
)

PROBABILITY_COLUMNS = REGIME_COLUMNS[2:6]


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    resolved = tmp_path / "regime-api-data"
    monkeypatch.setenv("QML_DATA_DIR", str(resolved))
    from app.config import get_settings

    get_settings.cache_clear()
    return resolved


@pytest.fixture
def client(data_dir: Path) -> Iterator[TestClient]:
    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    from app.config import get_settings

    get_settings.cache_clear()


@pytest.fixture
def app_catalog(data_dir: Path) -> Catalog:
    """The same catalog the app is using — same data dir, same SQLite file.

    Read directly rather than over HTTP because there is no lineage endpoint by design: a
    route that served lineage rows would be a second way to ask a question the dataset
    record already answers, and the two would eventually disagree.

    The path is taken from `Settings` rather than spelled out, so renaming the catalog file
    breaks the app and this fixture together instead of only the app.
    """
    from app.config import Settings

    catalog = Catalog(Settings(data_dir=data_dir).catalog_path)
    catalog.initialise()
    return catalog


def regime_frame(rows: int = 200, *, seed: int = 5, vol: float = 0.0015) -> pd.DataFrame:
    """A calm hourly OHLCV frame whose realized volatility sits near 0.14 annualised.

    Deliberately below the 0.35 crisis threshold. The default fixture in `conftest` moves
    ~0.5% per bar, which annualises to roughly 0.47 — every bar would be a crisis and the
    tests below would pass for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, vol, rows)))
    return pd.DataFrame(
        {
            "open_time": index,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.abs(rng.normal(1000, 100, rows)),
        }
    )


def register(
    data_dir: Path,
    frame: pd.DataFrame,
    *,
    name: str = "btc 1h",
    timeframe: Timeframe | None = Timeframe.H1,
    asset_class: AssetClass = AssetClass.CRYPTO_SPOT,
) -> str:
    """Write a canonical dataset the app can see, and return its id.

    Written straight through the store rather than uploaded, because an uploaded CSV
    declares no timeframe — and Module D *refuses* a dataset without one, since the crisis
    threshold is stated in annualised terms and there is no factor to annualise by. That
    refusal is a guardrail worth its own test (below), but it makes uploads useless as a
    subject for the other thirty assertions here. A Binance-fetched dataset is the realistic
    input anyway.

    The catalog is opened per call rather than shared: SQLite serialises writers on the
    file, and a long-lived connection held by the test thread while the app's own requests
    run on the portal thread is a threading question this test has no business raising.
    """
    from app.config import Settings

    catalog = Catalog(Settings(data_dir=data_dir).catalog_path)
    catalog.initialise()
    dataset_id = parquet_store.new_dataset_id()
    written = parquet_store.write_dataset(frame, data_dir, DatasetCategory.RAW, dataset_id)
    catalog.insert_dataset(
        DatasetMeta(
            dataset_id=dataset_id,
            name=name,
            category=DatasetCategory.RAW,
            source=Provider.BINANCE,
            asset_class=asset_class,
            symbol="BTCUSDT",
            market_type=None,
            timeframe=timeframe,
            timezone="UTC" if timeframe is not None else None,
            start_timestamp=written.start_timestamp,
            end_timestamp=written.end_timestamp,
            row_count=written.row_count,
            columns=written.columns,
            parent_dataset_id=None,
            checksum=written.checksum,
            storage_path=str(written.storage_path),
            created_at=datetime.now(UTC),
        )
    )
    return dataset_id


def upload(client: TestClient, frame: pd.DataFrame, name: str = "sample") -> str:
    """Register a frame through the real upload route and return its dataset id."""
    buffer = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))
    response = client.post(
        "/api/datasets/upload", files={"file": (f"{name}.csv", buffer, "text/csv")}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["dataset_id"])


def poll(client: TestClient, job_id: str, *, timeout: float = 120.0) -> dict[str, Any]:
    """Poll a job to a terminal state, as the frontend does.

    Real `sleep`, not a busy loop: the fit runs on the app's event loop, and a tight loop in
    the test thread would starve it. A deliberate `sleep` that lets the server work is the
    honest simulation of what the browser does.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach a terminal state within {timeout:.0f}s")


def detect(client: TestClient, dataset_id: str, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "dataset_id": dataset_id,
        "model_type": "hmm",
        "n_states": 3,
        "min_state_duration": 5,
        "crisis_threshold": 0.35,
        "min_train_bars": 60,
        "refit_every": 60,
        "volatility_window": 24,
        "write": True,
    }
    body.update(overrides)
    return dict(client.post("/api/regime/detect", json=body).json())


# ------------------------------------------------------------------- 5.6 polling


def test_detect_accepts_the_work_and_returns_a_job_rather_than_a_result(
    client: TestClient, data_dir: Path
) -> None:
    """202 with a handle. The fit has not happened and the response must not pretend it has."""
    dataset_id = register(data_dir, regime_frame())

    response = client.post(
        "/api/regime/detect",
        json={"dataset_id": dataset_id, "model_type": "hmm", "min_train_bars": 60,
              "refit_every": 60},
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "running"
    assert body["job_id"]
    assert body["poll"] == f"/api/jobs/{body['job_id']}"
    # The handle carries no result. A `report` key here would mean the request blocked.
    assert "report" not in body


def test_polling_reaches_a_succeeded_job_carrying_the_result(
    client: TestClient, data_dir: Path
) -> None:
    """The full client lifecycle: accept, poll, read the payload."""
    dataset_id = register(data_dir, regime_frame())
    accepted = detect(client, dataset_id)

    finished = poll(client, accepted["job_id"])

    assert finished["status"] == "succeeded", finished.get("error")
    result = finished["result"]
    report = result["report"]
    assert report["model_type"] == "hmm"
    assert report["rows"] == 200
    assert report["labelled_rows"] > 0, "a 200-bar frame with a 60-bar warm-up must label"
    # Two separate gaps, and conflating them would misstate how much of the series the
    # module could not use. The first is the observation warm-up — the volatility window —
    # which `warmup_bars` names. The second is the training window the first fit needs on
    # top of it, which is the request's own `min_train_bars`. Only after both does the first
    # bar get a label.
    assert report["warmup_bars"] == 24
    assert report["labelled_rows"] == report["rows"] - report["warmup_bars"] - 60
    # The counts account for every row, with the unlabelled prefix named `warmup` rather
    # than dropped. A tally that silently omitted 84 of 200 bars would make the state
    # shares in the dashboard add up to a picture of the series that the series does not
    # support. Which states are present is the model's business — a calm 200-bar series
    # genuinely contains no trend and no crisis, and asserting every state appears would be
    # asserting something about the sample rather than about the contract.
    counts = report["state_counts"]
    assert set(counts) <= {
        "trending_up", "trending_down", "ranging", "crisis", "transition", "warmup"
    }
    assert sum(counts.values()) == report["rows"]
    assert counts["warmup"] == report["rows"] - report["labelled_rows"]
    # The job knows which dataset it produced, so the UI can link straight to it.
    assert finished["output_dataset_id"] == result["dataset"]["dataset_id"]


def test_a_running_job_is_distinguishable_from_a_job_that_never_existed(
    client: TestClient, data_dir: Path
) -> None:
    """Two different questions, two different answers.

    Collapsing "still running" and "no such job" into one response — as a plain
    `{status: null}` or a 200 with an empty body would — turns a mistyped id into an
    infinite spinner. This test pins both halves of the distinction.
    """
    missing = client.get("/api/jobs/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "job_not_found"

    dataset_id = register(data_dir, regime_frame())
    accepted = detect(client, dataset_id)
    running = client.get(f"/api/jobs/{accepted['job_id']}")
    assert running.status_code == 200
    assert running.json()["status"] in {"running", "succeeded"}


def test_the_job_appears_in_the_recent_list(
    client: TestClient, data_dir: Path
) -> None:
    """The job registry is generic, so Module D's run shows up alongside everything else."""
    dataset_id = register(data_dir, regime_frame())
    accepted = detect(client, dataset_id)
    poll(client, accepted["job_id"])

    listed = client.get("/api/jobs", params={"limit": 25})
    assert listed.status_code == 200
    ids = [row["job_id"] for row in listed.json()]
    assert accepted["job_id"] in ids

    record = next(row for row in listed.json() if row["job_id"] == accepted["job_id"])
    assert record["kind"] == "regime_detect"
    assert record["dataset_id"] == dataset_id
    assert record["output_dataset_id"]


def test_a_failure_inside_the_job_is_reported_not_raised(
    client: TestClient, data_dir: Path
) -> None:
    """The request is over; there is no 422 left to attach the refusal to.

    A hybrid model with no carried indicator is refused by the engine. The refusal has to
    arrive through the job record, in the same envelope every other endpoint uses — the
    alternative is an exception escaping into the event loop, logged to a console nobody is
    reading, while the job sits at `running` forever.
    """
    dataset_id = register(data_dir, regime_frame())
    accepted = detect(client, dataset_id, model_type="hybrid", features=["log_return"])

    finished = poll(client, accepted["job_id"])

    assert finished["status"] == "failed"
    assert finished["error"]["code"], "the refusal must carry a code, not just prose"
    assert finished["error"]["message"]
    assert finished["output_dataset_id"] is None


def test_an_unknown_dataset_is_refused_before_any_job_exists(
    client: TestClient, data_dir: Path
) -> None:
    """404 on the POST, not a job that fails thirty seconds later.

    A caller who mistyped an id should not have to poll to find that out — and a job row
    recording a request that could never have run is noise in the registry.
    """
    response = client.post("/api/regime/detect", json={"dataset_id": "nope", "model_type": "hmm"})

    assert response.status_code == 404
    before = client.get("/api/jobs").json()
    assert [row for row in before if row["kind"] == "regime_detect"] == []


def test_an_invalid_request_is_refused_with_422(
    client: TestClient, data_dir: Path
) -> None:
    """Parameter bounds are the request model's job, and they fail as a 422 like everything else."""
    dataset_id = register(data_dir, regime_frame())
    response = client.post(
        "/api/regime/detect", json={"dataset_id": dataset_id, "n_states": 99}
    )
    assert response.status_code == 422


# ------------------------------------------------------------------ 5.7 lineage


def lineage_rows(app_catalog: Catalog, child_id: str) -> list[dict[str, Any]]:
    rows = app_catalog.conn.execute(
        "SELECT parent_id, operation, detail FROM lineage WHERE child_id = ?", (child_id,)
    ).fetchall()
    return [
        {"parent_id": row["parent_id"], "operation": row["operation"],
         "detail": json.loads(row["detail"])}
        for row in rows
    ]


def test_the_output_lands_in_outputs_under_a_new_dataset(
    client: TestClient, app_catalog: Catalog, data_dir: Path
) -> None:
    """`outputs`, with the source named as parent and the source itself untouched.

    The category matters beyond tidiness: `processed_data` is an input to further analysis
    and `outputs` is a product of it, and a regime frame is the latter. It is also the
    guardrail that this module reads a source and never rewrites one.
    """
    dataset_id = register(data_dir, regime_frame())
    before = parquet_store.verify_checksum(Path(app_catalog.conn.execute(
        "SELECT storage_path FROM datasets WHERE dataset_id = ?", (dataset_id,)
    ).fetchone()["storage_path"]))

    accepted = detect(client, dataset_id)
    finished = poll(client, accepted["job_id"])
    child_id = finished["result"]["dataset"]["dataset_id"]

    assert child_id != dataset_id
    child = app_catalog.get_dataset(child_id)
    assert child is not None
    assert child.category.value == "outputs"
    assert child.parent_dataset_id == dataset_id
    assert child.row_count == 200

    # The source is byte-identical. The whole storage design rests on this.
    source_path = Path(app_catalog.conn.execute(
        "SELECT storage_path FROM datasets WHERE dataset_id = ?", (dataset_id,)
    ).fetchone()["storage_path"])
    assert parquet_store.verify_checksum(source_path) == before


def test_the_saved_frame_carries_the_specified_schema(
    client: TestClient, app_catalog: Catalog, data_dir: Path
) -> None:
    """Column names, order, and the softness the specification requires.

    Hard 0/1 labels are forbidden outright: a downstream position sizer needs to know how
    much the model believed, and a thresholded column throws that away irreversibly.
    """
    dataset_id = register(data_dir, regime_frame())
    finished = poll(client, detect(client, dataset_id)["job_id"])
    child = app_catalog.get_dataset(finished["result"]["dataset"]["dataset_id"])

    frame = parquet_store.read_dataset(Path(child.storage_path))
    assert tuple(frame.columns) == REGIME_COLUMNS
    assert str(frame["open_time"].dtype).startswith("datetime64")

    labelled = frame[frame["dominant_state"].notna()]
    assert not labelled.empty
    probabilities = labelled[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
    # Not a hard classifier: the frame carries genuine ambiguity, not 0s and 1s.
    assert (probabilities > 0.0).sum() > len(probabilities)
    labelled_states = set(labelled["dominant_state"])
    assert labelled_states <= {
        "trending_up", "trending_down", "ranging", "crisis", "transition"
    }


def test_lineage_records_the_whole_request_and_the_fit_diagnostics(
    client: TestClient, app_catalog: Catalog, data_dir: Path
) -> None:
    """A regime column without its thresholds is an assertion with no argument attached."""
    dataset_id = register(data_dir, regime_frame())
    correlation = "regime"
    accepted = detect(
        client,
        dataset_id,
        model_type="gmm",
        n_states=3,
        crisis_threshold=0.4,
        min_state_duration=7,
        switching_cost_penalty=0.002,
    )
    finished = poll(client, accepted["job_id"])
    child_id = finished["result"]["dataset"]["dataset_id"]

    rows = lineage_rows(app_catalog, child_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["parent_id"] == dataset_id
    assert row["operation"] == correlation

    detail = row["detail"]
    assert detail["module_source"] == "regime"

    parameters = detail["parameters_used"]
    # Every knob, not a summary. Reproducing the run means replaying this dict verbatim.
    assert parameters["model_type"] == "gmm"
    assert parameters["n_states"] == 3
    assert parameters["crisis_threshold"] == 0.4
    assert parameters["min_state_duration"] == 7
    assert parameters["switching_cost_penalty"] == 0.002
    assert parameters["dataset_id"] == dataset_id

    diagnostics = detail["model_diagnostics"]
    for key in ("bic_latest_refit", "aic_latest_refit", "switch_count",
                "average_state_duration_bars", "net_value_add"):
        assert key in diagnostics, key
    # The metric is disclosed as what it is. Reporting a Net Value-Add without the
    # in-sample flag would read as out-of-sample evidence it does not have.
    assert diagnostics["in_sample"] is True
    assert "long in trending_up" in diagnostics["net_value_add_definition"]


def test_provenance_carries_the_headline_numbers(
    client: TestClient, app_catalog: Catalog, data_dir: Path
) -> None:
    """The dataset row's own provenance, readable without joining to lineage."""
    dataset_id = register(data_dir, regime_frame())
    finished = poll(client, detect(client, dataset_id)["job_id"])
    child = app_catalog.get_dataset(finished["result"]["dataset"]["dataset_id"])

    provenance = child.provenance
    assert provenance["module_source"] == "regime"
    assert provenance["regime_model"] == "hmm"
    assert provenance["crisis_threshold"] == "0.35"
    # Recorded as a string so a float column cannot lose its precision in the JSON blob.
    float(provenance["net_value_add"])


# ------------------------------------------------------------------- 5.6b options


def test_options_report_what_the_dataset_can_support(
    client: TestClient, data_dir: Path
) -> None:
    """Availability is decided server-side so a greyed-out control and a refusal agree."""
    dataset_id = register(data_dir, regime_frame())

    response = client.get("/api/regime/options", params={"dataset_id": dataset_id})
    assert response.status_code == 200
    body = response.json()

    assert body["dataset_id"] == dataset_id
    assert body["rows"] == 200
    assert body["timeframe"] == "1h"

    by_id = {feature["id"]: feature for feature in body["features"]}
    assert by_id["log_return"]["available"] is True
    assert by_id["realized_volatility"]["available"] is True
    assert by_id["volume_change"]["available"] is True
    # Carried, not computed — and absent here, so unavailable with a reason that says where
    # to get one rather than a bare `false`. The `kind` field is what lets the UI offer
    # "produce it in C5" for these and "this dataset has no volume" for the computed ones.
    assert by_id["adx"]["available"] is False
    assert by_id["adx"]["kind"] == "carried"
    assert "adx" in by_id["adx"]["reason"]
    assert by_id["log_return"]["kind"] == "computed"

    models = {model["id"]: model for model in body["models"]}
    assert models["hmm"]["available"] is True
    assert models["gmm"]["available"] is True
    # Rules has no feature requirement at all, which is the point of it: it is the one path
    # that still produces a crisis label on a dataset with nothing but a close column.
    assert models["rules"]["available"] is True
    assert models["hybrid"]["available"] is False
    assert models["hybrid"]["reason"]

    # 1h crypto bars over 365 days.
    assert body["annualisation"]["available"] is True
    assert body["annualisation"]["periods_per_year"] == pytest.approx(24 * 365)
    assert not [w for w in body["warnings"] if w["level"] == "error"]


def test_options_flag_a_short_dataset_without_blocking_it(
    client: TestClient, data_dir: Path
) -> None:
    """A caution the user may overrule, not a refusal.

    The first label needs `min_train_bars` usable rows before it, so a 100-row frame at the
    250-bar default produces a result of nothing but nulls — an outcome that looks like a
    success and contains no labels. Saying so is the difference between the two, but it is a
    `warning` rather than an `error` because lowering the warm-up is a legitimate choice.
    """
    dataset_id = register(data_dir, regime_frame(100), name="short")

    body = client.get("/api/regime/options", params={"dataset_id": dataset_id}).json()

    short = [w for w in body["warnings"] if w["code"] == "short_dataset"]
    assert len(short) == 1
    assert short[0]["level"] == "warning"
    assert not [w for w in body["warnings"] if w["level"] == "error"]


def test_options_refuse_an_unknown_dataset(
    client: TestClient, data_dir: Path
) -> None:
    assert client.get("/api/regime/options", params={"dataset_id": "nope"}).status_code == 404


def test_a_dataset_with_no_timeframe_is_refused_with_the_reason(
    client: TestClient, data_dir: Path
) -> None:
    """The annualisation guardrail, end to end through the real upload path.

    A crisis threshold of 35% is an *annualised* figure. Annualising a per-bar volatility
    needs the bar length, and an uploaded CSV does not declare one — so there is no factor
    to multiply by and the threshold silently means something else. This is a refusal rather
    than a default, because the wrong factor (252 against 8760, say) is a silent
    multiplicative error of 6x in the number that triggers a crisis label, not a missing
    value the user would notice.

    It surfaces in `options` as an `error`-level warning so the form can explain itself
    before Run is pressed, *and* from the run endpoint as a failed job — the two agree
    because both read the same function.
    """
    dataset_id = upload(client, regime_frame(), "no-timeframe")

    body = client.get("/api/regime/options", params={"dataset_id": dataset_id}).json()
    assert body["annualisation"]["available"] is False
    assert body["annualisation"]["periods_per_year"] is None
    assert body["timeframe"] is None
    errors = [w for w in body["warnings"] if w["level"] == "error"]
    assert [w["code"] for w in errors] == ["no_timeframe"]
    assert "annualised" in errors[0]["message"]

    finished = poll(client, detect(client, dataset_id)["job_id"])
    assert finished["status"] == "failed"
    assert finished["error"]["code"] == "validation_failure"


def test_a_write_false_run_computes_but_saves_nothing(
    client: TestClient, app_catalog: Catalog, data_dir: Path
) -> None:
    """The preview path. A user tuning thresholds should not fill the catalog with runs."""
    dataset_id = register(data_dir, regime_frame())
    before = len(app_catalog.list_datasets())

    finished = poll(client, detect(client, dataset_id, write=False)["job_id"])

    assert finished["status"] == "succeeded"
    assert finished["result"]["dataset"] is None
    assert finished["result"]["report"]["labelled_rows"] > 0
    assert finished["output_dataset_id"] is None
    assert len(app_catalog.list_datasets()) == before
