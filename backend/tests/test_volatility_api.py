"""Module E's HTTP surface: the job handshake, the artifact, and the refusals.

The statistics are covered in `test_volatility_eval.py` and `test_volatility_models.py`. What
is checked here is the pair of promises the specification makes about *how* a forecast runs,
plus the one property that is easier to lose over HTTP than anywhere else:

**It does not block the server.** `POST /api/volatility/forecast` returns 202 and a job id.
A walk-forward refits at every scored bar, and a hybrid refits a second model inside that
loop. A 200 here would mean the request waited for thousands of fits — and a test that
accepted any 2xx would pass on exactly the implementation the design forbids.

**The artifact is reproducible and does not rewrite its source.** The output lands in
`outputs` with the whole request in lineage. A volatility forecast is uninterpretable beside
a different distribution or a different train window; without the lineage row it is a column
of numbers with no argument attached.

**Nothing future reaches the store.** The forward path the fan chart draws is in the report,
not in the dataset. That split is deliberate — a dataset row in this application is an
observation — and it is asserted here, because "write the projection too, it's convenient"
is the kind of change that would look harmless.

These go through the real app rather than the service because the frontend's types are
hand-written: a renamed JSON field surfaces in the browser as `undefined`, not as a compile
error. Every field the page reads is asserted by name.
"""

from __future__ import annotations

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
from app.models.volatility import VOL_FORECAST_COLUMNS
from app.storage import parquet_store
from app.storage.catalog import Catalog

#: The schedule the tests below run on: 30 scored bars, refit once per bar.
#:
#: ``test_window`` is 30 because that is exactly ``MIN_SCORED_STEPS`` — the smallest protocol
#: that produces a metric at all, and therefore the one where a single unusable step turns a
#: model's row into an unscored one. Every count assertion in this file is about that edge.
#:
#: ``train_window`` is 180 rather than the module's own floor of 50, and that is a fact about
#: the models rather than a preference. Over a 50-bar window on this fixture's 259 returns,
#: EGARCH's fit is *unidentified*: ``β`` pins at ≈ 0.999 — a random walk in log-variance the
#: likelihood cannot pin down on 50 points — and the recursion then reports a one-step
#: variance around 3 500 times the largest squared return in its own window, from a fit whose
#: optimiser status is "converged". The service refuses those steps (see
#: ``MAX_FORECAST_VARIANCE_MULTIPLE``) and says so, which is the correct behaviour and is
#: tested separately; but a fixture in which the *ordinary* path diverges cannot also be the
#: fixture that checks the ordinary path's arithmetic. At 180 bars every model in the zoo
#: fits cleanly on all 30 steps, which is what these tests need and what the page's derived
#: default — half the series — gives a real user.
FAST = {"train_window": 180, "test_window": 30, "step_size": 1}


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    resolved = tmp_path / "vol-api-data"
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
    """The same catalog the app is using — same data dir, same SQLite file, opened per test."""
    from app.config import Settings

    catalog = Catalog(Settings(data_dir=data_dir).catalog_path)
    catalog.initialise()
    return catalog


def price_frame(rows: int = 260, *, seed: int = 11, vol: float = 0.004) -> pd.DataFrame:
    """An hourly OHLCV frame whose returns come from a GARCH(1,1) process.

    **The clustering is the point.** ``vol`` is the *unconditional* log volatility, and each
    bar's conditional variance is drawn from a recursion with ``α + β = 0.92`` — so shocks
    arrive in bursts, which is the one feature every volatility model in this module exists
    to capture, and the feature a fixture of i.i.d. normal returns does not have.

    On an i.i.d. series a GARCH model has nothing to estimate. The ARCH and GARCH terms trade
    off against each other along a ridge of near-equally-likely parameter values, and over a
    50-bar window — the smallest one this module will fit — arch's optimiser walks that ridge
    and returns a variance forecast of ``8.3e213`` from one window and exactly ``0.0`` from
    another, *while reporting ``convergence_flag = 0``*. That is a real behaviour, and
    ``_forecast_refusal`` in the service exists to catch it; but a fixture that triggers it on
    the ordinary path tests the guard on every run and leaves no configuration in which the
    metrics themselves are exercised. Returns that cluster are also the honest fixture: real
    return series cluster, and a walk-forward over one asks the models the question they were
    built to answer.

    The series is stationary by construction, so it passes the stationarity guard — which is
    what makes a failure of that guard in the tests below mean something. A frame whose close
    barely moves would also pass, for a reason that has nothing to do with the guard.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")

    persistence = 0.92
    alpha = 0.12
    beta = persistence - alpha
    omega = vol**2 * (1.0 - persistence)
    shocks = rng.normal(0.0, 1.0, rows)
    variance = np.empty(rows)
    variance[0] = vol**2
    for step in range(1, rows):
        shock = shocks[step - 1] * np.sqrt(variance[step - 1])
        variance[step] = omega + alpha * shock**2 + beta * variance[step - 1]

    close = 100.0 * np.exp(np.cumsum(shocks * np.sqrt(variance)))
    return pd.DataFrame(
        {
            "open_time": index,
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.abs(rng.normal(1000.0, 100.0, rows)),
        }
    )


def regime_frame(rows: int = 260, *, seed: int = 7) -> pd.DataFrame:
    """A Module D-shaped regime output: the observation instant and four posteriors.

    Constructed rather than produced by running Module D, because what is under test here is
    the *join* — whether the conditioning columns reach the hybrid legs aligned with the bars
    they describe — and re-running a hidden Markov model to obtain them would add a second
    module's failure modes to this file without testing anything more about this one.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")
    raw = rng.dirichlet([1.0, 1.0, 1.0, 0.5], rows)
    frame = pd.DataFrame(
        {
            "open_time": index,
            "dominant_state": rng.choice(
                ["trending_up", "trending_down", "ranging", "crisis"], rows
            ),
        }
    )
    for column, values in zip(
        ("prob_trending_up", "prob_trending_down", "prob_ranging", "prob_crisis"),
        raw.T,
        strict=True,
    ):
        frame[column] = values
    frame["confidence"] = raw.max(axis=1)
    return frame


def register(
    data_dir: Path,
    frame: pd.DataFrame,
    *,
    name: str = "btc 1h",
    category: DatasetCategory = DatasetCategory.RAW,
    timeframe: Timeframe | None = Timeframe.H1,
) -> str:
    """Write a dataset the app can see, and return its id.

    Written straight through the store rather than uploaded: an uploaded CSV declares no
    timeframe, and the forward path's timestamps then come from the observed spacing rather
    than from the declared bar length — a legitimate path, but not the one under test here.
    """
    from app.config import Settings

    catalog = Catalog(Settings(data_dir=data_dir).catalog_path)
    catalog.initialise()
    dataset_id = parquet_store.new_dataset_id()
    written = parquet_store.write_dataset(frame, data_dir, category, dataset_id)
    catalog.insert_dataset(
        DatasetMeta(
            dataset_id=dataset_id,
            name=name,
            category=category,
            source=Provider.BINANCE,
            asset_class=AssetClass.CRYPTO_SPOT,
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


def poll(client: TestClient, job_id: str, *, timeout: float = 300.0) -> dict[str, Any]:
    """Poll a job to a terminal state, as the frontend does.

    A real `sleep` rather than a busy loop: the walk-forward runs on the app's event loop, and
    a tight loop in the test thread would starve it — turning a slow test into a hung one.
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


def forecast(client: TestClient, dataset_id: str, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "dataset_id": dataset_id,
        "price_source": "close",
        "models": ["garch"],
        "distribution": "students_t",
        "mean_model": "constant",
        "p": 1,
        "o": 1,
        "q": 1,
        "lags": 3,
        "walk_forward": dict(FAST),
        "forecast_horizon": 6,
        "write": True,
    }
    body.update(overrides)
    return dict(client.post("/api/volatility/forecast", json=body).json())


def run_to_completion(client: TestClient, dataset_id: str, **overrides: Any) -> dict[str, Any]:
    accepted = forecast(client, dataset_id, **overrides)
    finished = poll(client, accepted["job_id"])
    assert finished["status"] == "succeeded", finished.get("error")
    return finished


# ------------------------------------------------------------------- acceptance


def test_forecast_accepts_the_work_and_returns_a_job_rather_than_a_result(
    client: TestClient, data_dir: Path
) -> None:
    """202 with a handle. No fit has happened and the response must not imply one has."""
    dataset_id = register(data_dir, price_frame())

    response = client.post(
        "/api/volatility/forecast",
        json={"dataset_id": dataset_id, "models": ["garch"], "walk_forward": dict(FAST)},
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "running"
    assert body["job_id"]
    assert body["poll"] == f"/api/jobs/{body['job_id']}"
    # The handle carries no result. A `report` key here would mean the request blocked until
    # the walk-forward finished, which for a hybrid over a real series is minutes.
    assert "report" not in body


def test_polling_reaches_a_succeeded_job_carrying_the_result(
    client: TestClient, data_dir: Path
) -> None:
    """The full client lifecycle: accept, poll, read the payload."""
    dataset_id = register(data_dir, price_frame())
    finished = run_to_completion(client, dataset_id)

    result = finished["result"]
    report = result["report"]

    assert report["dataset_id"] == dataset_id
    assert report["price_source"] == "close"
    assert report["return_column"] == "log_return"
    # Derived rather than read: 260 bars of close give 259 log returns, because the first
    # difference has no predecessor and is written as a leading NaN. Saying so here is what
    # keeps `return_rows != rows` from reading as an off-by-one in the engine.
    assert report["rows"] == 260
    assert report["derived_returns"] is True
    assert report["return_rows"] == 259

    assert report["walk_forward"]["train_window"] == FAST["train_window"]
    assert report["walk_forward"]["test_window"] == FAST["test_window"]
    assert report["walk_forward"]["n_scored"] == 30
    assert report["first_scored_time"] < report["last_scored_time"]

    # The forward path is a separate object from the scored run and has its own length.
    assert report["forecast_horizon"] == 6
    assert len(report["forward_paths"]) == 1
    path = report["forward_paths"][0]
    assert path["model"] == "garch"
    assert len(path["times"]) == 6
    assert len(path["variance"]) == 6
    assert len(path["volatility"]) == 6
    # Bands are per step, not one interval around the whole path.
    assert {band["label"] for band in path["bands"]} == {"5–95%", "25–75%"}
    assert all(len(band["low"]) == 6 for band in path["bands"])

    assert finished["output_dataset_id"] == result["dataset"]["dataset_id"]


def test_a_single_model_scores_over_thirty_steps(
    client: TestClient, data_dir: Path
) -> None:
    """The scored path really is scored — not silently unscored.

    ``MIN_SCORED_STEPS`` is 30, so a run producing 29 finite forecasts reports no metric at
    all and says why. The fast schedule used here produces exactly 30, which is the boundary;
    asserting a finite MQLIKE is what confirms the boundary is inclusive and that the fits
    converged rather than being counted and discarded.
    """
    dataset_id = register(data_dir, price_frame())
    report = run_to_completion(client, dataset_id)["result"]["report"]

    assert len(report["scores"]) == 1
    score = report["scores"][0]
    assert score["model"] == "garch"
    assert score["family"] == "simple"
    assert score["n_steps"] == 30
    assert score["n_finite"] == 30
    assert score["n_nan"] == 0
    assert score["n_inf"] == 0
    assert score["unscored_reason"] is None

    assert score["mqlike"] is not None and np.isfinite(score["mqlike"])
    assert score["rmse"] is not None and score["rmse"] >= 0.0
    assert score["mae"] is not None and score["mae"] >= 0.0
    assert score["mz_slope"] is not None and np.isfinite(score["mz_slope"])
    # The proxy is named in the artifact, because a reader who does not know it is a squared
    # return will read the near-zero R² as a broken model.
    assert report["actual_proxy"] == "squared log return"
    assert report["scale_factor"] == 100.0


def test_two_models_produce_a_pairwise_diebold_mariano_row(
    client: TestClient, data_dir: Path
) -> None:
    """One model cannot be compared to anything, and two must be.

    The comparison is asserted to have actually run — ``n_effective`` is the intersected
    finite mask, so a zero there would mean the test silently found no comparable steps.
    """
    dataset_id = register(data_dir, price_frame())
    report = run_to_completion(client, dataset_id, models=["garch", "egarch"])["result"]["report"]

    assert {score["model"] for score in report["scores"]} == {"garch", "egarch"}
    assert len(report["comparisons"]) == 1
    entry = report["comparisons"][0]
    assert {entry["model_a"], entry["model_b"]} == {"garch", "egarch"}
    assert entry["n_effective"] == 30
    assert 0.0 <= entry["p_value"] <= 1.0
    # `better` names the lower mean loss, so the page can write a sentence rather than leaving
    # a bare statistic for the reader to over-read.
    assert entry["better"] in {"garch", "egarch"}


# ------------------------------------------------------------------- the guards


def test_an_unknown_dataset_is_refused_before_any_job_exists(
    client: TestClient, data_dir: Path
) -> None:
    """404 on the POST, not a job that fails thirty seconds later."""
    response = client.post("/api/volatility/forecast", json={"dataset_id": "nope"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"
    jobs = client.get("/api/jobs").json()
    assert [row for row in jobs if row["kind"] == "volatility_forecast"] == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"models": []},
        {"models": ["garch", "garch"]},
        # EGARCH and the hybrids model an asymmetry; o=0 asks for a symmetric model under an
        # asymmetric name, which arch would accept and fit without complaint.
        {"models": ["egarch"], "o": 0},
        {"models": ["gjr_garch"], "o": 0},
        {"price_source": "   "},
    ],
)
def test_an_invalid_request_is_refused_with_422(
    client: TestClient, data_dir: Path, overrides: dict[str, Any]
) -> None:
    dataset_id = register(data_dir, price_frame())
    body: dict[str, Any] = {"dataset_id": dataset_id, "walk_forward": dict(FAST)}
    body.update(overrides)

    response = client.post("/api/volatility/forecast", json=body)

    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "validation_failure"


def test_an_impossible_walk_forward_is_refused_inside_the_job_with_the_numbers(
    client: TestClient, data_dir: Path
) -> None:
    """The refusal names the row count it needed and the one it had.

    It arrives through the job rather than the POST because the row count is not knowable
    until the file has been read — and *which* row count matters: log returns are one shorter
    than the frame, so a schedule validated against the frame would be accepted here and then
    fail inside the fit.
    """
    dataset_id = register(data_dir, price_frame(70))
    accepted = forecast(
        client, dataset_id, walk_forward={"train_window": 50, "test_window": 30, "step_size": 1}
    )

    finished = poll(client, accepted["job_id"])

    assert finished["status"] == "failed", finished
    assert finished["error"]["code"] == "validation_failure"
    message = finished["error"]["message"]
    assert "50" in message and "30" in message and "69" in message
    assert finished["output_dataset_id"] is None


def test_a_non_stationary_source_is_refused_with_the_adf_numbers(
    client: TestClient, data_dir: Path
) -> None:
    """The stationarity guardrail, on the case it exists for.

    Not on prices — a price level is near-unit-root by construction, so refusing prices would
    refuse every dataset. The guard catches a column that is not a return series at all: here
    a cumulative total. The refusal carries the statistic and the p-value, because "not
    stationary" without them is a verdict the user cannot act on.
    """
    frame = price_frame()
    frame["cum_flow"] = frame["volume"].cumsum()
    dataset_id = register(data_dir, frame)

    finished = poll(client, forecast(client, dataset_id, price_source="cum_flow")["job_id"])

    assert finished["status"] == "failed", finished
    assert finished["error"]["code"] == "validation_failure"
    message = finished["error"]["message"]
    assert "Dickey" in message and "cum_flow" in message
    assert finished["error"]["details"]["hint"]


def test_a_dataset_with_no_close_column_is_refused_with_a_hint(
    client: TestClient, data_dir: Path
) -> None:
    """A price level is never a model input, so a frame without one cannot derive returns."""
    frame = price_frame().drop(columns=["close", "open", "high", "low"])
    dataset_id = register(data_dir, frame)

    finished = poll(client, forecast(client, dataset_id)["job_id"])

    assert finished["status"] == "failed"
    assert "close" in finished["error"]["message"]
    assert "price level" in finished["error"]["details"]["hint"]


def test_a_gap_inside_the_return_series_is_refused_rather_than_bridged(
    client: TestClient, data_dir: Path
) -> None:
    """A hole would silently change what "the last ``test_window`` bars" means.

    Interpolating across it would produce a walk-forward over positions that are not
    contiguous in time, with nothing in the output saying so — and the scored region is chosen
    by position, so the mapping from bar to forecast would be wrong from the gap onward.
    """
    frame = price_frame()
    frame.loc[100:105, "close"] = np.nan
    dataset_id = register(data_dir, frame)

    finished = poll(client, forecast(client, dataset_id)["job_id"])

    assert finished["status"] == "failed"
    assert finished["error"]["code"] == "validation_failure"
    assert "empty values inside its usable span" in finished["error"]["message"]


def test_an_unsorted_time_column_is_refused_because_positions_are_not_chronological(
    client: TestClient, data_dir: Path
) -> None:
    frame = price_frame()
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    dataset_id = register(data_dir, shuffled)

    finished = poll(client, forecast(client, dataset_id)["job_id"])

    assert finished["status"] == "failed"
    assert "not sorted" in finished["error"]["message"]


# ------------------------------------------------------------------ the artifact


def test_the_saved_frame_carries_the_specified_schema(
    client: TestClient, app_catalog: Catalog, data_dir: Path
) -> None:
    """Column names, order, dtypes, and the ordering the next consumer will rely on."""
    dataset_id = register(data_dir, price_frame())
    finished = run_to_completion(client, dataset_id, models=["garch", "egarch"])
    child = app_catalog.get_dataset(finished["result"]["dataset"]["dataset_id"])
    assert child is not None

    frame = parquet_store.read_dataset(Path(child.storage_path))
    assert tuple(frame.columns) == VOL_FORECAST_COLUMNS
    # No `close_time` column: carrying one "for readability" would reintroduce the name that
    # makes a mistaken join look correct. See the module docstring of `app.models.volatility`.
    assert "close_time" not in frame.columns
    assert "is_forward_path" not in frame.columns

    assert str(frame["open_time"].dtype) == "datetime64[ns, UTC]"
    assert len(frame) == 60, "two models over a 30-bar scored region"
    assert set(frame["model"]) == {"garch", "egarch"}

    # Chronological overall, so the frame is a series like every other one in the store; and
    # unique per model, so a consumer grouping by model gets 30 distinct bars, not 60.
    assert frame["open_time"].is_monotonic_increasing
    for model, group in frame.groupby("model", sort=False):
        assert group["open_time"].is_monotonic_increasing, model
        assert not group["open_time"].duplicated().any(), model
        assert len(group) == 30, model

    finite = frame[frame["finite"]]
    assert len(finite) == 60
    assert np.isfinite(finite["forecast_variance"]).all()
    assert (finite["forecast_variance"] > 0.0).all()
    # `forecast_volatility` is the square root of the variance column, not an independent
    # quantity — a second estimate of the same thing would be a second thing to disagree.
    np.testing.assert_allclose(
        finite["forecast_volatility"] ** 2, finite["forecast_variance"], rtol=1e-12
    )
    np.testing.assert_allclose(
        finite["actual_variance"] - finite["forecast_variance"],
        finite["error_variance"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        finite["ratio"], finite["actual_variance"] / finite["forecast_variance"], rtol=1e-12
    )


def test_a_diverged_fit_is_refused_end_to_end_and_never_reaches_the_frame(
    client: TestClient, app_catalog: Catalog, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard as the page sees it: nothing absurd stored, and a reason in words.

    The divergence is injected rather than waited for. An EGARCH fit that pins ``β`` at
    ≈ 0.999 lands there on some windows and not others, so a test that depends on which window
    the optimiser gives up on is a test that starts failing the day arch changes its starting
    values. The value injected is one this fixture really produced — ``8.3e213`` against a
    window whose largest squared return is ``4e-5`` — and what is under test is the *service's*
    response to such a value: that it is counted, reported, kept out of the dataset, left as a
    gap in the fan chart, and answered with the likeliest fix.
    """
    from app.services import volatility_models

    real_build = volatility_models.build_model

    def exploding(name: Any, **kwargs: Any) -> Any:
        model = real_build(name, **kwargs)
        model.forecast_variance = lambda horizon: np.full(horizon, 8.3e213)
        return model

    monkeypatch.setattr(volatility_models, "build_model", exploding)

    dataset_id = register(data_dir, price_frame())
    finished = run_to_completion(client, dataset_id, models=["garch"])

    report = finished["result"]["report"]
    score = report["scores"][0]
    assert score["n_nan"] == 30
    assert score["n_finite"] == 0
    assert score["mqlike"] is None and score["rmse"] is None
    assert "Only 0 of 30 steps" in score["unscored_reason"]
    assert "longer train_window" in score["unscored_reason"]
    # The run's own warning says how many steps were lost, not how many reasons were kept:
    # the failure list is capped at five, the count is not.
    assert any("30 walk-forward steps produced no usable forecast" in w for w in report["warnings"])

    # The fan chart keeps its shape and loses its `8e213`: a gap, which the report flags,
    # rather than an axis that flattens every other line on the page.
    path = report["forward_paths"][0]
    assert len(path["times"]) == report["forecast_horizon"]
    assert all(value is None or value != value for value in path["variance"])

    # Nothing absurd is stored: the frame keeps the *step* with a null forecast rather than a
    # number every consumer downstream would read as an observation.
    child = app_catalog.get_dataset(finished["result"]["dataset"]["dataset_id"])
    frame = parquet_store.read_dataset(Path(child.storage_path))
    assert len(frame) == 30
    assert frame["forecast_variance"].isna().all()
    assert not frame["finite"].any()


def test_the_saved_frame_holds_no_future_timestamps(
    client: TestClient, app_catalog: Catalog, data_dir: Path
) -> None:
    """The forward path stays in the report.

    A dataset row in this application is an *observation*, and the projection the fan chart
    draws is not one. Writing those rows would put timestamps into the store that no cleaning
    step, no split plan and no backtest is entitled to treat as data — and every one of them
    would be silently on the wrong side of the train/test boundary the user draws later.
    """
    frame = price_frame()
    dataset_id = register(data_dir, frame)
    finished = run_to_completion(client, dataset_id, forecast_horizon=24)

    child = app_catalog.get_dataset(finished["result"]["dataset"]["dataset_id"])
    saved = parquet_store.read_dataset(Path(child.storage_path))
    source_last = pd.to_datetime(frame["open_time"], utc=True).max()

    assert saved["open_time"].max() == source_last
    assert (saved["open_time"] <= source_last).all()
    # The projection exists — it is just not here.
    assert len(finished["result"]["report"]["forward_paths"][0]["times"]) == 24


def test_the_output_lands_in_outputs_under_a_new_dataset(
    client: TestClient, app_catalog: Catalog, data_dir: Path
) -> None:
    """`outputs`, with the source named as parent and the source itself untouched."""
    dataset_id = register(data_dir, price_frame())
    source_path = Path(
        app_catalog.conn.execute(
            "SELECT storage_path FROM datasets WHERE dataset_id = ?", (dataset_id,)
        ).fetchone()["storage_path"]
    )
    before = parquet_store.verify_checksum(source_path)

    finished = run_to_completion(client, dataset_id)
    child_id = finished["result"]["dataset"]["dataset_id"]

    assert child_id != dataset_id
    child = app_catalog.get_dataset(child_id)
    assert child is not None
    assert child.category is DatasetCategory.OUTPUTS
    assert child.parent_dataset_id == dataset_id
    assert child.row_count == 30
    assert child.timeframe is Timeframe.H1

    assert parquet_store.verify_checksum(source_path) == before


def test_lineage_records_the_whole_request_and_the_scored_region(
    client: TestClient, app_catalog: Catalog, data_dir: Path
) -> None:
    """A volatility forecast without its distribution and train window is unreproducible."""
    dataset_id = register(data_dir, price_frame())
    finished = run_to_completion(
        client,
        dataset_id,
        models=["garch", "egarch"],
        distribution="ged",
        mean_model="constant",
        p=2,
        o=1,
        q=2,
        forecast_horizon=12,
    )
    child_id = finished["result"]["dataset"]["dataset_id"]

    rows = app_catalog.conn.execute(
        "SELECT parent_id, operation, detail FROM lineage WHERE child_id = ?", (child_id,)
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["parent_id"] == dataset_id
    assert row["operation"] == "volatility"

    detail = json.loads(row["detail"])
    assert detail["module_source"] == "volatility"
    parameters = detail["parameters_used"]
    # The whole request, verbatim — replaying this dict reproduces the run.
    assert parameters["dataset_id"] == dataset_id
    assert parameters["distribution"] == "ged"
    assert parameters["p"] == 2
    assert parameters["o"] == 1
    assert parameters["q"] == 2
    assert parameters["forecast_horizon"] == 12
    assert parameters["walk_forward"]["train_window"] == FAST["train_window"]

    # The scored region, so a reader can tell what the metrics are over without re-deriving
    # the schedule.
    walk = detail["walk_forward"]
    assert walk["n_scored"] == 30
    assert walk["first_scored_time"] < walk["last_scored_time"]

    stationarity = detail["stationarity"]
    assert stationarity["stationary"] is True
    assert stationarity["alpha"] == 0.05
    assert np.isfinite(stationarity["adf_statistic"])

    scores = detail["model_scores"]
    assert set(scores) == {"garch", "egarch"}
    assert scores["garch"]["family"] == "simple"
    # The steps the protocol produced, and how many of them were finite. Two names, because
    # the walk-forward block above already uses `n_scored` for the first of them.
    assert scores["garch"]["n_steps"] == 30
    assert scores["garch"]["n_finite"] == 30
    assert scores["garch"]["n_nan"] == 0
    assert scores["garch"]["n_inf"] == 0

    assert len(detail["diebold_mariano"]) == 1
    comparison = detail["diebold_mariano"][0]
    assert {comparison["a"], comparison["b"]} == {"garch", "egarch"}
    assert comparison["n"] == 30


def test_provenance_carries_the_headline_parameters(
    client: TestClient, app_catalog: Catalog, data_dir: Path
) -> None:
    """The dataset row's own provenance, readable without joining to lineage."""
    dataset_id = register(data_dir, price_frame())
    finished = run_to_completion(client, dataset_id, models=["garch"], distribution="skew_student")
    child = app_catalog.get_dataset(finished["result"]["dataset"]["dataset_id"])
    assert child is not None

    provenance = child.provenance
    assert provenance["module_source"] == "volatility"
    assert provenance["distribution"] == "skew_student"
    assert provenance["mean_model"] == "constant"
    assert provenance["models"] == "garch"
    # Recorded as strings so a float column cannot lose precision in the JSON blob.
    assert provenance["train_window"] == str(FAST["train_window"])
    assert provenance["test_window"] == str(FAST["test_window"])
    assert provenance["step_size"] == "1"


def test_a_write_false_run_computes_but_saves_nothing(
    client: TestClient, app_catalog: Catalog, data_dir: Path
) -> None:
    """The preview path. Tuning a train window should not fill the catalog with runs."""
    dataset_id = register(data_dir, price_frame())
    before = len(app_catalog.list_datasets())

    finished = run_to_completion(client, dataset_id, write=False)

    assert finished["result"]["dataset"] is None
    assert finished["result"]["report"]["scores"][0]["mqlike"] is not None
    assert finished["output_dataset_id"] is None
    assert len(app_catalog.list_datasets()) == before


# --------------------------------------------------------------- conditioning


def test_conditioning_joins_the_regime_posteriors_and_records_a_second_parent(
    client: TestClient, app_catalog: Catalog, data_dir: Path
) -> None:
    """The hybrid reads a Module D output, and the artifact says which one.

    This is the join that the ``close_time`` mislabel in the regime engine would have
    corrupted: aligned on the bar-close instant, every posterior would inform the *following*
    bar's forecast, off by one and with no error anywhere. The assertion that the columns
    arrived is therefore an assertion about alignment as much as about presence.
    """
    dataset_id = register(data_dir, price_frame(), name="btc 1h")
    conditions = register(
        data_dir, regime_frame(), name="btc 1h regime", category=DatasetCategory.OUTPUTS
    )

    finished = run_to_completion(
        client,
        dataset_id,
        models=["garch", "arma_egarch_xgboost"],
        conditioning_dataset_id=conditions,
    )
    report = finished["result"]["report"]

    assert report["conditioning_dataset_id"] == conditions
    assert report["conditioning_dataset_name"] == "btc 1h regime"
    assert set(report["conditioning_columns"]) == {
        "prob_trending_up",
        "prob_trending_down",
        "prob_ranging",
        "prob_crisis",
        "confidence",
    }
    # The hybrid had enough usable history here (the regime frame covers every bar), so it did
    # not fall back — which is what makes this a test of the conditioning rather than of the
    # fallback path.
    hybrid = next(score for score in report["scores"] if score["model"] == "arma_egarch_xgboost")
    assert hybrid["family"] == "hybrid"
    assert hybrid["n_finite"] == 30

    child_id = finished["result"]["dataset"]["dataset_id"]
    parents = app_catalog.conn.execute(
        "SELECT parent_id, detail FROM lineage WHERE child_id = ?", (child_id,)
    ).fetchall()
    parent_ids = {row["parent_id"] for row in parents}
    assert parent_ids == {dataset_id, conditions}
    # The primary row, chosen by parent rather than by position — the two rows carry
    # different details and the order they were inserted in is not part of the contract.
    detail = json.loads(next(row["detail"] for row in parents if row["parent_id"] == dataset_id))
    assert detail["conditioning_dataset_id"] == conditions
    assert detail["conditioning_columns"]
    # The conditioning parent's own row says what it was for.
    role = json.loads(next(row["detail"] for row in parents if row["parent_id"] == conditions))
    assert role["role"] == "conditioning"


def test_a_plain_model_ignores_the_conditioning_dataset_but_still_records_it(
    client: TestClient, data_dir: Path
) -> None:
    """Family A has no place to put a posterior, and the report says so rather than implying
    the conditioning had an effect it could not have had."""
    dataset_id = register(data_dir, price_frame())
    conditions = register(
        data_dir, regime_frame(), name="regime", category=DatasetCategory.OUTPUTS
    )

    finished = run_to_completion(
        client, dataset_id, models=["garch"], conditioning_dataset_id=conditions
    )
    report = finished["result"]["report"]

    assert report["conditioning_dataset_id"] == conditions
    assert report["scores"][0]["mqlike"] is not None


def test_a_conditioning_dataset_without_an_observation_column_is_refused(
    client: TestClient, data_dir: Path
) -> None:
    """The refusal that names the mislabelled column without repeating the name.

    A regime output produced before the repair carries the bar-*close* instant under a column
    this module will not join on, so its rows cannot be aligned. Refusing is the only correct
    answer: joining on the wrong instant would date every posterior one bar late, and nothing
    downstream would report it.
    """
    dataset_id = register(data_dir, price_frame())
    malformed = regime_frame().rename(columns={"open_time": "observation_instant"})
    conditions = register(
        data_dir, malformed, name="regime", category=DatasetCategory.OUTPUTS
    )

    finished = poll(
        client,
        forecast(client, dataset_id, conditioning_dataset_id=conditions)["job_id"],
    )

    assert finished["status"] == "failed"
    assert finished["error"]["code"] == "validation_failure"
    assert "observation_instant" in finished["error"]["message"] or "open_time" in (
        finished["error"]["message"]
    )
    assert "one bar late" in finished["error"]["details"]["hint"]


def test_a_conditioning_dataset_that_is_not_a_regime_output_is_refused(
    client: TestClient, data_dir: Path
) -> None:
    """Conditioning expects Module D. A dataset with no posterior columns is a mistake."""
    dataset_id = register(data_dir, price_frame())
    wrong = register(data_dir, price_frame(rows=50), name="not a regime")

    finished = poll(
        client, forecast(client, dataset_id, conditioning_dataset_id=wrong)["job_id"]
    )

    assert finished["status"] == "failed"
    assert "regime" in finished["error"]["message"].lower()
    assert "prob_crisis" in finished["error"]["message"]


# -------------------------------------------------------------------- options


def test_options_report_what_the_dataset_can_support(
    client: TestClient, data_dir: Path
) -> None:
    """One call for the whole page, with availability decided server-side.

    The same rule the feature catalogue and Module D follow: one function answers "is this
    available, and if not why", so a disabled control and a refusal from the run endpoint
    cannot disagree about the reason.
    """
    dataset_id = register(data_dir, price_frame())

    response = client.get("/api/volatility/options", params={"dataset_id": dataset_id})
    assert response.status_code == 200
    body = response.json()

    assert body["dataset_id"] == dataset_id
    assert body["rows"] == 260
    # The count the schedule is actually sized against. One fewer than the rows, because the
    # first log return has no predecessor — and a form sized to the frame would open on a
    # refusal by exactly that bar.
    assert body["usable_returns"] == 259
    assert body["timeframe"] == "1h"
    assert "close" in body["columns"]

    sources = {source["id"]: source for source in body["sources"]}
    assert sources["close"]["available"] is True
    assert sources["log_return"]["available"] is False
    assert sources["log_return"]["reason"], "an unavailable source states why"

    models = {model["id"]: model for model in body["models"]}
    assert set(models) == {
        "garch", "egarch", "gjr_garch", "arma_egarch_xgboost", "tdnn_gjr_garch"
    }
    assert models["garch"]["family"] == "simple"
    assert models["garch"]["needs"] == ["p", "q"]
    assert models["egarch"]["needs"] == ["p", "o", "q"]
    assert models["arma_egarch_xgboost"]["family"] == "hybrid"
    assert "lags" in models["arma_egarch_xgboost"]["needs"]
    assert models["arma_egarch_xgboost"]["note"]

    # The direction that counts as better is sent from the server, because marking the
    # *highest* MQLIKE as best inverts the conclusion of the whole report.
    metrics = {metric["id"]: metric for metric in body["metrics"]}
    assert metrics["mqlike"]["lower_is_better"] is True
    assert metrics["rmse"]["lower_is_better"] is True
    assert metrics["mae"]["lower_is_better"] is True
    # A diagnostic, not a ranking: a high MZ R² is not "better" in the sense MQLIKE is.
    assert metrics["mz_r2"]["lower_is_better"] is False

    assert body["feasible"] is True
    assert body["defaults"]["price_source"] == "close"
    assert body["defaults"]["distribution"] == "students_t"
    assert body["defaults"]["train_window"] + body["defaults"]["test_window"] <= 260
    assert not [warning for warning in body["warnings"] if warning["level"] == "error"]


def test_options_open_on_a_schedule_the_dataset_can_actually_run(
    client: TestClient, data_dir: Path
) -> None:
    """A form that opens on its own refusal teaches the user the module is broken.

    The defaults are derived from the row count rather than fixed, so a short dataset opens on
    a configuration that runs. The assertion is the arithmetic the run endpoint would apply,
    not a hard-coded pair of numbers.
    """
    dataset_id = register(data_dir, price_frame(rows=140))

    body = client.get("/api/volatility/options", params={"dataset_id": dataset_id}).json()
    defaults = body["defaults"]

    assert body["rows"] == 140
    assert defaults["train_window"] >= 50
    assert defaults["test_window"] >= 30
    assert defaults["train_window"] + defaults["test_window"] <= 139


def test_options_flag_a_dataset_too_short_for_any_schedule(
    client: TestClient, data_dir: Path
) -> None:
    """Said before Run is pressed, and said as an error rather than a caution — there is no
    configuration that would work."""
    dataset_id = register(data_dir, price_frame(rows=60))

    body = client.get("/api/volatility/options", params={"dataset_id": dataset_id}).json()

    assert body["feasible"] is False
    errors = [warning for warning in body["warnings"] if warning["level"] == "error"]
    assert [warning["code"] for warning in errors] == ["too_short"]
    # Both counts, so the reader can reconcile the message with the row count on the card.
    assert "59" in errors[0]["message"] and "60" in errors[0]["message"]


def test_options_offer_a_regime_dataset_for_conditioning(
    client: TestClient, data_dir: Path
) -> None:
    """Filtered by the columns a regime output carries, not by the category — so a regime
    dataset re-saved elsewhere is still offered and a non-regime dataset never is."""
    dataset_id = register(data_dir, price_frame())
    conditions = register(
        data_dir, regime_frame(), name="regime", category=DatasetCategory.OUTPUTS
    )
    register(data_dir, price_frame(rows=40), name="unrelated", category=DatasetCategory.OUTPUTS)

    body = client.get("/api/volatility/options", params={"dataset_id": dataset_id}).json()

    offered = {entry["dataset_id"]: entry for entry in body["conditioning"]}
    assert set(offered) == {conditions}
    assert offered[conditions]["usable"] is True
    assert "prob_crisis" in offered[conditions]["columns"]


def test_options_refuse_an_unknown_dataset(client: TestClient) -> None:
    assert client.get("/api/volatility/options", params={"dataset_id": "nope"}).status_code == 404


def test_the_job_appears_in_the_recent_list(client: TestClient, data_dir: Path) -> None:
    """The job registry is generic, so Module E's run shows up alongside everything else."""
    dataset_id = register(data_dir, price_frame())
    accepted = forecast(client, dataset_id)
    poll(client, accepted["job_id"])

    listed = client.get("/api/jobs", params={"limit": 25}).json()
    record = next(row for row in listed if row["job_id"] == accepted["job_id"])
    assert record["kind"] == "volatility_forecast"
    assert record["dataset_id"] == dataset_id
    assert record["output_dataset_id"]
