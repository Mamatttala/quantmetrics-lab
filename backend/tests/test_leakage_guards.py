"""Look-ahead and data-leakage guards.

"Prevention of look-ahead / data leakage" sits third in this project's stated order of
priorities, above every architectural and UX concern. It earns that place because it is
the failure mode that survives review: a series keyed off the wrong timestamp, or a
correction applied retroactively, produces a plausible number with no error attached.
The result is not obviously broken; it is subtly wrong, and it is discovered — if at all
— by a comparison nobody thought to run.

Two rules are pinned here.

**Observation time.** `open_time` is the instant a bar opens; `close_time` is the instant
it closes. They differ by one bar length, which is exactly the size of a look-ahead error
that no output would reveal. Both columns are present in a Binance kline, so choosing the
wrong one is not merely possible, it is the *easier* mistake. Every place that needs "when
did this observation occur" must therefore select from a candidate list that omits
`close_time` entirely, and this file asserts that mechanically as well as behaviourally.

**Content determinism.** `IngestService.register` deduplicates on content hash, so asking
for the same data twice yields one dataset rather than two identical ones. That is what
makes a checksum usable as an identity — two rows sharing one checksum would make it
useless for the "did this dataset change?" question the whole storage design rests on.

**Causal preprocessing.** Modules C6 and C7 decide which rows a model may see and what the
numbers *are*, and both can leak without producing a frame that looks any different. The
two mechanical guards for them live here beside the observation-time one, because the
technique is the same: a source-level assertion catches the mistake a behavioural test on
one particular series would miss.

**Module E's feature builder.** A forecasting module can leak in a way none of the above
can: by handing a model a feature that is *contemporaneous with its target*. That is not a
misdated timestamp or a badly fitted scaler — it is the target itself, arriving through the
front door, and the resulting model reports a spectacular score for reasons that have
nothing to do with the market. The last two tests pin the delay line that prevents it.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.config import Settings
from app.core.errors import ValidationFailure
from app.models.enums import (
    AssetClass,
    DatasetCategory,
    Provider,
    ScalerType,
    SplitRole,
    SplitType,
    Timeframe,
)
from app.models.scaling import ScalingRequest
from app.models.splits import SplitCreateRequest, SplitParameters
from app.services import quality, volatility_models
from app.services import scaling as scaling_module
from app.services import splits as splits_module
from app.services.datasets import DatasetService
from app.services.ingest import IngestService
from app.services.scaling import ScalingService
from app.services.splits import SplitService
from app.storage import parquet_store
from app.storage.catalog import Catalog
from tests.conftest import hourly_frame, register

SERVICES = Path(__file__).resolve().parent.parent / "app" / "services"


# ------------------------------------------------------- observation-time rule


def test_close_time_is_not_a_candidate_observation_time() -> None:
    """The observation-time selectors must not mention `close_time` at all.

    Behavioural tests below cover the cases that matter, but they share a blind spot:
    with both columns present, a candidate list that has `open_time` *first* still picks
    the right one even if `close_time` was appended after it. The bug would only appear
    on a frame lacking `open_time` — which is precisely the frame where nobody is
    looking. Checking the source closes that gap directly.

    If this fails, it is not necessarily wrong to have added `close_time` somewhere in
    these modules; it means a human has to confirm it is not being used as the instant
    an observation is dated. That is a decision, not a refactor.
    """
    assert "close_time" not in inspect.getsource(quality.find_time_column)
    assert "close_time" not in (SERVICES / "datasets.py").read_text(encoding="utf-8")


def test_open_time_wins_over_close_time_when_both_are_present() -> None:
    frame = hourly_frame(20)
    assert "open_time" in frame.columns and "close_time" in frame.columns
    assert quality.find_time_column(frame) == "open_time"


def test_a_frame_carrying_only_close_time_is_reported_as_undated() -> None:
    """A row's *close* is not the instant the row is observed.

    Falling back to `close_time` here would date every marker one bar late and, worse,
    make the quality report for such a dataset look like a complete one. Reporting "no
    recognisable timestamp column" is the honest answer.
    """
    frame = hourly_frame(20).drop(columns=["open_time"])
    assert quality.find_time_column(frame) is None

    report = quality.analyse(frame, Timeframe.H1)
    assert report["time_column"] == ""
    assert report["markers"] == []
    assert report["counts"] == {}
    assert any("no recognisable timestamp column" in reason for reason in report["skipped"])


def test_markers_are_dated_at_the_bar_open_not_the_bar_close() -> None:
    """The one-bar difference made explicit, on a frame where both columns exist."""
    frame = hourly_frame(10)
    frame.loc[0, "close"] = -1.0
    report = quality.analyse(frame, Timeframe.H1)

    open_epoch = int(frame.loc[0, "open_time"].timestamp())
    close_epoch = int(frame.loc[0, "close_time"].timestamp())
    # Without this the assertion below could pass by the two columns being equal.
    assert open_epoch != close_epoch

    flagged = [m for m in report["markers"] if m["kind"] == "non_positive_price"]
    assert flagged[0]["time"] == open_epoch


async def test_ohlcv_refuses_a_frame_with_only_close_time(
    settings: Settings, catalog: Catalog
) -> None:
    """Raw Data View must not quietly chart a close-time axis.

    A candlestick whose x values are bar-close instants is offset from every other
    series in the application by one bar, and looks entirely normal while being so.
    """
    frame = hourly_frame(20).drop(columns=["open_time"])
    meta = register(settings, catalog, frame)

    service = DatasetService(settings=settings, catalog=catalog)
    with pytest.raises(ValidationFailure, match="no recognisable timestamp column"):
        await service.ohlcv(meta.dataset_id)


async def test_ohlcv_dates_bars_from_open_time(settings: Settings, catalog: Catalog) -> None:
    """The positive half of the same rule, on a frame that has both columns."""
    frame = hourly_frame(20)
    meta = register(settings, catalog, frame)

    service = DatasetService(settings=settings, catalog=catalog)
    payload = await service.ohlcv(meta.dataset_id)

    open_epoch = int(frame.loc[0, "open_time"].timestamp())
    close_epoch = int(frame.loc[0, "close_time"].timestamp())
    assert open_epoch != close_epoch
    assert payload["bars"][0]["time"] == open_epoch


# ------------------------------------------------------------ content identity


def _fetch_kwargs(name: str) -> dict:
    return {
        "name": name,
        "category": DatasetCategory.RAW,
        "source": Provider.BINANCE,
        "asset_class": AssetClass.CRYPTO_SPOT,
        "symbol": "BTCUSDT",
        "market_type": "spot",
        "timeframe": Timeframe.H1,
        "timezone": "UTC",
    }


async def test_identical_content_is_registered_once(settings: Settings, catalog: Catalog) -> None:
    """Same bytes in, same dataset out — the fetch is idempotent.

    Re-running a fetch is an ordinary thing to do, so the second one has to be a no-op
    rather than an accumulating duplicate. The stored frame is returned from the first
    registration and the second copy of the file is removed, because two catalog rows
    sharing one checksum would make the checksum useless as an identity.
    """
    service = IngestService(settings=settings, catalog=catalog)
    frame = hourly_frame(50)

    first, first_deduped = await service.register(frame, **_fetch_kwargs("first"))
    second, second_deduped = await service.register(frame, **_fetch_kwargs("second"))

    assert first_deduped is False
    assert second_deduped is True
    assert second.dataset_id == first.dataset_id
    assert second.checksum == first.checksum
    # The duplicate's name is discarded along with its file — the first registration
    # owns the content, and renaming it to match the second call would be a silent
    # edit to a dataset the user already has.
    assert second.name == "first"

    assert len(catalog.list_datasets()) == 1
    assert len(list(settings.data_dir.rglob("*.parquet"))) == 1


async def test_a_single_changed_value_is_not_deduplicated(
    settings: Settings, catalog: Catalog
) -> None:
    """The other half of the guarantee: deduplication must not be over-eager.

    A hash collision is not the risk here — matching on content means any real change
    produces a new dataset, and a deduplicator that returned the old one for changed
    data would silently discard the fetch the user just asked for.
    """
    service = IngestService(settings=settings, catalog=catalog)

    original = hourly_frame(50)
    changed = original.copy()
    changed.loc[17, "close"] = float(changed.loc[17, "close"]) + 0.01

    first, _ = await service.register(original, **_fetch_kwargs("original"))
    second, second_deduped = await service.register(changed, **_fetch_kwargs("changed"))

    assert second_deduped is False
    assert second.dataset_id != first.dataset_id
    assert second.checksum != first.checksum
    assert len(catalog.list_datasets()) == 2


async def test_deduplication_does_not_cross_categories(
    settings: Settings, catalog: Catalog
) -> None:
    """An uploaded copy of a raw dataset is a distinct dataset.

    Category is part of what a dataset *is* — it is the explorer's grouping and the
    statement about provenance — so collapsing an upload into a fetched dataset because
    the numbers match would move the user's file into a bucket they did not choose.
    """
    service = IngestService(settings=settings, catalog=catalog)
    frame = hourly_frame(50)

    raw, _ = await service.register(frame, **_fetch_kwargs("fetched"))
    uploaded, uploaded_deduped = await service.register(
        frame,
        **{**_fetch_kwargs("uploaded"), "category": DatasetCategory.UPLOADED},
    )

    assert uploaded_deduped is False
    assert uploaded.dataset_id != raw.dataset_id
    assert uploaded.checksum == raw.checksum, "same content, deliberately different identity"
    assert len(catalog.list_datasets()) == 2


async def test_the_checksum_is_reproducible_from_the_stored_bytes(
    settings: Settings, catalog: Catalog
) -> None:
    """Determinism is only meaningful if the recorded hash describes the file.

    Serialising the same frame twice must produce the same digest, or "did this dataset
    change?" cannot be answered from the catalog alone — which is the entire reason the
    checksum is stored.
    """
    frame = hourly_frame(50)
    first_id = parquet_store.new_dataset_id()
    second_id = parquet_store.new_dataset_id()

    first = parquet_store.write_dataset(frame, settings.data_dir, DatasetCategory.RAW, first_id)
    second = parquet_store.write_dataset(frame, settings.data_dir, DatasetCategory.RAW, second_id)

    assert first.checksum == second.checksum
    assert parquet_store.verify_checksum(first.storage_path) == first.checksum
    assert parquet_store.verify_checksum(second.storage_path) == second.checksum

    # And the stored bytes survive a read/write round trip unchanged, so the digest
    # keeps describing the data rather than only the first serialisation of it.
    reloaded = parquet_store.read_dataset(first.storage_path)
    assert reloaded["open_time"].dtype == pd.DatetimeTZDtype(tz="UTC")
    assert reloaded["close"].tolist() == frame["close"].tolist()


# ------------------------------------------------------ causal preprocessing
#
# C6 decides which rows a model may see; C7 decides what the numbers *are*. Both can leak
# without producing a frame that looks any different, so both get a mechanical guard here
# and a behavioural one beside it.


def test_the_split_module_never_names_the_bar_close() -> None:
    """C6 dates every span by the *observation* instant, and the source says so.

    Splitting on bar-close instants moves the last bar of training into the test span — a
    leak of exactly one bar with no visible symptom: the spans stay contiguous, ordered and
    non-overlapping, the row counts still add up, and the plan table reads correctly.

    The behavioural half is in `test_splits.py`. This half closes the gap that one leaves:
    with `open_time` present, a candidate list that had `close_time` appended after it would
    still pick the right column, and the bug would only appear on a frame *lacking*
    `open_time` — which is exactly where nobody looks.
    """
    assert "close_time" not in (SERVICES / "splits.py").read_text(encoding="utf-8")


def test_a_module_never_names_its_time_column_for_the_wrong_instant() -> None:
    """A column called `close_time` holding `open_time` values is a mislabelled join key.

    `regime_engine` built its output frame with a column named `close_time` whose values were
    `frame[time_column]` — the *observation* instant. The frame was correct and the name was
    a lie, which is the more dangerous of the two: any consumer that joined on the name
    rather than reading it (Module E's conditioning join is exactly such a consumer) would
    have aligned every regime with the following bar, off by one, with no error and a chart
    that looks entirely plausible.

    The behavioural half lives in `test_regime_engine.py` and `test_regime_api.py`, which
    assert the output schema. This half catches the mistake at the source, where it is
    unambiguous: the engine must not mention the bar-close instant at all.
    """
    assert "close_time" not in (SERVICES / "regime_engine.py").read_text(encoding="utf-8")


def test_the_split_module_reads_the_shared_timestamp_candidates() -> None:
    """C6 must not roll its own list of time columns.

    One candidate list, in `quality`, is what makes the observation-time rule reviewable: a
    second list inside C6 could add a column — `close_time`, say — and every test of the
    shared list would keep passing while the splitter used something else.
    """
    assert "quality_checks.TIME_COLUMN_CANDIDATES" in inspect.getsource(splits_module)


def test_the_scaling_module_never_centres_a_rolling_window() -> None:
    """`rolling(center=True)` reads half a window into the future.

    It is one keyword, it is a plausible thing to reach for, and the column it produces is
    indistinguishable from a causal one — so no assertion about a particular series would
    necessarily catch it. `center=False` is stated explicitly in `rolling_zscore`, and the
    affine field is called `center_`; neither is the forbidden form.
    """
    assert "center=True" not in (SERVICES / "scaling.py").read_text(encoding="utf-8")


def test_a_fitted_scaler_cannot_be_requested_without_a_plan() -> None:
    """The golden rule, held by the schema rather than by the implementation.

    `ScalingRequest` makes `split_plan_id` mandatory for the three fitted scalers, so there
    is no request shape — and therefore no code path — that fits on everything. Asserting
    the rule where it is *enforced* is the difference between "the service remembers to
    check" and "the request cannot say it".
    """
    assert "needs_split_plan" in inspect.getsource(scaling_module)
    for scaler in (ScalerType.MINMAX, ScalerType.STANDARD, ScalerType.ROBUST):
        assert scaler.needs_split_plan is True
        with pytest.raises(ValueError, match="needs split_plan_id"):
            ScalingRequest(dataset_id="x", scaler_type=scaler)
    assert ScalerType.ROLLING_ZSCORE.needs_split_plan is False


async def test_a_standard_scaler_does_not_carry_the_test_spans_mean(
    settings: Settings, catalog: Catalog
) -> None:
    """The leak stated as a number, end to end through the service.

    A frame whose first 140 rows average 100 and whose last 60 average 200, standard-scaled
    against a 70/15/15 plan. A scaler that had seen the whole series would have subtracted
    ~130 and left the test rows centred at zero; the fitted centre must instead be the
    *training* mean, and the test rows must sit a hundred units above the origin.
    """
    frame = hourly_frame(200)
    frame["close"] = np.concatenate([np.full(140, 100.0), np.full(60, 200.0)])
    parent = register(settings, catalog, frame)

    plan = await SplitService(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=parent.dataset_id,
            split_type=SplitType.PERCENTAGE_70_15_15,
        )
    )
    result = await ScalingService(settings, catalog).run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close"],
        )
    )

    assert result.dataset is not None
    # `storage_path` is recorded on the row as a string; `parquet_store` deals in `Path`
    # (see `datasets._read`), so the conversion belongs here rather than in the store.
    scaled = parquet_store.read_dataset(Path(result.dataset.storage_path))["close"]
    assert result.parameters[0].center_ == pytest.approx(100.0)
    assert abs(scaled.iloc[:140].mean()) < 1e-9
    assert abs(scaled.iloc[170:].mean()) > 1.0


async def test_a_perturbed_test_span_cannot_move_a_fitted_parameter(
    settings: Settings, catalog: Catalog
) -> None:
    """The complement of the test above: the fit is a function of the training rows alone.

    A mean-based assertion can pass while a parameter is slightly contaminated, so this
    states the stronger property directly — change the last quarter of the series by 5,000
    units and every fitted number, and every transformed *training* row, must be identical.
    """
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame, name="baseline")
    plan = await SplitService(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15
        )
    )
    service = ScalingService(settings, catalog)
    first = await service.run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.ROBUST,
            features=["close"],
        )
    )

    changed = frame.copy()
    changed.loc[150:, "close"] = changed.loc[150:, "close"] + 5_000.0
    other = register(settings, catalog, changed, name="perturbed")
    other_plan = await SplitService(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=other.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15
        )
    )
    second = await service.run(
        ScalingRequest(
            dataset_id=other.dataset_id,
            split_plan_id=other_plan.dataset.dataset_id,
            scaler_type=ScalerType.ROBUST,
            features=["close"],
        )
    )

    assert first.parameters[0].center_ == second.parameters[0].center_
    assert first.parameters[0].scale_ == second.parameters[0].scale_
    assert first.dataset is not None and second.dataset is not None
    pd.testing.assert_frame_equal(
        parquet_store.read_dataset(Path(first.dataset.storage_path)).iloc[:150],
        parquet_store.read_dataset(Path(second.dataset.storage_path)).iloc[:150],
    )


def test_the_rolling_z_score_is_causal_at_every_row() -> None:
    """Truncating the series must not change the rows that remain.

    Computed over the whole frame and over the first 80% of it, the shared prefix has to be
    identical to the last bit. Any forward-looking statistic — a centred window, a
    whole-series mean — makes row *t* depend on rows the truncated run does not have, and
    the two disagree inside the prefix. This is the property itself, not a proxy for it.
    """
    frame = hourly_frame(200)
    whole, _, _ = scaling_module.rolling_zscore(frame, ["close", "volume"], 60)
    truncated, _, _ = scaling_module.rolling_zscore(frame.iloc[:160], ["close", "volume"], 60)

    for column in ("close", "volume"):
        assert np.allclose(
            whole[column].iloc[:160].to_numpy(),
            truncated[column].to_numpy(),
            equal_nan=True,
        ), f"{column} depends on rows after it"


async def test_the_tscv_purge_removes_the_last_label_horizon_rows(
    settings: Settings, catalog: Catalog
) -> None:
    """`purge_flag=True` with a 5-bar horizon must end training exactly 5 rows early.

    A label with a forward horizon reads into the next span, so the last rows of training
    are not really training data. The check is on the plan's own numbers — the training span
    ends at the cut minus the horizon, and the rows removed are reported — because a purge
    that silently removed nothing would leave a plan that *looks* purged and is not.
    """
    parent = register(settings, catalog, hourly_frame(100))
    created = await SplitService(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=parent.dataset_id,
            split_type=SplitType.TSCV_EXPANDING,
            parameters=SplitParameters(
                n_folds=4, gap_size=0, purge_flag=True, label_horizon_bars=5
            ),
        )
    )

    first = created.report.boundaries[0]
    assert first["cut_index"] == 20
    assert first["purge_rows_removed"] == 5
    assert first["effective_end_index"] == 15

    train = next(r for r in created.plan if r.fold == 1 and r.role is SplitRole.TRAIN)
    test = next(r for r in created.plan if r.fold == 1 and r.role is SplitRole.TEST)
    assert train.end_index == test.start_index - 5


async def test_the_tscv_purge_removes_nothing_when_the_gap_already_covers_it(
    settings: Settings, catalog: Catalog
) -> None:
    """Gap and purge overlap rather than add up, and the report has to say so.

    The embargo drops `[c - gap, c)` and the purge drops `[c - h, c)`. With `gap >= h` the
    second set is a subset of the first, so the training span ends at `c - gap` and the
    purge removed zero rows. A plan that reported the two as independent counts would read
    as "twice as safe" while being exactly as safe as the gap alone.
    """
    parent = register(settings, catalog, hourly_frame(100))
    created = await SplitService(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=parent.dataset_id,
            split_type=SplitType.TSCV_EXPANDING,
            parameters=SplitParameters(
                n_folds=4, gap_size=5, purge_flag=True, label_horizon_bars=5
            ),
        )
    )

    first = created.report.boundaries[0]
    assert first["gap_rows_removed"] == 5
    assert first["purge_rows_removed"] == 0
    assert first["effective_end_index"] == 15
    assert any("purge_flag removed no rows" in note for note in created.report.warnings)


async def test_a_plan_refuses_a_frame_it_was_not_computed_on(
    settings: Settings, catalog: Catalog
) -> None:
    """A plan is row indices, so it is only meaningful against one exact frame.

    The checksum is what makes that checkable, and it catches the case the lineage cannot:
    the same dataset id, rewritten in place. Exercised through the service's own guard
    rather than over HTTP, because for a genuinely *different* dataset the lineage check
    fires first and the checksum would never be reached.
    """
    parent = register(settings, catalog, hourly_frame(100))
    created = await SplitService(settings, catalog).create(
        SplitCreateRequest(dataset_id=parent.dataset_id, split_type=SplitType.TSCV_EXPANDING)
    )

    service = ScalingService(settings, catalog)
    stale = parent.model_copy(update={"checksum": "f" * 64})
    with pytest.raises(ValidationFailure, match="different version"):
        await service._load_plan(created.dataset.dataset_id, stale, None)


# ------------------------------------------------- Module E: the feature delay line
#
# A forecasting module has a leak available to it that none of the module above can commit:
# a feature that is contemporaneous with the target. Every model here predicts the variance
# of bar `t` from the history of bars before `t`, and the one place that can be got wrong is
# the delay line — one off-by-one in the shift and `r_t` sits in the feature row that exists
# to predict `r_t²`, which the learner will find immediately and report as skill.


def test_the_forecasting_modules_never_name_the_bar_close() -> None:
    """The observation-time rule, applied to the module that has the most to lose by it.

    A forecast is the one artifact in this application whose value depends on *when it could
    have been made*. Stamping it with `close_time` would date it one bar after the
    information it used, which is precisely the property that makes a backtest of it
    meaningless — and nothing in the output frame would look wrong.

    The behavioural half is in `test_volatility_api.py`, which asserts the output schema has
    no such column. This half catches a *reader* of the name, which is the harder failure:
    a join keyed on `close_time` between a forecast frame and a price frame aligns each
    forecast with the following bar, silently, with both frames individually correct.
    """
    assert "close_time" not in (SERVICES / "volatility.py").read_text(encoding="utf-8")
    assert "close_time" not in (SERVICES / "volatility_models.py").read_text(encoding="utf-8")


def test_the_delay_line_starts_at_lag_one() -> None:
    """The feature builder must shift, and its first shift must be one bar.

    A loop written as `range(0, lags)` would put the current return into the feature row
    beside its own target. It is a single character, it produces an array of exactly the
    right shape, and the only symptom is a model that scores far too well — which reads as
    success. Asserting the enumeration on the source is unambiguous in a way that a test on
    one series is not: the assertion is about the loop, not about a number it produced.

    `.shift(-` is the pandas spelling of the same mistake and is banned outright, since the
    delay line is built from arrays and has no reason to call it.
    """
    source = (SERVICES / "volatility_models.py").read_text(encoding="utf-8")
    assert "shift(-" not in source
    assert "center=True" not in source

    design = inspect.getsource(volatility_models.delayed_design)
    assert "range(1, lags + 1)" in design, "the lag enumeration must begin at 1"
    assert "range(0" not in design


def test_a_feature_row_cannot_see_its_own_bar() -> None:
    """The property itself, stated as the one experiment that would reveal a violation.

    Build the delay line over a series, change a single bar, and rebuild. Every row at or
    before that bar's index must come back bit-identical — those rows describe bars that
    happened before the edit — and the rows after it must move. A row *at* the changed index
    that moved would mean the row reads its own bar, and that row is the training pair whose
    label is that bar's squared return.

    Row `j` reads indices `j-lags .. j-1`, so a change at index 10 first appears in row 11
    when `lags >= 1`. Both halves are asserted, because an assertion that the prefix is
    unchanged would also pass for a builder that ignored its input entirely.
    """
    series = np.arange(1.0, 21.0)
    changed = series.copy()
    changed[10] = -999.0

    before = volatility_models.delayed_design(series, None, 3)
    after = volatility_models.delayed_design(changed, None, 3)

    np.testing.assert_array_equal(before[:11], after[:11], "a row reads a bar at or after itself")
    # ``np.allclose``'s third positional parameter is ``rtol``, not a message: passing the
    # sentence there raises a TypeError from inside numpy's ``isfinite`` rather than
    # reporting a failed assertion, which is how this test spent its first run failing for
    # a reason that had nothing to do with leakage.
    assert not np.allclose(before[11:], after[11:]), "the change never propagated forwards"


def test_a_lagged_exogenous_column_is_shifted_like_the_returns_are() -> None:
    """The conditioning columns are the easiest ones to forget.

    Returns go through the shift loop on their own; a regime posterior column arrives as a
    separate matrix and needs the *same* treatment. If it were concatenated unshifted, the
    hybrid would be reading today's regime to predict today's volatility — which is not
    available at the forecast origin, and which is exactly what the conditioning join exists
    to make impossible. The check is the same experiment as above, run on the exog block.
    """
    series = np.arange(1.0, 21.0)
    exog = np.column_stack([np.linspace(0.0, 1.0, 20), np.linspace(1.0, 0.0, 20)])
    changed = exog.copy()
    changed[10, 0] = 99.0

    before = volatility_models.delayed_design(series, exog, 2)
    after = volatility_models.delayed_design(series, changed, 2)

    np.testing.assert_array_equal(before[:11], after[:11])
    assert not np.allclose(before[11:], after[11:])
