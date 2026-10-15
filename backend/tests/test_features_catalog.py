"""The catalogue's own invariants.

The catalogue is the single place a feature's formula, parameters, source requirements and
availability rule are written down. Three consumers read it — the engine that computes, the
API that serves it to the info icon, and the tests below — so a defect in it is not a bug in
one place: it is a disagreement between the code and its own description.

These tests need no data and no server. They are about whether the catalogue is internally
consistent, and about whether availability says the right thing for datasets with
particular shapes.
"""

from __future__ import annotations

import pytest

from app.models.enums import AssetClass, FeatureGroup, FeatureKind, Timeframe, WindowKind
from app.services import feature_catalog as cat
from app.services import features

ALL = cat.catalogue()
COMPUTED = [spec for spec in ALL if spec.kind is FeatureKind.COMPUTED]
CARRIED = [spec for spec in ALL if spec.kind is FeatureKind.CARRIED]

#: The columns a Binance kline dataset has, which most availability tests are written on.
KLINE_COLUMNS = frozenset(
    {
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "num_trades",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    }
)


def context(
    *,
    asset_class: AssetClass = AssetClass.CRYPTO_SPOT,
    timeframe: Timeframe | None = Timeframe.H1,
    columns: frozenset[str] = KLINE_COLUMNS,
    unpopulated: frozenset[str] = frozenset(),
    has_time: bool = True,
    category: str = "raw_data",
    intraday: bool | None = None,
) -> cat.DatasetContext:
    return cat.DatasetContext(
        asset_class=asset_class,
        timeframe=timeframe,
        columns=columns,
        unpopulated=unpopulated,
        has_time=has_time,
        category=category,
        intraday=intraday,
    )


def available(spec: cat.FeatureSpec, ctx: cat.DatasetContext) -> bool:
    return cat.availability(spec, ctx).available


def tokens(spec: cat.FeatureSpec, ctx: cat.DatasetContext) -> list[str]:
    return [window.token for window in cat.availability(spec, ctx).windows if window.available]


# --------------------------------------------------------------- structural integrity


def test_the_catalogue_is_not_empty() -> None:
    assert len(ALL) >= 50, "the specification enumerates far more features than this"


@pytest.mark.parametrize("spec", ALL, ids=lambda spec: spec.id)
def test_every_entry_states_a_formula_and_a_reason_for_existing(spec: cat.FeatureSpec) -> None:
    assert spec.formula.strip(), "a feature with no formula is a feature nobody can check"
    assert spec.detail.strip(), "the info icon would open onto nothing"


@pytest.mark.parametrize("spec", ALL, ids=lambda spec: spec.id)
def test_the_column_name_carries_its_constants(spec: cat.FeatureSpec) -> None:
    """A baked-in period must be visible in the name, not hidden in the implementation.

    Bollinger's 20/2 and MACD's 12/26/9 are conventions the specification does not state.
    Choosing them is unavoidable; leaving them invisible is not.
    """
    if spec.param is not None:
        assert any("{arg}" in template for template in spec.columns), (
            f"{spec.id} takes a window but none of its columns name it, so two windows "
            "would produce the same column"
        )
    for template in spec.columns:
        assert template == template.strip()
        assert "{" not in template.replace("{arg}", ""), f"{spec.id}: unknown placeholder"


def test_ids_are_unique() -> None:
    ids = [spec.id for spec in ALL]
    assert len(ids) == len(set(ids))


def test_no_two_features_claim_the_same_column() -> None:
    """Templates are compared with the placeholder normalised away.

    Two entries producing `sma_20` would put one column in the frame twice, and the second
    would silently win.
    """
    seen: dict[str, str] = {}
    for spec in ALL:
        for template in spec.columns:
            key = template.replace("{arg}", "#")
            assert key not in seen, f"{spec.id} and {seen[key]} both produce {template}"
            seen[key] = spec.id


@pytest.mark.parametrize("spec", ALL, ids=lambda spec: spec.id)
def test_every_entry_has_a_computer(spec: cat.FeatureSpec) -> None:
    assert spec.id in features.COMPUTERS, f"{spec.id} is offered but cannot be computed"


def test_the_computer_table_has_no_orphans() -> None:
    known = {spec.id for spec in ALL}
    assert set(features.COMPUTERS) == known


@pytest.mark.parametrize("spec", ALL, ids=lambda spec: spec.id)
def test_needs_names_real_columns(spec: cat.FeatureSpec) -> None:
    for name in spec.needs:
        assert name in cat.SOURCE_COLUMNS, f"{spec.id} requires an unknown column {name!r}"


@pytest.mark.parametrize("spec", ALL, ids=lambda spec: spec.id)
def test_no_feature_is_built_on_a_publication_timestamp(spec: cat.FeatureSpec) -> None:
    """`close_time` is when a bar finished, so a value stamped with it already sees the bar.

    No entry may require one. The check reads the catalogue's own set rather than restating
    the list, so adding a timestamp column to the vocabulary cannot silently outrun it.
    """
    overlap = set(spec.needs) & cat.LEAKY_COLUMNS
    assert not overlap, f"{spec.id} is built on {sorted(overlap)}"


@pytest.mark.parametrize("spec", CARRIED, ids=lambda spec: spec.id)
def test_a_carried_entry_carries_exactly_its_own_column(spec: cat.FeatureSpec) -> None:
    """Nothing is computed for a carried feature, so nothing can be invented for it."""
    assert spec.columns == (spec.id,)
    assert spec.needs == (spec.id,)
    assert spec.param is None
    assert not spec.needs_price
    assert spec.warmup(0) == 0


@pytest.mark.parametrize("spec", [s for s in ALL if s.param], ids=lambda spec: spec.id)
def test_the_default_window_is_one_of_the_ticks(spec: cat.FeatureSpec) -> None:
    assert spec.param is not None
    assert spec.param.default in spec.param.ticks
    assert spec.param.ticks, f"{spec.id} offers no window at all"
    assert len(set(spec.param.ticks)) == len(spec.param.ticks)


def test_every_group_is_represented_and_ordered() -> None:
    assert set(cat.GROUP_ORDER) == set(FeatureGroup)
    present = {spec.group for spec in ALL}
    assert present == set(FeatureGroup)


def test_the_guard_sets_name_real_features() -> None:
    known = {spec.id for spec in ALL}
    assert known >= cat.NEEDS_TIME
    assert known >= cat.NEEDS_TIMEFRAME


def test_duration_ticks_are_parseable_and_in_the_vocabulary() -> None:
    for spec in ALL:
        if spec.param is None or spec.param.kind is not WindowKind.DURATION:
            continue
        for token in spec.param.ticks:
            assert token in cat.WINDOW_SECONDS, f"{spec.id}: {token} is not a known window"


# --------------------------------------------------------------------- duration maths


@pytest.mark.parametrize(
    ("token", "expected"),
    [("15m", 900), ("1h", 3600), ("4h", 14400), ("1d", 86400), ("30d", 2592000)],
)
def test_parse_duration(token: str, expected: int) -> None:
    assert cat.parse_duration(token) == expected


@pytest.mark.parametrize("token", ["", "h", "4", "4x", "4 h", "-1h", "4hh"])
def test_parse_duration_refuses_what_it_cannot_read(token: str) -> None:
    assert cat.parse_duration(token) is None


def test_a_window_is_a_whole_number_of_bars_or_nothing() -> None:
    assert cat.duration_bars(14400, Timeframe.H1) == 4
    assert cat.duration_bars(604800, Timeframe.H1) == 168
    assert cat.duration_bars(86400, Timeframe.D1) == 1
    # 15 minutes does not divide into an hour, and rounding it to one bar would turn a
    # quarter-hour return into an hourly one.
    assert cat.duration_bars(900, Timeframe.H1) is None
    assert cat.duration_bars(86400, Timeframe.W1) is None
    assert cat.duration_bars(3600, None) is None


# ----------------------------------------------------------------------- availability


def test_a_daily_dataset_cannot_support_an_intraday_window() -> None:
    ctx = context(timeframe=Timeframe.D1)
    spec = cat.CATALOGUE["realized_vol"]
    offered = tokens(spec, ctx)
    assert "7d" in offered and "21d" in offered
    assert "15m" not in offered and "1h" not in offered and "4h" not in offered

    by_token = {window.token: window for window in cat.availability(spec, ctx).windows}
    assert "not a whole number" in (by_token["4h"].reason or "")
    # "1d" is a whole number of daily bars — one of them — and one bar is not enough for a
    # standard deviation. This is the same rule catching a second case.
    assert by_token["1d"].bars == 1
    assert not by_token["1d"].available


def test_realized_volatility_needs_two_bars() -> None:
    """On an hourly dataset a one-hour realized volatility is a single absolute return."""
    ctx = context(timeframe=Timeframe.H1)
    spec = cat.CATALOGUE["realized_vol"]
    by_token = {window.token: window for window in cat.availability(spec, ctx).windows}
    assert by_token["1h"].bars == 1
    assert not by_token["1h"].available
    assert "at least 2" in (by_token["1h"].reason or "")
    assert by_token["4h"].bars == 4
    assert by_token["4h"].available


def test_an_interval_window_is_refused_without_a_declared_timeframe() -> None:
    ctx = context(timeframe=None)
    spec = cat.CATALOGUE["roc"]
    assert not cat.availability(spec, ctx).available
    assert "no timeframe" in (cat.availability(spec, ctx).reason or "")


def test_bar_count_windows_survive_a_missing_timeframe() -> None:
    """The specification says candle counts work for every dataset, and they do."""
    ctx = context(timeframe=None)
    assert cat.availability(cat.CATALOGUE["sma"], ctx).available


def test_a_crypto_only_field_is_disabled_on_an_equity_dataset() -> None:
    ctx = context(asset_class=AssetClass.EQUITY_SPOT, columns=KLINE_COLUMNS | {"last_funding_rate"})
    verdict = cat.availability(cat.CATALOGUE["last_funding_rate"], ctx)
    assert not verdict.available
    assert "crypto_futures" in (verdict.reason or "")
    assert "equity_spot" in (verdict.reason or "")


def test_an_equity_dataset_still_gets_the_ordinary_price_features() -> None:
    ctx = context(asset_class=AssetClass.EQUITY_SPOT)
    assert cat.availability(cat.CATALOGUE["sma"], ctx).available
    assert cat.availability(cat.CATALOGUE["rsi"], ctx).available
    assert cat.availability(cat.CATALOGUE["atr"], ctx).available


def test_a_missing_column_disables_the_feature_that_needs_it() -> None:
    ctx = context(columns=KLINE_COLUMNS - {"volume"})
    verdict = cat.availability(cat.CATALOGUE["obv"], ctx)
    assert not verdict.available
    assert "volume" in (verdict.reason or "")


def test_a_column_that_is_empty_in_every_row_does_not_count() -> None:
    """Yahoo returns a `settlement` column that never carries a value."""
    ctx = context(
        asset_class=AssetClass.EQUITY_FUTURES,
        columns=KLINE_COLUMNS | {"settlement"},
        unpopulated=frozenset({"settlement"}),
    )
    verdict = cat.availability(cat.CATALOGUE["settlement"], ctx)
    assert not verdict.available
    assert "empty in every row" in (verdict.reason or "")


def test_time_of_day_is_disabled_on_a_daily_dataset() -> None:
    ctx = context(timeframe=Timeframe.D1)
    verdict = cat.availability(cat.CATALOGUE["hour_of_day"], ctx)
    assert not verdict.available
    assert "same time of day" in (verdict.reason or "")


def test_time_of_day_is_offered_intraday() -> None:
    assert available(cat.CATALOGUE["hour_of_day"], context(timeframe=Timeframe.M15))


def test_time_of_day_is_refused_when_the_bars_never_move_off_one_clock_time() -> None:
    """A declared timeframe is not needed: the timestamps answer the question directly.

    An uploaded daily CSV declares no timeframe at all. Reading the bars instead of the
    metadata is what lets the refusal be about the data rather than about a missing label.
    """
    ctx = context(timeframe=None, intraday=False)
    verdict = cat.availability(cat.CATALOGUE["hour_of_day"], ctx)
    assert not verdict.available
    assert "same time of day" in (verdict.reason or "")


def test_time_of_day_is_offered_on_an_intraday_dataset_that_declares_no_timeframe() -> None:
    """The uploaded-hourly-CSV case: no declared bar length, but an honest clock time."""
    assert available(cat.CATALOGUE["hour_of_day"], context(timeframe=None, intraday=True))


def test_the_timestamps_override_a_declaration_that_disagrees_with_them() -> None:
    """If the bars sit at one time of day, a declared 1h timeframe does not make them
    intraday — the column would still be constant."""
    assert not available(
        cat.CATALOGUE["hour_of_day"], context(timeframe=Timeframe.H1, intraday=False)
    )


def test_time_of_day_is_refused_when_nothing_can_answer_the_question() -> None:
    ctx = context(timeframe=None, intraday=None)
    verdict = cat.availability(cat.CATALOGUE["hour_of_day"], ctx)
    assert not verdict.available
    assert "declares no timeframe" in (verdict.reason or "")


def test_calendar_features_need_only_a_date_not_a_declared_timeframe() -> None:
    """Which day is the last of its month follows from the calendar, not from bar length.

    Refusing these on a timeframe-less dataset would grey out a column that is perfectly
    well defined in it.
    """
    ctx = context(timeframe=None)
    for feature_id in ("month_end", "days_to_month_end", "days_to_quarter_end", "day_of_week"):
        assert available(cat.CATALOGUE[feature_id], ctx), f"{feature_id} should be available"


def test_calendar_features_need_a_time_column() -> None:
    ctx = context(has_time=False)
    assert not available(cat.CATALOGUE["day_of_week"], ctx)
    assert not available(cat.CATALOGUE["vwap_daily"], ctx)
    # A feature that never looks at the clock is unaffected.
    assert available(cat.CATALOGUE["sma"], ctx)


def test_annualized_volatility_is_refused_without_a_timeframe() -> None:
    """The annualisation factor is unknowable, and a wrong one is a silent scaling error."""
    ctx = context(timeframe=None)
    verdict = cat.availability(cat.CATALOGUE["annualized_vol"], ctx)
    assert not verdict.available
    assert "bar length" in (verdict.reason or "")
    assert available(cat.CATALOGUE["annualized_vol"], context(timeframe=Timeframe.D1))


def test_a_sub_day_interval_is_refused_on_a_daily_dataset() -> None:
    """The window is a duration, so 15m on a daily bar is not a whole number of bars.

    It is refused with a reason rather than rounded to one bar: rounding a quarter-hour
    window to a single daily bar would silently turn a 15-minute volatility into a daily
    one. The feature stays available on the daily dataset through its 7d/21d/30d windows.
    """
    verdict = cat.availability(cat.CATALOGUE["annualized_vol"], context(timeframe=Timeframe.D1))
    assert verdict.available
    windows = {window.token: window for window in verdict.windows}
    assert windows["15m"].available is False
    assert "not a whole number" in (windows["15m"].reason or "")
    # A 1d window on a daily bar resolves to a single bar, below the minimum of two.
    assert windows["1d"].available is False


def test_every_advertised_window_has_a_reason_when_it_is_unavailable() -> None:
    for spec in ALL:
        if spec.param is None:
            continue
        for ctx in (
            context(),
            context(timeframe=Timeframe.D1),
            context(timeframe=Timeframe.W1),
            context(asset_class=AssetClass.EQUITY_SPOT, timeframe=Timeframe.D1),
        ):
            verdict = cat.availability(spec, ctx)
            if not verdict.available and verdict.windows:
                assert verdict.reason, f"{spec.id} is unavailable with no explanation"
            for window in verdict.windows:
                if not window.available:
                    assert window.reason, f"{spec.id}/{window.token} has no reason"


# ------------------------------------------------------------------- custom windows


def test_a_custom_bar_count_is_accepted() -> None:
    option = cat.resolve_custom(cat.CATALOGUE["sma"], context(), "37")
    assert option.available
    assert option.bars == 37
    assert option.token == "37"


def test_a_custom_bar_count_that_is_not_a_number_is_refused() -> None:
    option = cat.resolve_custom(cat.CATALOGUE["sma"], context(), "4h")
    assert not option.available
    assert "count of candles" in (option.reason or "")


def test_a_custom_interval_below_the_minimum_is_refused() -> None:
    # `realized_vol` takes an interval, not a count: on this hourly dataset "1h" resolves to
    # a single bar, and one bar's realized volatility is an absolute return wearing a
    # volatility's name. The refusal names the floor rather than rounding the window up.
    option = cat.resolve_custom(cat.CATALOGUE["realized_vol"], context(), "1h")
    assert not option.available
    assert "at least 2" in (option.reason or "")


def test_a_custom_interval_is_resolved_against_the_dataset() -> None:
    option = cat.resolve_custom(cat.CATALOGUE["roc"], context(timeframe=Timeframe.H1), "6h")
    assert option.available
    assert option.bars == 6
    assert option.token == "6h"


def test_an_absurd_window_is_refused_rather_than_attempted() -> None:
    option = cat.resolve_custom(cat.CATALOGUE["sma"], context(), "99999999")
    assert not option.available


# ------------------------------------------------------------------------- the payload


def test_spec_payload_carries_what_the_ui_renders() -> None:
    spec = cat.CATALOGUE["bollinger"]
    payload = cat.spec_payload(spec, cat.availability(spec, context()))
    assert payload["formula"] == spec.formula
    assert payload["detail"] == spec.detail
    assert payload["group"] == "indicators"
    assert payload["kind"] == "computed"
    assert payload["columns"] == list(spec.columns)
    assert payload["available"] is True
    assert payload["param"] is None
    assert payload["windows"] == []


def test_spec_payload_carries_a_reason_when_unavailable() -> None:
    spec = cat.CATALOGUE["last_funding_rate"]
    verdict = cat.availability(spec, context(asset_class=AssetClass.EQUITY_SPOT))
    payload = cat.spec_payload(spec, verdict)
    assert payload["available"] is False
    assert payload["reason"]


def test_the_payload_reports_the_warm_up_per_window() -> None:
    """A single warm-up number for a parameterised feature would be wrong for every tick
    but one, so it is reported against the window it belongs to."""
    spec = cat.CATALOGUE["sma"]
    payload = cat.spec_payload(spec, cat.availability(spec, context()))
    assert payload["warmup_bars"] is None
    by_token = {window["token"]: window for window in payload["windows"]}
    assert by_token["20"]["bars"] == 20
    assert by_token["20"]["warmup_bars"] == 19
    assert by_token["100"]["warmup_bars"] == 99
    # The same ticks on a daily dataset: a bar count needs no declared bar length, so every
    # window still resolves and still carries its warm-up.
    daily = cat.availability(spec, context(timeframe=Timeframe.D1))
    payload = cat.spec_payload(spec, daily)
    assert all(window["warmup_bars"] is not None for window in payload["windows"])

    # A window that cannot be resolved has no bar count and therefore no warm-up. The payload
    # reports the absence rather than a number derived from a bar count it does not have —
    # which is what stops the UI from printing "warm-up: 14 bars" beside a disabled checkbox.
    interval = cat.CATALOGUE["realized_vol"]
    payload = cat.spec_payload(interval, cat.availability(interval, context(timeframe=None)))
    assert payload["available"] is False
    assert payload["windows"]
    assert all(window["bars"] is None for window in payload["windows"])
    assert all(window["warmup_bars"] is None for window in payload["windows"])


def test_the_payload_reports_a_warm_up_for_a_parameterless_feature() -> None:
    spec = cat.CATALOGUE["bollinger"]
    payload = cat.spec_payload(spec, cat.availability(spec, context()))
    assert payload["warmup_bars"] == 19
    assert payload["windows"] == []


def test_every_window_that_resolves_has_a_non_negative_warm_up() -> None:
    """A full window of n bars cannot exist before row n−1, so no warm-up is below that."""
    for spec in ALL:
        if spec.param is None:
            continue
        for ctx in (context(), context(timeframe=Timeframe.D1), context(timeframe=Timeframe.W1)):
            for window in cat.spec_payload(spec, cat.availability(spec, ctx))["windows"]:
                if window["bars"] is None:
                    assert window["warmup_bars"] is None
                    continue
                assert window["warmup_bars"] >= window["bars"] - 1, (
                    f"{spec.id}/{window['token']} claims to have a value before a full "
                    "window of input exists"
                )
