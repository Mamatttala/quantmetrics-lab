"""Phase 3 regressions: immutable D11 history and boundary behavior."""

import inspect

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.core.errors import ValidationFailure
from app.services.datasets import persist_dataset
from app.services.regimes.artifacts import Store
from app.services.regimes.contracts import Approval, Inference, Target, ValidationRequest
from app.services.regimes.data import read
from app.services.regimes.estimators import baseline, rule_labels
from app.services.regimes.evaluation import approve, validate
from app.services.regimes.experiments import execute
from app.services.regimes.inference import confirm, infer
from app.services.regimes.registry import parameters
from tests.conftest import register
from tests.test_regimes_workflow import config_of
from tests.test_regimes_workflow import prepared as prepared_fixture

prepared = prepared_fixture


@pytest.fixture
async def deployment(settings, catalog, prepared):
    store = Store(settings, catalog)
    config = config_of(prepared)
    identifier = execute(config, settings, catalog, store)["selected_experiment_id"]
    validation = validate(identifier, ValidationRequest(), settings, catalog, store)
    approval = approve(
        identifier,
        Approval(validation_id=validation["id"], rationale="History reference"),
        settings,
        catalog,
        store,
    )
    return store, approval


def observation_copy(settings, catalog, parent, frame, mode):
    if mode == "replay":
        return register(settings, catalog, frame)
    return persist_dataset(
        parent=parent,
        frame=frame,
        data_dir=settings.data_dir,
        catalog=catalog,
        category=parent.category,
        detail={},
        operation="derive",
        name="History candidate",
    )


@pytest.mark.parametrize("mode", ["prepared", "replay"])
@pytest.mark.parametrize("changed_row", [10, 120])
async def test_f6_revised_saved_features_refused(
    settings,
    catalog,
    prepared,
    deployment,
    mode,
    changed_row,
):
    store, approval = deployment
    source = prepared[0].dataset_id if mode == "replay" else prepared[2].dataset.dataset_id
    parent, frame = read(catalog, settings, source)
    frame.loc[changed_row, "signal"] += 1
    candidate = observation_copy(settings, catalog, parent, frame, mode)
    with pytest.raises(ValidationFailure, match="changed saved.*features"):
        infer(
            Inference(approval_id=approval["id"], dataset_id=candidate.dataset_id, input_mode=mode),
            settings,
            catalog,
            store,
        )
    assert store.list("inference") == []


@pytest.mark.parametrize("mode", ["prepared", "replay"])
async def test_f6_unchanged_overlap_and_appends_preserve_causal_history(
    settings,
    catalog,
    prepared,
    deployment,
    mode,
):
    store, approval = deployment
    source = prepared[0].dataset_id if mode == "replay" else prepared[2].dataset.dataset_id
    request = Inference(approval_id=approval["id"], dataset_id=source, input_mode=mode)
    original = infer(request, settings, catalog, store)
    parent, frame = read(catalog, settings, source)
    appended = frame.iloc[-1:].copy()
    for column in ["open_time", "close_time"]:
        appended[column] += pd.Timedelta("1h")
    extended = pd.concat([frame, appended], ignore_index=True)
    candidate = observation_copy(settings, catalog, parent, extended, mode)
    actual = infer(
        request.model_copy(update={"dataset_id": candidate.dataset_id}), settings, catalog, store
    )
    expected = store.frame(original["id"])
    pd.testing.assert_frame_equal(store.frame(actual["id"]).iloc[:-1], expected)
    # A later run may not revise an observation accepted by a saved appended run.
    extended.loc[160, "signal"] += 1
    revised = observation_copy(settings, catalog, parent, extended, mode)
    with pytest.raises(ValidationFailure, match="changed saved.*features"):
        infer(
            request.model_copy(update={"dataset_id": revised.dataset_id}), settings, catalog, store
        )
    pd.testing.assert_frame_equal(store.frame(original["id"]), expected)


def test_f6_explicit_revision_override_is_unsupported():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Inference(approval_id="approval", dataset_id="dataset", allow_revisions=True)


def test_f10_optional_entry_uses_existing_driver(monkeypatch):
    from tests import test_e2e_live

    called = []
    monkeypatch.setattr(test_e2e_live, "run_driver", lambda name: called.append(name))
    entry = test_e2e_live.test_regime_endpoints
    entry(**dict.fromkeys(inspect.signature(entry).parameters))
    assert called == ["verify_regimes_browser.py"]
    assert (test_e2e_live.BACKEND / "tools" / called[0]).is_file()


def test_f10_isolated_driver_does_not_restart_user_servers():
    from tests import test_e2e_live

    assert "servers" not in inspect.signature(test_e2e_live.test_regime_endpoints).parameters


@pytest.mark.parametrize("probabilistic", [False, True])
@pytest.mark.parametrize(
    "raw,expected,event_index",
    [
        ([0, 1, -1, 1], [0, 0, 0, 0], None),
        ([0, 1, -1, 1, 1], [0, 0, 0, 0, 1], 4),
        ([0, 1, -1, -1, 1, 1], [0, 0, 0, 0, 0, 1], 5),
        ([-1, -1, 0, 1, 1], [-1, -1, 0, 0, 1], 4),
        ([0, 1, 1, -1], [0, 0, 1, 1], 2),
    ],
)
def test_f12_invalid_observations_reset_pending_persistence(
    probabilistic,
    raw,
    expected,
    event_index,
):
    probabilities = (
        np.array(
            [
                [0.9, 0.1] if state == 0 else [0.1, 0.9] if state == 1 else [np.nan, np.nan]
                for state in raw
            ]
        )
        if probabilistic
        else None
    )
    actual, events = confirm(raw, probabilities, minimum=2)
    assert actual.tolist() == expected
    assert events == (
        []
        if event_index is None
        else [
            {
                "index": event_index,
                "from_state": 0,
                "to_state": 1,
                "confirmation_delay_bars": 1,
            }
        ]
    )
    for end in range(1, len(raw) + 1):
        prefix, _ = confirm(raw[:end], probabilities[:end] if probabilistic else None, minimum=2)
        np.testing.assert_array_equal(prefix, actual[:end])


@pytest.mark.parametrize(
    "band,expected",
    [
        (0, [0, 0, 1, 2, 2]),
        (1e-12, [0, 0, 1, 2, 2]),
        (0.125, [0, 1, 1, 1, 2]),
    ],
)
def test_f13_moving_average_band_boundaries(band, expected):
    p = parameters(
        "moving_average",
        {
            "fast_column": "fast",
            "slow_column": "slow",
            "neutral_band": band,
        },
    )
    x = np.column_stack([[6.0, 7.0, 8.0, 9.0, 10.0], np.full(5, 8.0)])
    assert baseline(x, ["fast", "slow"], "moving_average", p).tolist() == expected


@pytest.mark.parametrize("band", [-1e-12, -0.1, float("nan"), float("inf")])
def test_f13_invalid_bands_rejected_before_execution(band):
    with pytest.raises(ValidationFailure, match="neutral_band"):
        parameters(
            "moving_average",
            {
                "fast_column": "fast",
                "slow_column": "slow",
                "neutral_band": band,
            },
        )


def test_f13_other_rule_and_target_thresholds_remain_strict():
    for lower, upper in [(0, 0), (1, -1)]:
        with pytest.raises(ValidationFailure, match="Lower rule threshold"):
            rule_labels(np.array([-1, 0, 1]), lower, upper)
        with pytest.raises(ValidationError, match="Lower label threshold"):
            Target(source="rules", rule_feature="signal", lower=lower, upper=upper)


async def test_f13_zero_band_experiment_persists_and_reopens(settings, catalog, prepared):
    config = config_of(prepared, "moving_average")
    config.inputs.features = ["fast_ma", "slow_ma"]
    config.parameters = {"fast_column": "fast_ma", "slow_column": "slow_ma", "neutral_band": 0}
    store = Store(settings, catalog)
    identifier = execute(config, settings, catalog, store)["selected_experiment_id"]
    model = store.model(identifier)
    assert model.parameters["neutral_band"] == 0
    _, frame = read(catalog, settings, config.inputs.dataset_id)
    expected = np.where(
        frame.fast_ma < frame.slow_ma, 0, np.where(frame.fast_ma > frame.slow_ma, 2, 1)
    )
    np.testing.assert_array_equal(store.frame(identifier).state_id, expected)
    np.testing.assert_array_equal(
        model.predict(frame[config.inputs.features].to_numpy())["states"], expected
    )
