"""D11 approved-model inference, causal confirmation and monitoring."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from app.core.errors import ValidationFailure
from app.models.enums import TIMEFRAME_SECONDS
from app.services.regimes.data import matrix, preprocessing, read, replay, signature, times_of
from app.services.regimes.evaluation import reopen, require_predictive_evidence
from app.services.regimes.metrics import OCCUPANCY_NOTE, occupancy_drift, runs, transitions
from app.services.regimes.structural import cusum


def confirm(states, probabilities, minimum=3, threshold=0.55, hysteresis=0.1):
    """Causal confirmation; invalid observations reset pending persistence, retaining state."""
    current = None
    pending = None
    length = 0
    confirmed = []
    events = []
    for i, state in enumerate(states):
        state = int(state)
        if state < 0:
            pending, length = None, 0
            confirmed.append(-1 if current is None else current)
            continue
        if current is None:
            current = state
        confidence = probabilities[i, state] if probabilities is not None else None
        old_probability = probabilities[i, current] if probabilities is not None else None
        eligible = state != current and (
            confidence is None
            or confidence >= threshold
            and confidence - old_probability >= hysteresis
        )
        if eligible:
            length = length + 1 if pending == state else 1
            pending = state
            if length >= minimum:
                events.append(
                    {
                        "index": i,
                        "from_state": current,
                        "to_state": state,
                        "confirmation_delay_bars": length - 1,
                    }
                )
                current, pending, length = state, None, 0
        else:
            pending, length = None, 0
        confirmed.append(current)
    return np.asarray(confirmed), events


def feature_drift(train, current, features):
    output = []
    for j, feature in enumerate(features):
        edges = np.unique(np.quantile(train[:, j], np.linspace(0, 1, 11)))
        if len(edges) < 3:
            scale = max(float(train[:, j].std()), 1e-8)
            drift = float(np.mean(np.abs(current[:, j] - train[:, j].mean()) > 3 * scale))
            method = "fraction outside constant-training tolerance"
        else:
            edges[0], edges[-1] = -np.inf, np.inf
            p = np.histogram(train[:, j], bins=edges)[0].astype(float)
            q = np.histogram(current[:, j], bins=edges)[0].astype(float)
            drift = float(jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2)
            method = "Jensen-Shannon divergence of training-quantile bins"
        output.append({"feature": feature, "drift": drift, "method": method})
    return output


def unchanged_features(times, x, saved_times, saved_x):
    """Compare model inputs in the approved coordinates, allowing replay roundoff only."""
    positions = pd.DatetimeIndex(saved_times).get_indexer(times)
    overlap = positions >= 0
    if not np.allclose(x[overlap], saved_x[positions[overlap]], rtol=1e-12, atol=1e-12):
        raise ValidationFailure(
            "Inference changed saved model features. Historical revisions require a separate "
            "experiment and approval; existing history cannot be revised."
        )


def infer(request, settings, catalog, store):
    approval = store.get(request.approval_id, "approval")
    artifact, config, prepared, model = reopen(approval["experiment_id"], settings, catalog, store)
    if model.family in {"supervised", "hybrid"}:
        validation = store.get(approval["validation_id"], "validation")
        require_predictive_evidence(model, validation["test"], validation.get("folds", []))
    if (
        approval["model_checksum"] != artifact["files"]["model.bin"]
        or approval["contract"] != artifact["contract"]
    ):
        raise ValidationFailure("Approved model/preprocessing identity no longer matches.")
    if model.name == "pelt":
        raise ValidationFailure(
            "PELT is an offline segmenter and cannot produce a live market regime."
        )
    meta, frame = read(catalog, settings, request.dataset_id)
    if (meta.symbol, meta.timeframe.value if meta.timeframe else None, meta.asset_class.value) != (
        approval["contract"]["asset"],
        approval["contract"]["timeframe"],
        approval["contract"]["asset_class"],
    ):
        raise ValidationFailure(
            "Inference asset, timeframe and market type must match the approved model."
        )
    _, times = times_of(meta, frame)
    as_of = pd.Timestamp(request.as_of) if request.as_of else pd.Timestamp.now(tz="UTC")
    if as_of.tzinfo is None:
        raise ValidationFailure("Inference as_of must include an explicit timezone.")
    if as_of > pd.Timestamp.now(tz="UTC"):
        raise ValidationFailure("Live inference cannot use a future prediction origin.")
    if "close_time" in frame:
        available = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")
        if available.isna().any() or (available < times).any():
            raise ValidationFailure("Invalid upstream bar availability times.")
    else:
        available = times + pd.to_timedelta(TIMEFRAME_SECONDS[meta.timeframe], unit="s")
    # Truncate before feature extraction or fitted preprocessing replay.
    eligible = available <= as_of
    frame = frame.loc[eligible].reset_index(drop=True)
    times = times.loc[eligible].reset_index(drop=True)
    available = available.loc[eligible].reset_index(drop=True)
    source_steps = preprocessing(catalog, settings, meta)
    price_name = config.inputs.price_column
    transformed_price = any(
        step["kind"] == "scaling" and any(p["feature"] == price_name for p in step["parameters"])
        for step in source_steps
    )
    prices = frame[price_name].copy() if price_name in frame and not transformed_price else None
    train_end = pd.Timestamp(approval["contract"]["train_end"])
    if frame.empty or not (times > train_end).any():
        raise ValidationFailure(
            "No completed post-training observations are available at this origin."
        )
    if request.input_mode == "prepared":
        steps = source_steps
        if signature(steps) != approval["contract"]["preprocessing_signature"]:
            raise ValidationFailure(
                "Prepared inference data uses different fitted preprocessing. "
                "Replay the approved artifacts on the upstream feature dataset."
            )
    else:
        if source_steps:
            raise ValidationFailure(
                "Replay expects the original upstream feature representation; "
                "this dataset is already transformed."
            )
        frame = replay(frame, approval["contract"]["preprocessing"])
    x = matrix(frame, model.features)
    after = np.flatnonzero((times > train_end).to_numpy())
    # Require the first post-fit observation, avoiding an implicit HMM restart after a gap.
    original_after = prepared.times[prepared.times > train_end]
    if len(original_after) and times.iloc[after[0]] != original_after.iloc[0]:
        raise ValidationFailure(
            "Inference needs the complete post-training sequence beginning with the "
            "first saved out-of-sample observation."
        )
    known = original_after[original_after <= times.iloc[after[-1]]]
    if not known.isin(times.iloc[after]).all():
        raise ValidationFailure("Inference history omits saved post-training observations.")
    unchanged_features(times, x, prepared.times, prepared.x)
    # Saved runs can extend beyond the experiment. Their checksummed source datasets
    # retain the accepted inputs, so legacy bundles need no migration or new files.
    for saved in store.list("inference"):
        if saved.get("experiment_id") != artifact["id"]:
            continue
        saved_output = store.frame(saved["id"])
        saved_times = pd.to_datetime(saved_output["timestamp"], utc=True)
        required = saved_times[saved_times <= times.iloc[after[-1]]]
        if not required.isin(times.iloc[after]).all():
            raise ValidationFailure("Inference history omits saved post-training observations.")
        saved_request = saved["request"]
        saved_meta, saved_frame = read(catalog, settings, saved_request["dataset_id"])
        _, source_times = times_of(saved_meta, saved_frame)
        selected = source_times.isin(required)
        saved_frame = saved_frame.loc[selected].reset_index(drop=True)
        if saved_request["input_mode"] == "replay":
            saved_frame = replay(saved_frame, approval["contract"]["preprocessing"])
        unchanged_features(times, x, source_times[selected], matrix(saved_frame, model.features))
    history_end = int(prepared.spans["train"][-1]) + 1
    history = prepared.x[:history_end]
    combined = np.vstack([history, x[after]])
    result = model.predict(combined, request.decoding, request.lag)
    states = result["states"][history_end:]
    probabilities = (
        result["probabilities"][history_end:] if result["probabilities"] is not None else None
    )
    distances = result["distances"][history_end:] if result["distances"] is not None else None
    confirmed, events = confirm(
        states,
        probabilities,
        request.minimum_persistence,
        request.probability_threshold,
        request.hysteresis,
    )
    current_times = times.iloc[after].reset_index(drop=True)
    output = pd.DataFrame(
        {
            "timestamp": current_times.astype(str),
            "available_at": available.iloc[after].astype(str).to_numpy(),
            "state_id": states,
            "confirmed_state_id": confirmed,
        }
    )
    labels = approval["labels"]
    output["regime_label"] = [
        labels.get(str(s), f"Segment {s}" if model.family == "structural" else f"State {s}")
        for s in confirmed
    ]
    if prices is not None:
        output["price"] = prices.iloc[after].to_numpy()
    elif price_name in prepared.market:
        original_prices = pd.Series(
            prepared.market[price_name].to_numpy(), index=pd.DatetimeIndex(prepared.times)
        )
        output["price"] = original_prices.reindex(pd.DatetimeIndex(current_times)).to_numpy()
    if probabilities is not None:
        for k in range(model.count):
            output[f"probability_{k}"] = probabilities[:, k]
    if distances is not None:
        for k in range(model.count):
            output[f"distance_{k}"] = distances[:, k]
    for event in events:
        event["timestamp"] = str(current_times.iloc[event["index"]])
    current = int(confirmed[-1])
    confidence = float(probabilities[-1, current]) if probabilities is not None else None
    train_indices = prepared.spans["train"]
    train = prepared.x[train_indices]
    window = x[after][-min(250, len(after)) :]
    drift = feature_drift(train, window, model.features)
    unusual = np.max(
        np.abs((window[-1] - train.mean(axis=0)) / np.maximum(train.std(axis=0), 1e-8))
    )
    train_states = model.predict(prepared.x[:history_end])["states"][train_indices]
    state_drift = occupancy_drift(train_states, states[-250:], model.count) if model.count else None
    warnings = []
    if confidence is not None and confidence < request.low_confidence:
        warnings.append(
            {
                "kind": "low_confidence",
                "message": "Current confirmed-state probability is below the configured threshold.",
            }
        )
    if unusual > 5:
        warnings.append(
            {
                "kind": "unusual_observation",
                "message": "Latest observation exceeds five training standard deviations "
                "in at least one feature.",
            }
        )
    if any(row["drift"] > request.drift_threshold for row in drift):
        warnings.append(
            {
                "kind": "feature_drift",
                "message": "Recent feature distributions differ from the training reference.",
            }
        )
    if state_drift is not None and state_drift > request.drift_threshold:
        warnings.append(
            {"kind": "occupancy_drift", "message": "Recent state occupancy differs from training."}
        )
    cusum_events = []
    if request.cusum_threshold is not None:
        _, changes, _ = cusum(
            x[after],
            train.mean(axis=0),
            np.maximum(train.std(axis=0), 1e-8),
            request.cusum_threshold,
        )
        cusum_events = [{"index": i, "timestamp": str(current_times.iloc[i])} for i in changes]
    if model.name == "cusum":
        cusum_events = [
            {"index": i - history_end, "timestamp": str(current_times.iloc[i - history_end])}
            for i in result["alerts"]
            if i >= history_end
        ]
    retrospective = request.decoding != "filter"
    duration = runs(confirmed)[-1]["bars"]
    label_text = labels.get(str(current), f"State {current}")
    snapshot = {
        "state_id": current,
        "raw_state_id": int(states[-1]),
        "regime": label_text,
        "trend": next(
            (word for word in ["Bullish", "Bearish", "Sideways"] if word in label_text), None
        ),
        "volatility": next(
            (word for word in ["High Volatility", "Low Volatility"] if word in label_text), None
        ),
        "probabilities": probabilities[-1].tolist() if probabilities is not None else None,
        "distances": distances[-1].tolist() if distances is not None else None,
        "confidence": confidence,
        "duration_bars": duration,
        "previous_state": events[-1]["from_state"] if events else None,
        "last_transition": events[-1] if events else None,
        "model": model.name,
        "version": approval["version"],
        "asset": meta.symbol,
        "timeframe": meta.timeframe.value,
        "timestamp": str(current_times.iloc[-1]),
        "available_at": str(available.iloc[after[-1]]),
        "inferred_at": datetime.now(UTC).isoformat(),
        "objective": approval["objective"],
        "horizon": config.target.horizon
        if config.target and config.target.objective == "future"
        else 0,
        "retrospective": retrospective,
        "live_current_regime": not retrospective and approval["objective"] == "current",
    }
    # Performance deterioration requires observed targets; confidence/drift cannot stand in.
    performance = {
        "available": False,
        "reason": "Independent realized labels are required for performance monitoring.",
    }
    if model.family == "supervised" and config.target:
        from app.services.regimes.metrics import predictive

        monitor_data = type("MonitorData", (), {"frame": frame, "x": x})()
        # Score the predictions actually returned above, including their saved
        # training context. Scoring does not run a second inference pipeline.
        scoring_states = np.full(len(x), -1, dtype=int)
        scoring_probabilities = np.full((len(x), model.count), np.nan)
        scoring_states[after] = states
        scoring_probabilities[after] = probabilities
        metric = predictive(model, monitor_data, config, after,
                            {"states": scoring_states, "probabilities": scoring_probabilities})
        performance = {"available": metric.get("observations", 0) > 0, "metrics": metric}
        if not performance["available"]:
            performance["reason"] = metric["unavailable"]
        validation = store.get(approval["validation_id"], "validation")
        reference_accuracy = validation["test"]["predictive"].get("accuracy")
        if reference_accuracy is not None and metric.get("accuracy", 1) < reference_accuracy - 0.1:
            warnings.append(
                {
                    "kind": "performance_deterioration",
                    "message": "Recent target accuracy is more than 10 percentage points "
                    "below the approved test reference.",
                }
            )
    count = model.count or int(confirmed.max() + 1)
    transition = (
        transitions(confirmed, count, current_times.astype(str).tolist())
        if count <= 100
        else {"unavailable": "More than 100 sequential segments; export the event timeline."}
    )
    return store.save(
        "inference",
        {
            "name": artifact["name"],
            "approval_id": approval["id"],
            "experiment_id": artifact["id"],
            "request": request.model_dump(mode="json"),
            "snapshot": snapshot,
            "labels": labels,
            "events": events,
            "transitions": transition,
            "monitoring": {
                "warnings": warnings,
                "feature_drift": drift,
                "state_occupancy_drift": state_drift,
                "occupancy_note": OCCUPANCY_NOTE,
                "cusum_events": cusum_events,
                "performance": performance,
            },
            "observation_count": len(output),
            "causal": not retrospective,
            "decoding_note": "Retrospective: state estimates can use later observations "
            "in the selected history."
            if retrospective
            else "Causal filtering: only completed observations available at each origin.",
        },
        meta,
        frame=output,
    )
