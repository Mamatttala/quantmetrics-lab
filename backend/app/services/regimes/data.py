"""Adapter to the existing dataset, C6, C7 and C8 contracts; never fits preprocessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.errors import DatasetNotFound, ValidationFailure
from app.models.scaling import ScalerParameter
from app.models.splits import SplitPlanRow
from app.services.datasets import resolved_path
from app.services.exploratory import observation_column
from app.services.exploratory_pca import transform_pca
from app.services.scaling import _apply
from app.services.splits import _from_record
from app.storage import parquet_store


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(catalog, settings, dataset_id):
    meta = catalog.get_dataset(dataset_id)
    if meta is None:
        raise DatasetNotFound(f"Dataset {dataset_id} does not exist in this project.")
    path = resolved_path(meta, settings)
    if digest(path) != meta.checksum:
        raise ValidationFailure(
            f"Dataset {dataset_id} checksum changed. Rebuild its upstream artifacts."
        )
    return meta, parquet_store.read_dataset(path)


def times_of(meta, frame):
    column = observation_column(meta)
    if not column or column not in frame:
        raise ValidationFailure("D requires an upstream observation timestamp column.")
    times = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
        raise ValidationFailure(
            "Observation times must be unique and strictly increasing. Correct them in B3/B4."
        )
    return column, times


def lineage_detail(catalog, meta):
    with catalog._lock:
        row = catalog.conn.execute(
            "SELECT detail FROM lineage WHERE child_id = ? AND parent_id = ? "
            "AND operation = ? ORDER BY rowid LIMIT 1",
            (meta.dataset_id, meta.parent_dataset_id, meta.operation),
        ).fetchone()
    if row is None:
        raise ValidationFailure(f"Missing fitted artifact lineage for {meta.dataset_id}.")
    return json.loads(row["detail"])


def contributing_lineage(catalog, dataset_id):
    """F1: dependency order across every parent, without changing Catalog.ancestors."""
    ordered, visited, active = [], set(), set()

    def visit(identifier):
        if identifier in active:
            raise ValidationFailure("Upstream lineage contains a cycle.")
        if identifier in visited:
            return
        if len(active) >= 64:
            raise ValidationFailure("Upstream lineage is too deep to verify safely.")
        item = catalog.get_dataset(identifier)
        if item is None:
            raise ValidationFailure("An upstream dataset was deleted; lineage cannot be verified.")
        active.add(identifier)
        with catalog._lock:
            rows = catalog.conn.execute(
                "SELECT parent_id FROM lineage WHERE child_id = ? ORDER BY rowid", (identifier,)
            ).fetchall()
        parents = [item.parent_dataset_id, *(row["parent_id"] for row in rows)]
        for parent in dict.fromkeys(p for p in parents if p is not None):
            visit(parent)
        active.remove(identifier)
        visited.add(identifier)
        ordered.append(identifier)

    visit(dataset_id)
    return ordered


def preprocessing(catalog, settings, meta):
    ids = contributing_lineage(catalog, meta.dataset_id)
    primary = {meta.dataset_id, *catalog.ancestors(meta.dataset_id)}
    steps = []
    for dataset_id in ids:
        item = catalog.get_dataset(dataset_id)
        if item is None:
            raise ValidationFailure(
                "An upstream dataset was deleted; its preprocessing cannot be verified."
            )
        if item.operation not in {"scaling", "pca_dimensionality_reduction"}:
            continue
        read(catalog, settings, dataset_id)
        detail = lineage_detail(catalog, item)
        if item.operation == "scaling":
            parameters = detail.get("scaler_parameters", [])
            resolved = detail.get("resolved", {})
            step = {
                "kind": "scaling",
                "parameters": parameters,
                "request": detail.get("parameters_used", {}),
                "fit_start": resolved.get("fit_start_time"),
                "fit_end": resolved.get("fit_end_time"),
                "fit_role": resolved.get("fit_role"),
            }
        else:
            parameters = json.loads(item.provenance.get("pca_parameters", "{}"))
            if not parameters:
                raise ValidationFailure("C8 PCA parameters are unavailable.")
            step = {
                "kind": "pca",
                "parameters": parameters,
                "fit_start": parameters.get("scope_start"),
                "fit_end": parameters.get("scope_end"),
                "fit_role": parameters.get("scope"),
            }
        if dataset_id not in primary:
            # Validate and pin branch artifacts, but do not apply them to another
            # branch's columns. D has no upstream merge/prefix replay interface.
            step["branch_only"] = True
        steps.append({**step, "dataset_id": dataset_id, "checksum": item.checksum})
    return steps


def check_preprocessing(steps, train_start, train_end, *, context="training"):
    for step in steps:
        if step["kind"] == "scaling" and step["request"].get("scaler_type") == "rolling_zscore":
            continue
        if not step["fit_start"] or not step["fit_end"]:
            raise ValidationFailure(
                "The upstream fitted artifact has no verifiable fit span. "
                "Fit it against a C6 training span."
            )
        if (
            pd.Timestamp(step["fit_start"]) < train_start
            or pd.Timestamp(step["fit_end"]) > train_end
        ):
            raise ValidationFailure(
                f"Preprocessing was fitted outside this {context} window. "
                "Supply the matching C7/C8 artifact for this C6 fold."
            )


def signature(steps):
    # Identity includes exact fitted values and source checksums, not only column names.
    return hashlib.sha256(json.dumps(steps, sort_keys=True, default=str).encode()).hexdigest()


def replay(frame, steps):
    if any(step.get("branch_only") for step in steps):
        raise ValidationFailure(
            "Merged preprocessing includes branch-specific transforms. "
            "Supply the verified prepared dataset; branch replay is unsupported."
        )
    result = frame.copy()
    for step in steps:
        if step["kind"] == "scaling":
            if step["request"].get("scaler_type") == "rolling_zscore":
                raise ValidationFailure(
                    "Rolling C7 replay needs its original history. "
                    "Supply the verified prepared series instead."
                )
            params = [ScalerParameter.model_validate(p) for p in step["parameters"]]
            missing = [p.feature for p in params if p.feature not in result]
            if missing:
                raise ValidationFailure(f"C7 replay is missing upstream features: {missing}.")
            result = _apply(result, params)
        else:
            parameters = step["parameters"]
            pre = parameters.get("preprocessing_parameters", {})
            if pre.get("pre_transform", "none") != "none" or pre.get("imputation"):
                raise ValidationFailure(
                    "This C8 preprocessing chain has no public fitted replay interface. "
                    "Supply its verified prepared output."
                )
            scores = transform_pca(result, parameters)
            result[scores.columns] = scores
    return result


def plan_rows(catalog, settings, plan_id):
    meta, frame = read(catalog, settings, plan_id)
    if meta.operation != "split":
        raise ValidationFailure("Choose an existing C6 chronological split plan.")
    rows = [_from_record(record) for record in frame.to_dict("records")]
    source = catalog.get_dataset(meta.parent_dataset_id)
    if source is None or any(row.dataset_checksum != source.checksum for row in rows):
        raise ValidationFailure("C6 split plan references a different source version.")
    read(catalog, settings, source.dataset_id)
    return meta, rows


def matrix(frame, features):
    missing = [f for f in features if f not in frame]
    if missing:
        raise ValidationFailure(f"Missing upstream features: {missing}. Configure them in C5–C8.")
    try:
        values = frame[features].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValidationFailure("Model features must be numeric upstream columns.") from exc
    if not np.isfinite(values).all():
        raise ValidationFailure(
            "Features contain missing or infinite values. "
            "Resolve them upstream; D does not impute or drop rows."
        )
    return values


@dataclass
class Prepared:
    meta: object
    frame: pd.DataFrame
    times: pd.Series
    time_column: str
    x: np.ndarray
    spans: dict[str, np.ndarray]
    plan: list[SplitPlanRow]
    steps: list[dict]
    market: pd.DataFrame
    contract: dict


def load(catalog, settings, inputs):
    meta, frame = read(catalog, settings, inputs.dataset_id)
    column, times = times_of(meta, frame)
    if not meta.symbol or not meta.timeframe:
        raise ValidationFailure("D requires the upstream asset and timeframe metadata.")
    if meta.operation in {"split", "scaler_params", "pca_parameters"}:
        raise ValidationFailure("Select prepared observations, not a parameter or split table.")
    plan_meta, all_rows = plan_rows(catalog, settings, inputs.split_plan_id)
    ancestors = [meta.dataset_id, *catalog.ancestors(meta.dataset_id)]
    if plan_meta.parent_dataset_id not in ancestors:
        raise ValidationFailure("The C6 plan does not belong to this dataset's upstream lineage.")
    rows = [row for row in all_rows if row.fold == inputs.fold]
    spans = {
        role: np.flatnonzero(
            np.logical_or.reduce(
                [
                    (times >= row.start_time) & (times <= row.end_time)
                    for row in rows
                    if row.role.value == role
                ]
            )
        )
        if any(row.role.value == role for row in rows)
        else np.array([], dtype=int)
        for role in ("train", "val", "test")
    }
    train = spans["train"]
    if len(train) < 12 or not len(spans["test"]):
        raise ValidationFailure(
            "The selected prepared dataset must contain this fold's training and test "
            "observations (at least 12 training rows)."
        )
    for role in ("val", "test"):
        if len(spans[role]) and spans[role][0] <= train[-1]:
            raise ValidationFailure("Validation/test must strictly follow training.")
    if len(spans["val"]) and spans["val"][-1] >= spans["test"][0]:
        raise ValidationFailure("Validation and test spans overlap or are reversed.")
    if not np.all(np.diff(train) == 1):
        raise ValidationFailure("A model training window must be contiguous.")
    for row in rows:
        actual = ((times >= row.start_time) & (times <= row.end_time)).sum()
        if actual != row.row_count:
            raise ValidationFailure(
                "Prepared rows do not match C6 spans. "
                "Rebuild the upstream plan after changing observations."
            )
    steps = preprocessing(catalog, settings, meta)
    check_preprocessing(steps, times.iloc[train[0]], times.iloc[train[-1]])
    market_id = inputs.market_dataset_id
    if not market_id:
        # Economic diagnostics use the nearest pre-scaling ancestor with price data.
        candidates = ancestors
        for item_id in candidates:
            item = catalog.get_dataset(item_id)
            if (
                item
                and not any(s["kind"] == "scaling" for s in preprocessing(catalog, settings, item))
                and any(c.name == inputs.price_column for c in item.columns)
            ):
                market_id = item_id
                break
    market = pd.DataFrame(index=frame.index)
    market_meta = None
    if market_id:
        market_meta, source = read(catalog, settings, market_id)
        _, market_times = times_of(market_meta, source)
        if (market_meta.symbol, market_meta.timeframe) != (meta.symbol, meta.timeframe):
            raise ValidationFailure("Economic observations must have the same asset and timeframe.")
        economic_names = {
            inputs.price_column,
            inputs.return_column,
            inputs.adx_column,
            inputs.rsi_column,
            inputs.volatility_column,
        } - {None}
        for step in preprocessing(catalog, settings, market_meta):
            if step["kind"] == "scaling" and any(
                row["feature"] in economic_names for row in step["parameters"]
            ):
                raise ValidationFailure(
                    "Economic diagnostics require original-unit market columns."
                )
        source = source.set_index(pd.DatetimeIndex(market_times))
        market = source.reindex(pd.DatetimeIndex(times)).reset_index(drop=True)
    contract = {
        "dataset_id": meta.dataset_id,
        "dataset_checksum": meta.checksum,
        "split_plan_id": plan_meta.dataset_id,
        "split_checksum": plan_meta.checksum,
        "fold": inputs.fold,
        "features": inputs.features,
        "asset": meta.symbol,
        "asset_class": meta.asset_class.value,
        "timeframe": meta.timeframe.value,
        "time_column": column,
        "preprocessing": steps,
        "preprocessing_signature": signature(steps),
        "train_start": times.iloc[train[0]].isoformat(),
        "train_end": times.iloc[train[-1]].isoformat(),
        "market_dataset_id": market_id,
        "market_checksum": market_meta.checksum if market_meta else None,
    }
    return Prepared(
        meta,
        frame,
        times,
        column,
        matrix(frame, inputs.features),
        spans,
        rows,
        steps,
        market,
        contract,
    )
