"""Names describe a result; stable dataset IDs identify its immutable file."""
from __future__ import annotations

import ntpath
import re
from dataclasses import dataclass
from typing import Any

from app.models.enums import DatasetCategory

NAMING_VERSION = "2"


@dataclass(frozen=True)
class Operation:
    suffix: str
    category: DatasetCategory
    page: str
    kind: str = "observations"


OPERATIONS = {
    "fetch": Operation("raw", DatasetCategory.RAW, "A1"),
    "upload": Operation("uploaded", DatasetCategory.UPLOADED, "Upload"),
    "edit": Operation("edited", DatasetCategory.PROCESSED, "DM"),
    "clean": Operation("cleaned", DatasetCategory.PROCESSED, "B3"),
    "resample": Operation("resampled", DatasetCategory.PROCESSED, "B4"),
    "merge": Operation("merged", DatasetCategory.PROCESSED, "B4"),
    "features": Operation("features", DatasetCategory.PROCESSED, "C5"),
    "split_partition": Operation("split", DatasetCategory.PROCESSED, "C6"),
    "split": Operation("split_plan", DatasetCategory.OUTPUTS, "C6", "split_plan"),
    "scaling": Operation("scaled", DatasetCategory.PROCESSED, "C7"),
    "scaler_params": Operation("scaler_parameters", DatasetCategory.OUTPUTS, "C7", "parameters"),
    "pca_dimensionality_reduction": Operation("pca", DatasetCategory.PROCESSED, "C8"),
    "pca_parameters": Operation("pca_parameters", DatasetCategory.OUTPUTS, "C8", "parameters"),
    "regime_experiment": Operation("regime_experiment", DatasetCategory.OUTPUTS, "D9", "model"),
    "regime_search": Operation("regime_search", DatasetCategory.OUTPUTS, "D9", "configuration"),
    "regime_validation": Operation("regime_validation", DatasetCategory.OUTPUTS, "D10", "report"),
    "regime_labels": Operation("regime_labels", DatasetCategory.OUTPUTS, "D10", "table"),
    "regime_robustness": Operation("regime_robustness", DatasetCategory.OUTPUTS, "D10", "report"),
    "regime_approval": Operation("regime_approval", DatasetCategory.OUTPUTS, "D10", "selection"),
    "regime_inference": Operation("regime_inference", DatasetCategory.OUTPUTS, "D11", "assignments"),
    "volatility": Operation("volatility_forecast", DatasetCategory.FORECAST, "E1"),
    "return_prediction": Operation(
        "return_forecast", DatasetCategory.FORECAST, "Return Prediction"
    ),
    "model": Operation("model", DatasetCategory.MODEL, "Models", "model"),
    "report": Operation("report", DatasetCategory.OUTPUTS, "Reports", "report"),
    "table": Operation("table", DatasetCategory.OUTPUTS, "Tables", "table"),
    "visual": Operation("visual", DatasetCategory.OUTPUTS, "Visuals", "visual"),
}


def category_for(operation: str | None, fallback: DatasetCategory) -> DatasetCategory:
    # Full-history PCA is exploratory; training-fitted PCA is preprocessing.
    if operation == "pca_dimensionality_reduction":
        return fallback
    spec = OPERATIONS.get(operation or "")
    return spec.category if spec else fallback


def effective_operation(operation: str | None, category: DatasetCategory) -> str:
    if operation:
        return operation
    return {DatasetCategory.RAW: "fetch", DatasetCategory.UPLOADED: "upload"}.get(category, "table")


def artifact_kind(operation: str | None, category: DatasetCategory) -> str:
    spec = OPERATIONS.get(effective_operation(operation, category))
    if spec:
        return spec.kind
    return "table" if category == DatasetCategory.OUTPUTS else "observations"


def origin_page(operation: str | None, category: DatasetCategory) -> str | None:
    spec = OPERATIONS.get(effective_operation(operation, category))
    return spec.page if spec else None


def slug(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_.") or "dataset"


def source_label(meta: Any) -> str:
    """Identify the original series without stacking every ancestor's name."""
    if meta.symbol:
        endpoint = meta.provenance.get("endpoint", "").lower()
        series = "funding" if "funding" in endpoint else "metrics" if "metrics" in endpoint else ""
        return "_".join(filter(None, [
            slug(meta.symbol), meta.timeframe.value if meta.timeframe else None,
            meta.asset_class.value, series,
        ]))
    original = meta.provenance.get("naming_source") or meta.provenance.get("uploaded_filename")
    return slug(ntpath.splitext(ntpath.basename(original or meta.name))[0])[:64]


def generated_name(source: str, operation: str, dataset_id: str,
                   provenance: dict[str, str] | None = None) -> str:
    info = provenance or {}
    spec = OPERATIONS.get(operation)
    suffix = spec.suffix if spec else slug(operation)
    variant_key = {
        "split_partition": "split_partition_label", "split": "split_type",
        "scaling": "scaler_type", "scaler_params": "scaler_type",
        "regime": "regime_model", "volatility": "models",
        "pca_dimensionality_reduction": "fit_scope", "pca_parameters": "fit_scope",
    }.get(operation, "")
    variant = info.get(variant_key, "")
    if variant:
        suffix += "_" + slug(variant)[:48]
    if operation == "scaling" and info.get("split_partition_label"):
        suffix += "_" + slug(info["split_partition_label"])
    dates = [info.get("naming_start", ""), info.get("naming_end", "")]
    span = "-".join(value[:10].replace("-", "") for value in dates if value)
    parts = [slug(source)[:80], span, suffix, dataset_id[-10:]]
    return "__".join(part for part in parts if part)


def named_metadata(meta: Any, *, custom: bool = False) -> Any:
    """Apply the format and retain the previous display name for an upgrade."""
    info = dict(meta.provenance)
    info.update(naming_source=source_label(meta), naming_version=NAMING_VERSION,
                naming_start=meta.start_timestamp.isoformat() if meta.start_timestamp else "",
                naming_end=meta.end_timestamp.isoformat() if meta.end_timestamp else "")
    canonical = generated_name(info["naming_source"],
                               effective_operation(meta.operation, meta.category),
                               meta.dataset_id, info)
    info["canonical_name"] = canonical
    info["naming_custom"] = "true" if custom else "false"
    if not custom and meta.name != canonical:
        info.setdefault("previous_name", meta.name)
    return meta.model_copy(update={"name": meta.name if custom else canonical, "provenance": info})
