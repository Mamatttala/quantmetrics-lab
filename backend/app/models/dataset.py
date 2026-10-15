"""Dataset metadata — the catalog record.

The spec enumerates these fields:

    dataset_id, name, source, symbol, market_type, timeframe, timezone,
    start_timestamp, end_timestamp, row_count, columns, parent_dataset_id,
    checksum, storage_path

All fourteen are modelled here as required-or-explicitly-null. Three fields are
additions, noted in DECISIONS.md:

* ``category``          — the Data Explorer's four buckets; the spec describes the
                          grouping in prose but does not name a field for it.
* ``unpopulated_fields``— columns that exist in the schema but are known-empty
                          (e.g. equity-futures settlement), so the UI can badge
                          them instead of presenting nulls as missing data.
* ``provenance``        — per-column notes for values this application derived
                          rather than received (e.g. funding_interval_hours).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.models.enums import AssetClass, DatasetCategory, Provider, Timeframe


class ColumnMeta(BaseModel):
    """Schema for a single column, carried in the catalog for offline introspection."""

    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str  # Arrow type as a string, e.g. "timestamp[ns, tz=UTC]" or "double"
    nullable: bool = True


class DatasetMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: str
    name: str
    category: DatasetCategory
    source: Provider
    asset_class: AssetClass

    # Null rather than absent when a concept does not apply to a dataset — an
    # absent key and a null value are indistinguishable after a JSON round trip,
    # so we commit to null.
    symbol: str | None = None
    market_type: str | None = None
    timeframe: Timeframe | None = None
    timezone: str | None = None

    start_timestamp: datetime | None = None
    end_timestamp: datetime | None = None
    row_count: int = Field(ge=0)

    columns: list[ColumnMeta] = Field(default_factory=list)
    parent_dataset_id: str | None = None

    #: sha256 of the canonical Parquet bytes, hex. This is the immutability witness:
    #: if it changes, the dataset changed.
    checksum: str

    storage_path: str

    created_at: datetime
    unpopulated_fields: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def operation(self) -> str | None:
        """Which transformation produced this dataset, e.g. ``"split"`` or ``"features"``.

        Read out of the provenance rather than stored twice, because `persist_dataset`
        already writes it there for every derived dataset and a second column could drift
        from it. It is lifted onto the model because callers have to *branch* on it — C6
        and C7 hide split-plan tables from their own dataset pickers, and a plan is
        otherwise indistinguishable from a frame of bars in a list.

        `None` for a dataset that was fetched or uploaded rather than derived: it is not
        the product of an operation inside this application.
        """
        return self.provenance.get("operation") or None


class DatasetSummary(BaseModel):
    """Explorer-list projection. Deliberately smaller than DatasetMeta so listing a
    few hundred datasets does not ship every column schema to the browser."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    name: str
    category: DatasetCategory
    source: Provider
    symbol: str | None
    timeframe: Timeframe | None
    row_count: int
    start_timestamp: datetime | None
    end_timestamp: datetime | None
    created_at: datetime
    parent_dataset_id: str | None

    #: The operation that produced this dataset, lifted out of the provenance for the same
    #: reason as on `DatasetMeta`. Named here rather than inherited from it because this
    #: model is the explorer's projection and carries no provenance to read it from.
    operation: str | None = None
