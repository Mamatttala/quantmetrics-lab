"""Dataset HTTP layer.

Request models are explicit rather than accepting loose dicts: the grid's edit
payloads are the one place a malformed request could corrupt a derived dataset, so
they are validated at the boundary.
"""

from __future__ import annotations

import io
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, File, Query, Response, UploadFile
from pydantic import BaseModel, Field, model_validator

from app.config import get_settings
from app.core.blocking import run_blocking
from app.core.errors import ValidationFailure
from app.models.dataset import DatasetMeta, DatasetSummary
from app.models.enums import (
    AlignMethod,
    AssetClass,
    DatasetCategory,
    DuplicatePolicy,
    MergeHow,
    MissingValuePolicy,
    OutlierPolicy,
    PriceSource,
    Provider,
    Timeframe,
)
from app.services.cleaning import CleaningPlan
from app.services.datasets import CellEdit, ColumnOp, DatasetService
from app.services.features import FeatureRequest, Selection
from app.services.merge import JoinSpec
from app.storage import parquet_store
from app.storage.catalog import Catalog

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _service() -> DatasetService:
    return DatasetService()


# ------------------------------------------------------------------- responses


class PreviewResponse(BaseModel):
    dataset_id: str
    row_count: int
    columns: list[dict[str, Any]]
    head: list[dict[str, Any]]
    tail: list[dict[str, Any]]


class PageResponse(BaseModel):
    dataset_id: str
    offset: int
    limit: int
    total: int
    rows: list[dict[str, Any]]


# -------------------------------------------------------------------- requests


class RenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class CellEditModel(BaseModel):
    row: int = Field(ge=0)
    column: str
    value: Any = None


class ColumnOpModel(BaseModel):
    op: Literal["rename", "drop", "add", "reorder"]
    column: str
    new_name: str | None = None
    # A place in the *final* schema, 0-based. Sent by the grid when a column is
    # dragged, and applied after every rename and add so it can name the column as it
    # will finally be called.
    position: int | None = Field(default=None, ge=0)


class DeriveRequest(BaseModel):
    name: str | None = None
    operation: str = "edit"
    edits: list[CellEditModel] = Field(default_factory=list)
    column_ops: list[ColumnOpModel] = Field(default_factory=list)


class CleanRequest(BaseModel):
    """A cleaning plan.

    Every policy defaults to "keep", so a request that omits a field repairs nothing
    rather than repairing something unasked.
    """

    name: str | None = None
    duplicates: DuplicatePolicy = DuplicatePolicy.KEEP
    missing: MissingValuePolicy = MissingValuePolicy.KEEP
    outliers: OutlierPolicy = OutlierPolicy.KEEP
    outlier_columns: list[str] = Field(default_factory=list)
    lower_quantile: float = Field(default=0.01, ge=0.0, lt=0.5)
    upper_quantile: float = Field(default=0.99, gt=0.5, le=1.0)
    iqr_multiplier: float = Field(default=1.5, gt=0.0, le=20.0)
    sort_by_time: bool = False
    #: `false` runs the plan and returns what it would do without storing anything.
    write: bool = True

    @model_validator(mode="after")
    def _check_quantiles(self) -> CleanRequest:
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError(
                "lower_quantile must be below upper_quantile; the fences would be "
                "inverted and every value would be treated as an outlier."
            )
        return self


class JoinRequest(BaseModel):
    dataset_id: str
    how: MergeHow = MergeHow.LEFT
    method: AlignMethod = AlignMethod.BACKWARD
    prefix: str = Field(default="", max_length=64)
    tolerance_seconds: int | None = Field(default=None, ge=1)


class MergeRequest(BaseModel):
    base_dataset_id: str
    name: str | None = None
    timeframe: Timeframe | None = None
    joins: list[JoinRequest] = Field(default_factory=list)
    column_ops: list[ColumnOpModel] = Field(default_factory=list)
    write: bool = True


class HeadTailPreview(BaseModel):
    row_count: int
    columns: list[dict[str, Any]]
    head: list[dict[str, Any]]
    tail: list[dict[str, Any]]


class FeatureSelectionModel(BaseModel):
    """One ticked feature, with the window the user chose for it.

    ``window`` is the token as the user spelled it — ``"20"`` for a count of candles,
    ``"4h"`` for an interval. The backend resolves a duration against the dataset's
    timeframe, so the request never has to know how long a bar is.
    """

    feature_id: str = Field(min_length=1, max_length=80)
    window: str | None = Field(default=None, max_length=20)


class FeatureRequestModel(BaseModel):
    """A feature selection.

    `price_source` is a field of the request rather than an option on each feature,
    because the specification offers exactly two price series and every price-derived
    feature is measured on whichever one is chosen. Per-feature choice would let one
    dataset mix two different prices under a single set of column names.
    """

    name: str | None = None
    selections: list[FeatureSelectionModel] = Field(default_factory=list)
    price_source: PriceSource = PriceSource.CLOSE
    #: Keep only the selected columns, plus the time column. The report names what went.
    keep_only_selected: bool = False
    write: bool = True


class MutateResponse(BaseModel):
    """Result of a cleaning or merge operation.

    `dataset` is null for a preview, which is the whole reason this is one shape: the
    frontend renders the same preview whether or not the operation was stored, so
    "what will this do?" and "what did this do?" cannot drift apart.
    """

    dataset: DatasetMeta | None = None
    report: dict[str, Any]
    preview: HeadTailPreview


# --------------------------------------------------------------------- routes


@router.get("", response_model=list[DatasetSummary])
async def list_datasets(
    category: DatasetCategory | None = Query(default=None),
) -> list[DatasetSummary]:
    return await _service().list_datasets(category)


@router.delete("")
async def delete_all_datasets() -> dict[str, int]:
    """Remove every dataset — the Data Explorer's "Delete" control.

    Declared on the collection path so it cannot be confused with the single-dataset
    DELETE below. Returns the count so the UI can report what actually happened.
    """
    return {"deleted": await _service().delete_all()}


@router.get("/{dataset_id}", response_model=DatasetMeta)
async def get_dataset(dataset_id: str) -> DatasetMeta:
    return await _service().get(dataset_id)


@router.get("/{dataset_id}/preview", response_model=PreviewResponse)
async def preview_dataset(
    dataset_id: str,
    head: int = Query(default=5, ge=0, le=100),
    tail: int = Query(default=5, ge=0, le=100),
) -> PreviewResponse:
    payload = await _service().preview(dataset_id, head, tail)
    return PreviewResponse(**payload)


@router.get("/{dataset_id}/page", response_model=PageResponse)
async def page_dataset(
    dataset_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1),
    sort: str | None = Query(default=None, description="column or column:desc"),
) -> PageResponse:
    payload = await _service().page(dataset_id, offset, limit, sort)
    return PageResponse(**payload)


@router.patch("/{dataset_id}", response_model=DatasetMeta)
async def rename_dataset(dataset_id: str, body: RenameRequest) -> DatasetMeta:
    return await _service().rename(dataset_id, body.name)


@router.post("/{dataset_id}/rename", response_model=DatasetMeta)
async def rename_dataset_action(dataset_id: str, body: RenameRequest) -> DatasetMeta:
    """Rename one catalog entry for the Data Explorer action.

    Keep the older PATCH endpoint above for API compatibility. The explicit action
    route is the browser client's mutation contract and makes the operation distinct
    from a partial metadata update without changing the immutable data file.
    """
    return await _service().rename(dataset_id, body.name)


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: str) -> Response:
    await _service().delete(dataset_id)
    return Response(status_code=204)


@router.post("/{dataset_id}/derive", response_model=DatasetMeta, status_code=201)
async def derive_dataset(dataset_id: str, body: DeriveRequest) -> DatasetMeta:
    """Apply edits and save as a new dataset.

    The response is the *child*. The parent is unchanged — the frontend then
    navigates to the new dataset, which is what makes the versioning rule visible
    to the user rather than a silent background behaviour.
    """
    return await _service().derive(
        dataset_id,
        name=body.name,
        operation=body.operation,
        edits=[CellEdit(row=e.row, column=e.column, value=e.value) for e in body.edits],
        column_ops=[
            ColumnOp(op=o.op, column=o.column, new_name=o.new_name, position=o.position)
            for o in body.column_ops
        ],
    )


@router.post("/{dataset_id}/clean", response_model=MutateResponse)
async def clean_dataset(
    dataset_id: str, body: CleanRequest, response: Response
) -> MutateResponse:
    """Apply a cleaning plan, optionally storing the result as a new dataset.

    `write=false` is a dry run: the same computation, the same report, the same
    preview, nothing stored. The frontend uses it to answer "what would this do?"
    before the user commits, which matters more here than elsewhere because the
    outlier policies are the only step in the pipeline that can silently discard
    rows.
    """
    plan = CleaningPlan(
        duplicates=body.duplicates,
        missing=body.missing,
        outliers=body.outliers,
        outlier_columns=tuple(body.outlier_columns),
        lower_quantile=body.lower_quantile,
        upper_quantile=body.upper_quantile,
        iqr_multiplier=body.iqr_multiplier,
        sort_by_time=body.sort_by_time,
    )
    payload = await _service().clean(dataset_id, plan, name=body.name, write=body.write)
    if body.write:
        response.status_code = 201
    return MutateResponse(**payload)


@router.post("/{dataset_id}/features", response_model=MutateResponse)
async def compute_features(
    dataset_id: str, body: FeatureRequestModel, response: Response
) -> MutateResponse:
    """Compute the selected features and optionally store them as a new dataset.

    `write=false` runs the whole computation and returns the report and preview without
    storing anything — the same dry run cleaning and merging offer, and for the same
    reason: the collinearity scan and the warm-up row count are things a user wants to see
    *before* committing a dataset, not after.
    """
    payload = await _service().features(
        dataset_id,
        FeatureRequest(
            selections=tuple(
                Selection(feature_id=item.feature_id, window=item.window)
                for item in body.selections
            ),
            price_source=body.price_source,
            keep_only_selected=body.keep_only_selected,
        ),
        name=body.name,
        write=body.write,
    )
    if body.write:
        response.status_code = 201
    return MutateResponse(**payload)


@router.post("/merge", response_model=MutateResponse)
async def merge_datasets(body: MergeRequest, response: Response) -> MutateResponse:
    """Align and join other datasets onto a base dataset.

    Declared before the `/{dataset_id}/...` routes below only for readability — the
    paths cannot collide, since this one has a single path segment where the others
    have two.
    """
    payload = await _service().merge(
        body.base_dataset_id,
        [
            JoinSpec(
                dataset_id=join.dataset_id,
                how=join.how,
                method=join.method,
                prefix=join.prefix,
                tolerance_seconds=join.tolerance_seconds,
            )
            for join in body.joins
        ],
        timeframe=body.timeframe,
        name=body.name,
        column_ops=[
            ColumnOp(op=o.op, column=o.column, new_name=o.new_name, position=o.position)
            for o in body.column_ops
        ],
        write=body.write,
    )
    if body.write:
        response.status_code = 201
    return MutateResponse(**payload)


@router.get("/{dataset_id}/ohlcv")
async def dataset_ohlcv(
    dataset_id: str,
    limit: int = Query(default=20000, ge=10, le=200000),
) -> dict[str, Any]:
    return await _service().ohlcv(dataset_id, limit)


@router.get("/{dataset_id}/quality")
async def dataset_quality(dataset_id: str) -> dict[str, Any]:
    return await _service().quality(dataset_id)


@router.get("/{dataset_id}/export")
async def export_dataset(
    dataset_id: str,
    format: Literal["csv", "json", "parquet"] = Query(default="csv"),
) -> Response:
    payload, media_type, filename = await _service().export(dataset_id, format)
    return Response(
        content=payload,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --------------------------------------------------------------------- upload


@router.post("/upload", response_model=DatasetMeta, status_code=201)
async def upload_csv(file: UploadFile = File(...)) -> DatasetMeta:
    """Ingest a user-supplied CSV as an immutable `uploaded_data` dataset.

    Uploads are made immutable for the same reason raw fetches are: the guarantee
    that a dataset does not change under you should not depend on where it came
    from. Editing an upload derives into processed_data, like anything else.
    """
    if file.filename is None:
        raise ValidationFailure("Upload has no filename.")
    if not file.filename.lower().endswith(".csv"):
        raise ValidationFailure(
            f"Only .csv uploads are supported, got {file.filename!r}.",
            hint="Convert other formats to CSV first.",
        )

    raw = await file.read()
    if not raw:
        raise ValidationFailure("Uploaded file is empty.")

    settings = get_settings()

    def work() -> DatasetMeta:
        try:
            frame = pd.read_csv(io.BytesIO(raw))
        except Exception as exc:  # pandas raises a wide variety of parse errors
            raise ValidationFailure(
                f"Could not parse the CSV: {exc}",
                hint="Check the delimiter and that the header row is present.",
            ) from exc

        if frame.empty:
            raise ValidationFailure("The uploaded CSV has a header but no rows.")

        dataset_id = parquet_store.new_dataset_id()
        written = parquet_store.write_dataset(
            frame, settings.data_dir, DatasetCategory.UPLOADED, dataset_id
        )

        meta = DatasetMeta(
            dataset_id=dataset_id,
            name=file.filename or dataset_id,
            category=DatasetCategory.UPLOADED,
            source=Provider.UPLOAD,
            # The one field below that is *not* genuinely unknown, because the catalog
            # column is NOT NULL and so something must be stored. This is a placeholder,
            # not a claim about the file — see DECISIONS §25 for what it costs and why
            # it is not simply left null.
            asset_class=AssetClass.CRYPTO_SPOT,
            symbol=None,
            market_type=None,
            timeframe=None,
            # Unknown at upload time. Left null rather than guessed: inferring
            # "this is 1d BTCUSDT" from a filename would be a fabrication.
            # Genuinely unknown. A CSV carries no timezone information, and guessing
            # one — from the filename, or from whether a column happens to be called
            # "utc" — would attach a confident-looking claim to data we know nothing
            # about. Null means "unstated", which is the truth.
            timezone=None,
            start_timestamp=written.start_timestamp,
            end_timestamp=written.end_timestamp,
            row_count=written.row_count,
            columns=written.columns,
            parent_dataset_id=None,
            checksum=written.checksum,
            storage_path=str(written.storage_path),
            created_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
            unpopulated_fields=[],
            provenance={
                "uploaded_filename": file.filename or "",
                "note": "Asset class and timeframe are unset until you supply them.",
            },
        )
        Catalog(settings.catalog_path).insert_dataset(meta)
        return meta

    return await run_blocking(work)
