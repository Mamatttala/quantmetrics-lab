"""D9 experiments → D10 validation/approval → D11 inference and integration."""

from typing import Literal

from fastapi import APIRouter, Query
from fastapi.responses import Response

from app.services.regimes import evaluation
from app.services.regimes.artifacts import clean
from app.services.regimes.contracts import (
    Approval,
    Comparison,
    Experiment,
    Inference,
    LabelRequest,
    RobustnessRequest,
    ValidationRequest,
)
from app.services.regimes.data import plan_rows
from app.services.regimes.service import RegimeService

router = APIRouter(prefix="/regime", tags=["regime"])


@router.get("/options")
def options(dataset_id: str | None = None):
    return clean(RegimeService().options(dataset_id))


@router.get("/plans/{identifier}")
def plan(identifier: str):
    service = RegimeService()
    _, rows = plan_rows(service.catalog, service.settings, identifier)
    return [row.model_dump(mode="json") for row in rows]


@router.post("/experiments", status_code=202)
async def fit(request: Experiment):
    return await RegimeService().submit("fit", request)


@router.get("/artifacts")
def artifacts(
    kind: Literal[
        "experiment", "search", "validation", "labels", "robustness", "approval", "inference"
    ] = "experiment",
):
    return RegimeService().store.list(kind)


@router.get("/artifacts/{identifier}")
def artifact(identifier: str):
    return RegimeService().store.get(identifier)


@router.get("/artifacts/{identifier}/history")
def history(identifier: str, offset: int = Query(0, ge=0), limit: int = Query(2500, ge=1, le=5000)):
    frame = RegimeService().store.frame(identifier)
    return {
        "rows": clean(frame.iloc[offset : offset + limit].to_dict("records")),
        "total": len(frame),
        "offset": offset,
    }


@router.get("/artifacts/{identifier}/export")
def export(identifier: str, format: Literal["json", "csv"] = "json"):
    service = RegimeService()
    result = service.store.get(identifier)
    if format == "json":
        if "assignments.parquet" in result["files"]:
            result["history"] = clean(service.store.frame(identifier).to_dict("records"))
        return result
    frame = service.store.frame(identifier)
    return Response(
        frame.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="regime-{identifier}.csv"'},
    )


@router.get("/artifacts/{identifier}/bundle")
def bundle(identifier: str):
    return Response(
        RegimeService().store.archive(identifier),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="regime-{identifier}.zip"'},
    )


@router.get("/experiments/{identifier}/evaluation")
def evaluate(identifier: str):
    service = RegimeService()
    return clean(evaluation.evaluate(identifier, service.settings, service.catalog, service.store))


@router.post("/experiments/{identifier}/labels", status_code=201)
def labels(identifier: str, request: LabelRequest):
    service = RegimeService()
    return clean(
        evaluation.label(identifier, request, service.settings, service.catalog, service.store)
    )


@router.post("/experiments/{identifier}/validation", status_code=202)
async def validation(identifier: str, request: ValidationRequest):
    return await RegimeService().submit("validation", request, identifier)


@router.post("/experiments/{identifier}/robustness", status_code=202)
async def robustness(identifier: str, request: RobustnessRequest):
    return await RegimeService().submit("robustness", request, identifier)


@router.post("/experiments/{identifier}/approval", status_code=201)
def approve(identifier: str, request: Approval):
    service = RegimeService()
    return clean(
        evaluation.approve(identifier, request, service.settings, service.catalog, service.store)
    )


@router.post("/comparison")
def compare(request: Comparison):
    service = RegimeService()
    return clean(
        evaluation.compare(request.experiment_ids, service.settings, service.catalog, service.store)
    )


@router.post("/inference", status_code=202)
async def infer(request: Inference):
    return await RegimeService().submit("inference", request)
