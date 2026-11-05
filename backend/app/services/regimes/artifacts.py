"""Immutable project-local bundles and checksummed fitted model bytes.

Model deserialization is confined to files written by this application. No uploaded
pickle or client-selected filesystem path is accepted.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import re
import zipfile
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from app.core.errors import ValidationFailure
from app.models.enums import DatasetCategory
from app.services.datasets import persist_dataset
from app.storage.parquet_store import new_dataset_id


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class Store:
    def __init__(self, settings, catalog):
        self.settings = settings
        self.catalog = catalog
        self.root = settings.outputs_dir / "regimes"

    def path(self, identifier):
        if not re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", identifier):
            raise ValidationFailure("Invalid regime artifact identifier.")
        return self.root / identifier

    def save(self, kind, payload, parent, model=None, frame=None):
        identifier = new_dataset_id()
        folder = self.path(identifier)
        folder.mkdir(parents=True, exist_ok=False)
        envelope = clean(
            {
                "schema_version": 1,
                "id": identifier,
                "kind": kind,
                "created_at": datetime.now(UTC).isoformat(),
                **payload,
            }
        )
        files = {}
        if model is not None:
            blob = pickle.dumps(model, protocol=5)
            (folder / "model.bin").write_bytes(blob)
            files["model.bin"] = hashlib.sha256(blob).hexdigest()
        if frame is not None:
            frame.to_parquet(folder / "assignments.parquet", index=False)
            files["assignments.parquet"] = hashlib.sha256(
                (folder / "assignments.parquet").read_bytes()
            ).hexdigest()
        envelope["files"] = files
        blob = json.dumps(envelope, sort_keys=True, allow_nan=False).encode()
        (folder / "manifest.json").write_bytes(blob)
        artifact = persist_dataset(
            parent=parent,
            frame=pd.DataFrame(
                {"manifest": [blob.decode()], "sha256": [hashlib.sha256(blob).hexdigest()]}
            ),
            data_dir=self.settings.data_dir,
            catalog=self.catalog,
            category=DatasetCategory.OUTPUTS,
            name=f"{kind}: {payload.get('name', identifier)}",
            operation=f"regime_{kind}",
            detail={"artifact_id": identifier},
            dataset_id=identifier,
        )
        return {**envelope, "catalog_dataset_id": artifact.dataset_id}

    def get(self, identifier, kind=None):
        from app.services.regimes.data import read

        meta, table = read(self.catalog, self.settings, identifier)
        if not meta.operation or not meta.operation.startswith("regime_"):
            raise ValidationFailure("This is not a new Module D artifact.")
        folder = self.path(identifier)
        manifest = folder / "manifest.json"
        if not manifest.is_file():
            raise ValidationFailure("Regime artifact manifest is missing.")
        blob = manifest.read_bytes()
        if hashlib.sha256(blob).hexdigest() != table["sha256"].iloc[0]:
            raise ValidationFailure("Regime manifest integrity check failed.")
        result = json.loads(blob)
        if kind and result["kind"] != kind:
            raise ValidationFailure(f"Expected a {kind} artifact.")
        for name, checksum in result["files"].items():
            if name not in {"model.bin", "assignments.parquet"}:
                raise ValidationFailure("Unknown artifact file.")
            path = folder / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValidationFailure("Regime artifact integrity check failed.")
        return result

    def model(self, identifier):
        self.get(identifier, "experiment")
        return pickle.loads((self.path(identifier) / "model.bin").read_bytes())

    def frame(self, identifier):
        self.get(identifier)
        return pd.read_parquet(self.path(identifier) / "assignments.parquet")

    def list(self, kind):
        output = []
        for row in self.catalog.list_datasets():
            if row.operation == f"regime_{kind}":
                try:
                    output.append(self.get(row.dataset_id, kind))
                except ValidationFailure:
                    # Damaged artifacts remain visible and can never be approved.
                    output.append({"id": row.dataset_id, "name": row.name, "integrity_error": True})
        return output

    def archive(self, identifier):
        buffer = io.BytesIO()
        visited = set()
        pending = [identifier]
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                manifest = self.get(current)
                visited.add(current)
                for name in ["manifest.json", *manifest["files"]]:
                    archive.write(self.path(current) / name, f"{current}/{name}")
                for key in ("experiment_id", "approval_id", "validation_id"):
                    if manifest.get(key):
                        pending.append(manifest[key])
        return buffer.getvalue()
