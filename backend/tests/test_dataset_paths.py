"""Recorded storage locations survive a WSL-to-Windows application launch."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import Settings
from app.services.datasets import resolved_path
from tests.conftest import hourly_frame, register


@pytest.mark.skipif(os.name != "nt", reason="Native Windows drive resolution")
def test_wsl_recorded_path_reads_the_same_file_without_changing_metadata(settings, catalog):
    meta = register(settings, catalog, hourly_frame(12), name="historical")
    native = resolved_path(meta, settings).resolve()
    recorded = f"/mnt/{native.drive[0].lower()}/{native.as_posix().split(':/', 1)[1]}"
    historical = meta.model_copy(update={"storage_path": recorded})

    assert resolved_path(historical, settings) == native
    assert resolved_path(historical, settings).read_bytes() == native.read_bytes()
    assert historical.storage_path == recorded


@pytest.mark.parametrize(
    "storage_path", ["old-layout/retained.parquet", "/other-mount/missing.parquet"]
)
def test_recorded_suffix_is_preserved_without_id_or_category_fallback(
    settings: Settings, catalog, storage_path: str
):
    meta = register(settings, catalog, hourly_frame(12), name="historical")
    historical = meta.model_copy(update={"storage_path": storage_path})
    expected = Path(storage_path)
    if not expected.is_absolute():
        expected = settings.data_dir / expected

    assert resolved_path(historical, settings) == expected
    assert not resolved_path(historical, settings).exists()

