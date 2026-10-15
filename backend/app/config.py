"""Application settings.

Paths are resolved once at import time and validated eagerly: a research tool that
silently writes datasets to an unexpected directory is worse than one that refuses
to start.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/
BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QML_",
        env_file=str(BACKEND_ROOT / ".env"),
        extra="ignore",
    )

    data_dir: Path = Field(default=BACKEND_ROOT / "data")

    # Upstream providers. No API key is required for the public endpoints used here.
    binance_spot_base: str = "https://api.binance.com"
    binance_futures_base: str = "https://fapi.binance.com"
    binance_metrics_base: str = "https://fapi.binance.com"

    # Vite dev server origin; the only cross-origin caller permitted.
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    # Network etiquette: public endpoints are rate limited and we are a guest on them.
    http_timeout_seconds: float = 30.0
    http_max_retries: int = 3

    # Server-side paging ceiling for the data grid. The browser must never request
    # an unbounded window from a multi-million row Parquet file.
    max_page_size: int = 500
    default_page_size: int = 100

    @field_validator("data_dir")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return v if v.is_absolute() else (BACKEND_ROOT / v).resolve()

    #: Where Part B–E outputs (reports, exports) will land. The one storage directory
    #: that is not a `DatasetCategory` name *and* is not created on demand.
    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.db"

    def ensure_dirs(self) -> None:
        """Create the directories that must exist before anything writes.

        Only `outputs` is listed. Dataset storage lives under the `DatasetCategory`
        value (`raw_data/`, `uploaded_data/`, `processed_data/`), and
        `parquet_store.write_dataset` creates that directory itself, so pre-creating it
        here would mean two places deciding the layout — and they did disagree: this
        method used to create `raw/`, `uploaded/` and `processed/`, three directories
        nothing ever wrote to, sitting next to the three that were actually used. That
        is worse than untidy, because the empty directory is the first place someone
        debugging a missing dataset looks.
        """
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
