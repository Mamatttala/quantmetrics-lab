# Quantmetrics Lab

A local, single-user quantitative-finance research workstation. FastAPI + pandas +
DuckDB + Parquet on the back; React + TypeScript on the front. No authentication, no
cloud dependency, no data leaves the machine.

The design priorities, in order, are: quantitative correctness, reproducibility,
prevention of look-ahead and data leakage, transparent calculations, modular
architecture, maintainability, clear UX, reliable testing. Where a requested feature
would conflict with those, the conflict is recorded in [`DECISIONS.md`](./DECISIONS.md)
rather than resolved quietly.

---

## Status

Every part supplied so far is built and verified. `context.md` is the authoritative summary of
what exists, `DECISIONS.md` the record of every deviation from the specification; the table
below is a one-line index into them.

| Milestone | Scope | State |
| --- | --- | --- |
| **M0** | Repository, scaffold, design tokens, three-pane shell, sidebar registry | **Verified** |
| **M1** | Storage layer, catalog, Data Management page, editable grid | **Verified** |
| **M2** | A1 · Raw Data Fetch (Binance + Yahoo) | **Verified** |
| **M3** | A2 · Raw Data View (candlestick + quality overlays) | **Verified** |
| **B3** | Cleaning: quality report, duplicate/missing/outlier policies, lineage | **Verified** |
| **B4** | Resample, align, merge; column rename/drop; 5+5 preview; export | **Verified** |
| **C5** | Feature engineering: catalogue, causal engine, report, page | **Verified** |
| **C6** | Chronological splits | **Verified** |
| **C7** | Scaling | **Verified** |
| **D** | Regime detection | **Verified** |
| **E** | Volatility forecasting | **Verified** |

"Verified" means the commands under **Verify** were run and passed *and* the endpoint or page was
exercised against a live server and a real browser — not that every input has been tried.
`tools/verify_all.py` runs all five commands below plus the eight end-to-end drivers.

| Command | Result |
| --- | --- |
| `ruff check .` | clean |
| `mypy app` | clean, 47 source files |
| `pytest` | 1284 passed, 8 skipped |
| `npm run typecheck` | clean |
| `npm run build` | built |

Beyond the unit suite, the drivers under `backend/tools/` exercise the app against
running processes rather than through imports: seven HTTP drivers replay exactly what
the browser sends, and `e2e_ui.py` clicks the real UI in a real engine. They are scripts
rather than tests because they need the servers up. See **Verify** for how to run them.

**What is still not verified.** Live provider fetches have been made for Binance klines,
funding, and metrics; the Yahoo paths and the intraday span limit recorded in
`DECISIONS.md` are still unrun, so only the Binance half of A1 is measured. Treat a claim
as measured only where the rows above say so.

Routes registered but unbuilt render an explicit *not implemented* bar rather than a
mock, so an absent feature cannot be mistaken for a broken one.

---

## Requirements

- Python 3.11+ (developed against 3.14.3)
- Node 20+ (developed against 26.7.0)
- Windows — the process management in `backend/tools/serve.py` uses `taskkill` and
  PowerShell, the platform this was built and verified on.

## Setup

One command, from the repository root:

```powershell
python setup.py
```

It creates `backend/.venv`, installs the backend package and its dev extras, and installs
the frontend dependencies. It calls the virtualenv's interpreter **by path**, so it never
needs the venv activated — safe under PowerShell's default script-execution policy. It is
idempotent; re-run it after a dependency change.

To do it by hand instead (no activation step is required anywhere in this project):

```powershell
cd backend
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
cd ..\frontend
npm install
```

## Run

One command, from the repository root:

```powershell
python dev.py
```

It starts the API and the web app, and waits until **both** answer before returning,
printing `api ready` and `web ready` — that is the signal the install is working. Then
open <http://localhost:5173>. It restarts anything already running, so a stale server from
a previous session can never serve the old code. Stop both with `python dev.py --stop`.

Vite proxies `/api` to port 8000, so the browser stays same-origin and CORS never becomes
load-bearing. It writes `backend/uvicorn.log` and `frontend/vite.log` (both gitignored) so
a start-up crash is readable after the fact rather than scrolling past in a terminal.

By hand, instead — one process per terminal:

```powershell
cd backend
uvicorn app.main:app --reload --port 8000
```

```powershell
cd frontend
npm run dev
```

## Verify

The fastest check that an install works is just to run it — `python dev.py` printing
`api ready` and `web ready`, and <http://localhost:5173> loading.

The full gate, one command from the repository root:

```powershell
python dev.py --verify
```

It runs every check in dependency order — `ruff`, `mypy`, the unit suite (`pytest`), the
frontend `typecheck` and `build` — then starts the servers and runs **all eight**
end-to-end drivers, and prints one summary with a single pass/fail. Pass `--no-lint`
(`python dev.py --verify --no-lint`) to skip the first five when only the drivers are in
question.

The eight drivers exercise the app against running processes rather than through imports;
each is also run by `verify_all.py`. They need both processes up, which the gate starts
for them:

| Driver | What it covers |
| --- | --- |
| `e2e_part_b.py` | Every cleaning policy and its refusals, a written child, both export formats |
| `e2e_part_b_merge.py` | Resample, alignment, join, column ops, a written child |
| `e2e_features.py` | The feature catalogue per dataset, every offered feature, refusals, and the availability rules against real uploads |
| `e2e_regime.py` | The regime detectors: options, a run, the report, and the output frame |
| `e2e_splits.py` | The six split procedures and their refusals |
| `e2e_scaling.py` | The scalers: plan binding, preview/write, determinism, and immutability |
| `e2e_volatility.py` | The walk-forward engine, the metrics, the forward path, and the refusals |
| `e2e_ui.py` | A real browser: every page, the fetch form, the charts, the grid, and each module's run path |

The drivers are additionally reachable from `pytest`, which is useful when only one is in
question or when the result belongs in the test report:

```powershell
$env:QM_E2E = "1"; python -m pytest -m e2e -q
```

They are skipped without `QM_E2E=1`, so a plain `pytest` runs only the unit suite. The
fixture starts both processes through `tools\serve.py` first, so the drivers can never be
pointed at a server still running the previous revision of the code. `e2e_ui.py` drives
Microsoft Edge over the DevTools Protocol — no new dependency, since Edge is installed and
`websockets` ships with `uvicorn[standard]` — and fails on a dead button or an uncaught
render error, the class of defect a type-checker and an HTTP driver both miss.

### What the tests are for

Beyond ordinary coverage, the suite enforces the invariants this project is built
around:

- **Immutability** — writing to an existing raw or uploaded path raises; deriving
  leaves the parent's checksum unchanged.
- **Lineage** — derived datasets point at a real parent; no cycles.
- **Leakage** — `close_time` is never used where an observation time is meant.
- **Point-in-time** — every Part C transformation is causal: recomputing a feature series
  over a truncated history leaves every row it shares unchanged. This is asserted for
  every catalogue feature, so a whole-sample statistic added later fails a test rather
  than a backtest.
- **Async rule** — no pandas/DuckDB/PyArrow call outside `app/core/blocking.py`.
- **Paging** — `offset`/`limit` windows are exact, non-overlapping, and sum to
  `row_count`.
- **Determinism** — identical fetch parameters produce an identical checksum.

---

## Layout

```
backend/
  app/
    core/       blocking.py is the only sanctioned way to call pandas/DuckDB
    models/     enums, dataset metadata, API DTOs
    storage/    catalog.py (SQLite), parquet_store.py, duckdb_engine.py
    services/   provider fetchers, ingest, dataset operations
    routers/    HTTP layer
  data/         runtime state, gitignored: catalog.db, outputs/, and one directory per
                dataset category (raw_data/ uploaded_data/ processed_data/), each
                created on demand by the first write to it
frontend/
  src/
    api/        typed client mirroring the backend DTOs
    layout/     Shell, Sidebar, DataExplorer, nav registry
    components/ shared UI
    charts/     lightweight-charts and Plotly wrappers
    pages/      routed content


📁 Project Structure

quantmetrics-lab/
├── backend/
│   ├── app/
│   │   ├── core/           # Core utilities (blocking.py, errors.py)
│   │   ├── models/         # Pydantic models, enums, DTOs
│   │   ├── storage/        # DuckDB, Parquet, catalog
│   │   ├── services/       # Business logic (fetchers, processors)
│   │   └── routers/        # FastAPI endpoints
│   ├── data/               # Runtime data (gitignored)
│   │   ├── catalog.db      # SQLite metadata
│   │   ├── raw_data/       # Fetched data
│   │   ├── uploaded_data/  # User uploads
│   │   ├── processed_data/ # Cleaned/transformed data
│   │   └── outputs/        # Model outputs
│   ├── tests/              # Unit tests
│   ├── tools/              # E2E drivers, serve.py
│   └── pyproject.toml      # Dependencies
│
├── frontend/
│   ├── src/
│   │   ├── api/            # Typed API client
│   │   ├── layout/         # Shell, sidebar, navigation
│   │   ├── components/     # Reusable UI components
│   │   ├── charts/         # Plotly + lightweight-charts wrappers
│   │   ├── pages/          # Route pages
│   │   └── styles/         # CSS tokens
│   └── package.json        # Dependencies
│
├── context.md              # Project context (authoritative)
└── DECISIONS.md            # Deviations from spec`


### Conventions

- **Parquet is canonical.** CSV is import/export only.
- **Datasets are immutable.** Edits fork a derived dataset with a
  `parent_dataset_id`; only `processed_data` is overwritten in place.
- **`dataset_id` is a ULID**, and storage paths derive from it — renaming never
  moves a file.
- **pandas, DuckDB, PyArrow, and yfinance are blocking.** Every call from the service
  layer goes through `app/core/blocking.py`.
- **The backend never renders images for the main UI.** It returns numbers; the
  frontend draws them.
