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

The milestone table records the implemented and checked surface. `context.md` is the
authoritative summary of what exists, and `DECISIONS.md` records every deviation from the
specification.

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
| **C8** | Exploratory correlation, lead–lag, groups, rolling correlation and static PCA | **Verified** |
| **D9** | Model selection, 18 executable methods, state discovery and optimization | **Implemented and tested** |
| **D10** | Diagnostics, interpretation, robustness, chronological validation and approval | **Implemented and tested** |
| **D11** | Approved causal inference, dashboard, monitoring and exports | **Implemented and tested** |
| **E** | Volatility forecasting | **Verified** |

"Verified" means the commands under **Verify** were run and passed *and* the endpoint or page was
exercised against a live server and a real browser — not that every input has been tried.
`tools/verify_all.py` runs the verification commands plus the retained end-to-end drivers.

| Command | Result |
| --- | --- |
| Regime implementation `ruff check` | clean |
| `mypy app` | clean, 67 source files |
| `pytest` | 1416 passed, 8 skipped |
| `npm test` | 67 passed in 12 files |
| `npm run typecheck` | clean |
| `npm run build` | built |

Beyond the unit suite, the drivers under `backend/tools/` exercise the app against
running processes rather than through imports. They are scripts rather than tests because
they need the servers up. See **Verify** for how to run them.

**What is still not verified.** Live provider fetches have been made for Binance klines,
funding, and metrics; the Yahoo paths and their intraday span limits remain unverified,
so only the Binance half of A1 is measured. Treat a claim
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

To enable LightGBM, end-to-end LSTM/GRU training and Optuna optimization, install the
research extra from `backend/`:

```powershell
.venv\Scripts\python -m pip install -e ".[dev,regime-research]"
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
frontend tests, `typecheck` and `build` — then starts the servers and runs the retained
end-to-end drivers, and prints one summary with a single pass/fail. Pass `--no-lint`
(`python dev.py --verify --no-lint`) to skip those checks when only the drivers are in
question.

The drivers exercise the app against running processes rather than through imports;
each is also run by `verify_all.py`. They need both processes up, which the gate starts
for them:

| Driver | What it covers |
| --- | --- |
| `e2e_part_b.py` | Every cleaning policy and its refusals, a written child, both export formats |
| `e2e_part_b_merge.py` | Resample, alignment, join, column ops, a written child |
| `e2e_features.py` | The feature catalogue per dataset, every offered feature, refusals, and the availability rules against real uploads |
| `e2e_splits.py` | The six split procedures and their refusals |
| `e2e_scaling.py` | The scalers: plan binding, preview/write, determinism, and immutability |
| `e2e_exploratory.py` | Synthetic correlation/CCF/groups/rolling, full/train PCA saves, lineage and strict refusal; optional browser pass |
| `e2e_volatility.py` | The walk-forward engine, the metrics, the forward path, and the refusals |
| `e2e_ui.py` | A real browser: every page, the fetch form, the charts, the grid, and each module's run path |

Module D is covered by `tests/test_regimes_api.py`, `tests/test_regimes_workflow.py`,
and `tests/test_regimes_numerics.py`. The separate `tools/verify_regimes_browser.py`
driver runs the rebuilt D9 → D10 → D11 flow on isolated copies of real local BTCUSDT
observations and saves browser evidence. See [Module D verification](docs/module-d.md#installation-and-verification).

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

## Dataset projects

Use the **Project** selector at the top of Data Explorer to reopen a project, or
choose **+ New project** and enter a name. Only the selected project's folder and
datasets appear in the explorer and dataset pickers. Switching projects returns to
Data Management and clears the previous page's dataset selection and unsaved form
state. The selection is remembered for the browser tab.

Uploads, fetched data, derivatives, exports, and job history belong to the selected
project. Work already running continues in the project where it started. The
explorer's **Delete** button removes datasets only from the selected project.

Existing data remains in **Default project**, using the original data directory
without moving files or changing checksums. New projects have their own catalog,
category folders, and outputs under `backend/data/projects/<project_id>/`. Project
names are stored in `backend/data/projects.db`; folder IDs remain stable.

API clients select a project with the `X-Project-Id` header. Browser download URLs
use the `project_id` query parameter. Omitting the selection uses Default project;
an unknown project returns 404. `GET /api/projects` lists projects and
`POST /api/projects` with `{"name": "My project"}` creates one.

## Appearance

The top-right sun button opens the theme selector: **Light**, **Dark**,
**Finance Light**, **Nord**, and **Simple Dark**. Each option shows background,
surface, accent, and text swatches; a checkmark identifies the selected theme.
Selection applies immediately and persists under the existing `theme` localStorage
key. Dark remains the first-visit default; existing `light` and `dark` preferences
and palettes are unchanged.

The original palettes remain in `frontend/src/styles/tokens.css`. The three
additions (`finance-light`, `nord`, `simple-dark`) and semantic/chart extensions
live in `frontend/src/styles/themes.css`. Existing token names are retained.
New semantic aliases distinguish financial direction, validation status, accents,
and focus. Categorical chart colors, regime states, quality markers, volume, and
forecast bands now use those tokens, retaining their original colors in Light/Dark.
Nord uses a Snow Storm tint of Aurora red for small error text to improve contrast.

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

## Module D · Regime Detection

**Implementation and verification recorded: 2026-11-04.** The legacy D implementation has been replaced. The principal flow is **A1 → A2 → B3 → B4 → C5 → C6 → C7 → C8 → D9 → D10 → D11**. Existing A1–C8 functionality and stored research remain in use.

### D9 — Model Selection & State Discovery

Route: `/regime/discovery`. D9 consumes a prepared dataset, an existing C6 split plan and fold, an ordered feature list, and the associated fitted C7/C8 artifacts. Economic diagnostics can use an explicit original-unit market dataset and price, return, ADX, RSI and volatility columns.

| Section | Implemented controls and results |
| --- | --- |
| **D9.1 Baseline Models** | Rule-based classification, moving-average trend, ADX trend, and trend × volatility; existing indicator columns and configurable thresholds |
| **D9.2 Unsupervised Models** | Four HMM variants, K-Means and GMM; separately identified PELT offline segmentation and CUSUM sequential detection |
| **D9.3 Supervised Models** | Logistic Regression, XGBoost, LightGBM, LSTM and GRU; explicit label generation, current classification or future forecasting, horizon and availability lag |
| **D9.4 Hybrid Models** | HMM → XGBoost: earlier teacher fitting, later causal filtered teacher targets, student training, imitation/forecast objectives and chronological agreement evaluation |
| **D9.5 State Configuration & Discovery** | Fixed or automatic counts; candidate ranges; statistical, 2-state, 3-state Bull/Bear/Sideways, 4-state, 6-state trend × volatility and custom templates; minimum occupancy and reproducibility |
| **D9.6 Hyperparameter Optimization** | Manual, Grid Search, Random Search, Bayesian Optimization and Optuna; conditional model-specific candidate spaces and compatible objectives |
| **D9.7 Model Fitting & Training** | Individual fits, staged independent experiments, candidate-count/search jobs, progress/status, failure reasons, execution time and artifact saving |
| **D9.8 Training Results** | Trial and experiment tables, diagnostics, convergence curves, learned parameters, saved configurations, assignments, CSV/JSON/ZIP downloads and handoff to D10 |

The registry contains **18 executable models**:

| Model / registry ID | Model-specific options |
| --- | --- |
| Rule-Based Classification — `rules` | Existing feature, lower and upper thresholds |
| Moving Average Trend — `moving_average` | Prepared fast/slow MA columns and neutral band |
| ADX Trend Classification — `adx` | Prepared ADX, +DI and −DI columns; trend threshold |
| Trend × Volatility — `trend_volatility` | Prepared trend/volatility columns, lower/upper trend thresholds and volatility threshold |
| Gaussian HMM — `gaussian_hmm` | One selected feature; shared HMM settings below |
| Multivariate Gaussian HMM — `multivariate_hmm` | Multiple selected features; shared HMM settings |
| Student-t HMM — `student_t_hmm` | Shared HMM settings and configured fixed degrees of freedom |
| GMM-HMM — `gmm_hmm` | Shared HMM settings and number of Gaussian mixtures per state |
| K-Means — `kmeans` | State/cluster count, K-Means/random initialization, initialization runs, maximum iterations and tolerance |
| Gaussian Mixture — `gmm` | Component count, initialization, initialization runs, covariance type, maximum iterations and tolerance |
| PELT — `pelt` | L2 cost, penalty and minimum segment length; offline segmentation |
| CUSUM — `cusum` | Detection threshold, drift and minimum segment length; sequential change events |
| Logistic Regression — `logistic` | Regularization parameter `C` and maximum iterations |
| XGBoost — `xgboost` | Estimator count, maximum depth and learning rate |
| LightGBM — `lightgbm` | Estimator count, maximum depth and learning rate |
| LSTM — `lstm` | Sequence length, hidden size, layers, epochs, learning rate and batch size; end-to-end CPU PyTorch training |
| GRU — `gru` | Same configurable recurrent-training controls as LSTM |
| HMM → XGBoost — `hmm_xgboost` | Student tree settings, teacher-training fraction, teacher maximum iterations and state count |

Shared HMM settings include state count, emission family selected through the model, full/diagonal/spherical/tied covariance, random/K-Means initialization, transition stickiness, optional initial-state probabilities and transition matrix, tolerance, maximum iterations and experiment seed. Student-t scale matrices are distinguished from return covariance; degrees of freedom remain fixed during EM.

Supervised targets use an existing label column or threshold labels generated from an existing upstream feature. The hybrid uses its HMM teacher. Current-state classification/imitation and future-state forecasting are explicit objectives. Target horizons and availability lags are purged at partition boundaries; sequence models use causal input windows. Teacher agreement measures imitation, not independent economic validity.

State/component counts range from **2 to 12**, where applicable. Count templates do not force discovered IDs into semantic names. Minimum training occupancy rejects insufficiently populated candidates. AIC/BIC apply to likelihood models; silhouette, Davies–Bouldin and Calinski–Harabasz assess compatible cluster geometry; supervised/hybrid optimization uses validation log loss or accuracy. Stability and economic profiles are evaluated separately in D10.

Search spaces map supported parameter names to candidate-value arrays. Random search is seeded. Both Bayesian and Optuna modes use a real seeded Optuna TPE sampler over those values. Grid search is bounded to 100 fits; random/TPE use a trial budget up to 100. The test span never selects hyperparameters, and walk-forward refits repeat the original search policy within each fold.

Indicator periods remain configured in C5. D selects their prepared columns and never duplicates cleaning, missing-value treatment, indicator generation, resampling, splitting, scaling or PCA.

### D10 — Evaluation & Interpretation

Route: `/regime/evaluation`, with optional `?experiment=ID`. All displayed analysis derives from the selected trained candidate and its actual observations.

| Section | Implemented controls and results |
| --- | --- |
| **D10.1 Model Diagnostics** | Model-specific fit/convergence diagnostics, train/validation/test selection, log likelihood where supported, compatible state-separation and predictive scores |
| **D10.2 State Analysis** | Occupancy/counts, mean/median returns, return volatility, skewness, kurtosis, downside deviation, episode maximum drawdown, supplied ADX/RSI characteristics, feature profiles and duration |
| **D10.3 Regime Labeling** | Automatic statistical labels, template-based labels and manual/custom mapping; original model IDs remain unchanged |
| **D10.4 Transition & Persistence** | Empirical transitions, fitted HMM transition matrix where available, expected/empirical durations, switching frequency, state entry/exit events and historical transition timeline |
| **D10.5 Stability & Robustness** | Independent variations of seeds, counts, feature sets, model configurations, training-window fractions and initializations; Hungarian state alignment and permutation-invariant Adjusted Rand Index |
| **D10.6 Walk-Forward Validation** | Saved chronological train/validation/test evaluation, existing C6 expanding/rolling plans, per-fold model refitting and preprocessing verification, out-of-sample prediction and state distribution drift |
| **D10.7 Model Comparison** | Side-by-side statistical fit, separation, economic-profile availability, persistence, stability, out-of-sample predictive metrics, execution time and validation status; researcher approval with rationale |

State analysis charts include original-unit price colored by state, return and upstream-volatility distributions, occupancy, feature heatmaps, state profiles, transition heatmaps, duration distributions and convergence. Missing economic inputs produce explicit unavailable values. Maximum drawdown is calculated within contiguous state episodes; edge durations are marked censored.

Label templates cover trend, volatility, trend × volatility and stress. Resulting labels can include Bullish, Bearish, Sideways, Stress, Low Volatility, High Volatility and combined descriptions. Automatic labels use training statistics; manual labels must cover every original state ID. Approval saves an immutable copy of the mapping.

Walk-forward validation consumes fold-specific prepared datasets when needed. Each scaler/PCA fit must lie inside that fold's training window; unsupported or leaking combinations are refused. Comparison identifies differing input/target contracts and does not rank incompatible candidates as one contest. Approval requires completed causal validation and a rationale, and pins the trained model, feature order, preprocessing, validation and labels for D11.

### D11 — Inference & Current Regime

Route: `/regime/inference`, with optional `?approval=ID`. Deployment uses a D10-approved version whose asset, timeframe and features are fixed by its saved contract.

| Section | Implemented controls and results |
| --- | --- |
| **D11.1 Model Deployment** | Approved model/version selection, compatible observation dataset, verified prepared inputs or replay of exact fitted C7/C8 parameters, optional historical prediction origin |
| **D11.2 Regime Inference** | Raw/confirmed state IDs, genuine model/class probabilities, cluster assignments/distances where supported and structural change events |
| **D11.3 Filtering & Smoothing** | Causal filtering by default; compatible HMM forward-backward smoothing, fixed-lag smoothing and Viterbi explicitly marked retrospective |
| **D11.4 Transition Detection** | State changes, probability threshold where available, minimum persistence, hysteresis, confirmation events and confirmation delay |
| **D11.5 Current Regime Dashboard** | Regime/state, available trend/volatility interpretation, duration, previous state, last transition, confidence, probabilities/distances, model/version and observation/inference timestamps |
| **D11.6 Historical Visualization** | Regime-colored price, historical timeline, probabilities, transition events, duration distributions, regime frequency and transition matrix; paged history |
| **D11.7 Monitoring & Alerts** | Regime changes, low confidence, unusual observations, feature distribution drift, state occupancy drift, optional CUSUM alerts and realized-target performance deterioration where assessable |
| **D11.8 Output & Integration** | Structured JSON, complete assignments CSV, model/dependency ZIP, metadata, explicit state-label mapping, saved inference history and reusable HTTP API |

Inputs are truncated at completed-bar availability **before** feature extraction or fitted preprocessing replay. Filtering uses only information available at each prediction origin. Smoothing/Viterbi outputs retain their retrospective designation; future-state models display forecasts separately from current-regime estimates. Transition confirmations are timestamped when detected and never backdated.

Default inference controls are filtering, minimum persistence **3 bars**, confirmation probability **0.55**, hysteresis **0.10**, low-confidence threshold **0.50**, distribution-drift threshold **0.20**, optional CUSUM alerts disabled, and fixed-lag length **5** when selected. Probability/decoding controls are hidden for incompatible methods. Baselines, K-Means and structural detectors never receive fabricated probabilities.

Monitoring compares recent observations with the training reference. Performance deterioration needs realized labels; absent targets remain unavailable. Trend/volatility dashboard descriptors come from the approved label interpretation and may be unavailable for arbitrary custom labels. Monitoring is computed on each submitted inference run; external streaming and message delivery are not implemented.

### Artifacts, API and application integration

The hierarchical D sidebar exposes all **8 D9**, **7 D10**, and **8 D11** subsections. Forms show model-compatible settings; jobs expose progress/errors; empty states and charts reflect real computation. The workspace uses the existing project selector, themes, PageHeader, shared grids, Plotly and job polling. Subsection links preserve the selected experiment/approval.

Backend implementation lives in `backend/app/services/regimes/`, with HTTP routes in `backend/app/routers/regimes.py`. Frontend pages live in `frontend/src/pages/regimes/`, typed API contracts in `frontend/src/api/regimes.ts`, and sidebar hierarchy in `frontend/src/layout/regimeNavigation.ts`.

Project-scoped immutable records include experiments, searches, labels, validation, robustness, approvals and inference. Bundles retain configuration, original optimization policy, ordered features, training boundaries, data/preprocessing checksums, fitted preprocessing, model bytes, complete assignments, diagnostics and runtime versions. JSON/CSV/ZIP export links use the active project. ZIPs follow deployment dependencies; retain source datasets and the matching runtime separately for reproduction.

| HTTP interface | Purpose |
| --- | --- |
| `GET /api/regime/options`, `GET /api/regime/plans/{id}` | Model capabilities, prepared features/artifacts and existing C6 spans |
| `POST /api/regime/experiments` | Fit/search job |
| `GET /api/regime/experiments/{id}/evaluation` | Computed diagnostics and interpretation |
| `POST /api/regime/experiments/{id}/labels`, `/validation`, `/robustness`, `/approval` | Labeling, validation/refits and approval |
| `POST /api/regime/comparison`, `POST /api/regime/inference` | Candidate comparison and approved inference |
| `GET /api/regime/artifacts`, `/artifacts/{id}` | Project-scoped saved results |
| `GET /api/regime/artifacts/{id}/history`, `/export?format=json\|csv`, `/bundle` | Paged history, full JSON/CSV and dependency ZIP |

Long-running operations use the existing HTTP 202 job/poll interface. External API callers use `X-Project-Id`; browser downloads use `project_id`. Dependency installation is covered under [Setup](#setup): `.[dev,regime-research]` enables LightGBM, PyTorch and Optuna. Unavailable dependencies produce explicit registry/execution messages.

### Verified implementation and remaining limits

| Check | Recorded result |
| --- | --- |
| Full backend suite, including upstream regression coverage | **1,416 passed; 8 optional tests skipped** |
| Frontend suite | **67 passed across 12 files** |
| Backend type check | **67 source files; no issues** |
| Frontend type check / production build | **Passed** |
| Ruff checks for regime code, tests and updated tools | **Passed** |
| Headless Edge D9 → D10 → D11 | **Passed on 2,131 real BTCUSDT daily observations; no browser errors** |

Numerical/integration coverage includes all 18 registry methods, HMM reference probabilities, PELT optimality, actual recurrent training, supervised horizon purging, state-count/optimizer execution, fold-local walk-forward selection, C7/C8 replay, future-observation perturbation, project isolation, artifact-integrity refusal and approval/inference. Browser checks cover the BTCUSDT Gaussian-HMM workflow, original-price charts, model bundles, subsection links, light/dark themes and compact layout. They do not establish economic validity or test every asset/model/parameter combination.

- Complex C8 pre-transforms and rolling C7 chains need verified prepared outputs where no upstream fitted replay interface exists. Fold preprocessing must be fitted within that fold's training window.
- PELT supports offline L2 segmentation and at most 20,000 observations; it cannot be approved as a live state classifier. CUSUM segment IDs represent sequential events.
- Student-t degrees of freedom are configured and fixed; Bayesian/Optuna modes use TPE over discrete candidates; LSTM/GRU training is CPU-based.
- D11 currently requires complete post-training history. Incremental checkpoints, external streaming schedules and alert delivery remain future extensions.
- Crypto/equity datasets share the existing contracts; live asset/timeframe/feature changes require a separate trained and approved version. Missing economic/realized-target inputs remain unavailable.
- Optional end-to-end tests remain skipped in the ordinary suite, and existing large frontend bundle warnings remain. Graphify could not update because its executable is unavailable.

See the [saved browser evidence](artifacts/regime-rebuild/verification.json), [detailed Module D reference](docs/module-d.md), [architecture diagram](docs/architecture.md), [decision entry](DECISIONS.md#decision-18--adopt-the-rebuilt-d9d11-regime-workflow-and-document-verified-capabilities), and [current implementation context](context.md).
