# Quantmetrics Lab — current context

**Last synchronized: 2026-11-04, following the D9–D11 rebuild and implementation verification.** The former D specification has been replaced. The current scope follows the user's master implementation plan and `docs/mermaid-diagram.png`.

Quantmetrics Lab is a local quantitative research application for crypto and equity datasets, multiple assets, timeframes and configured feature sets. Datasets, jobs and artifacts are project-scoped. The implementation preserves working A1–C8 workflows and stored user research. Old D records cannot serve as approvals for the replacement module.

## Application flow

| Step | Current responsibility |
| --- | --- |
| A1 — Raw Data Fetch | Existing market-data collection |
| A2 — Raw Data View | Existing source-data inspection |
| B3 — Cleaning | Existing quality checks and missing/duplicate/outlier policies |
| B4 — Resampling/Merging | Existing timeframe alignment and dataset merging |
| C5 — Feature Selection | Existing feature engineering and indicator configuration |
| C6 — Chronological Splits | Existing immutable train/validation/test and walk-forward definitions |
| C7 — Scaling | Existing fitted transformation/scaling artifacts |
| C8 — Exploratory Analysis | Existing correlation/PCA analysis and fitted PCA artifacts |
| D9 — Model Selection & State Discovery | Model configuration, experiments, state counts, optimization and training |
| D10 — Evaluation & Interpretation | Diagnostics, state analysis, labeling, persistence, robustness, validation, comparison and approval |
| D11 — Inference & Current Regime | Approved-model inference, current/historical views, transitions, monitoring and outputs |

**A1 → A2 → B3 → B4 → C5 → C6 → C7 → C8 → D9 → D10 → D11.** The existing E volatility application remains available. See [the architecture diagram](docs/architecture.md).

## Implementation and ownership

Frontend infrastructure remains React 19, TypeScript, Vite, React Router, existing theme CSS, ag-Grid, lightweight-charts and Plotly. The rebuilt pages reuse PageHeader, shared result grids, plot wrappers, project selection and asynchronous job polling. Sidebar subsections preserve selected experiment/approval query parameters. Forms display model-compatible fields; loading, failures, unavailable measurements and empty results are explicit.

Backend infrastructure remains Python/FastAPI/Pydantic, NumPy/SciPy/scikit-learn and XGBoost, with optional LightGBM, CPU PyTorch and Optuna research dependencies. Storage remains the project-scoped SQLite catalog plus immutable Parquet and checksummed output bundles.

| Component | Current files | Responsibility |
| --- | --- | --- |
| D page routing | `frontend/src/App.tsx`, `frontend/src/layout/registry.ts` | D9 discovery, D10 evaluation and D11 inference routes |
| Sidebar hierarchy | `frontend/src/layout/regimeNavigation.ts`, `Sidebar.tsx` | 8 D9, 7 D10 and 8 D11 subsection links |
| D frontend | `frontend/src/pages/regimes/Discovery.tsx`, `Evaluation.tsx`, `Inference.tsx` | Research, evaluation and inference workspaces |
| Shared D presentation | `frontend/src/pages/regimes/shared.tsx`, `charts.tsx`, `regimes.css` | Conditional fields, grids, plots, status and theme styling |
| Typed client | `frontend/src/api/regimes.ts` | Requests, result types, artifact/download URLs |
| HTTP and jobs | `backend/app/routers/regimes.py`, `backend/app/services/regimes/service.py` | `/api/regime` routes and existing job service integration |
| Data and contracts | `backend/app/services/regimes/data.py`, `contracts.py` | Upstream contracts, provenance, fitted replay and strict request validation |
| Registry and estimators | `registry.py`, `estimators.py`, `hmm.py`, `structural.py`, `neural.py` under the regime service package | Capability/options registry and actual numerical model implementations |
| Experiments and persistence | `experiments.py`, `artifacts.py` | Fits, searches, reproducibility and immutable bundles |
| D10 | `metrics.py`, `evaluation.py` | Statistical/economic analysis, labels, refits, comparison and approval |
| D11 | `inference.py` | Cutoff-safe prediction, confirmation, monitoring and historical outputs |

The former D setup/fitting/evaluation executors and dedicated pages/tests were removed. The shared browser driver now targets the new routes. Data access, features, models, experiments, evaluation, interpretation, inference, visualization and export remain separate responsibilities inside the existing architecture.

## D9–D11 sections, models and options

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

## Data contracts and chronological invariants

1. **Prepared inputs.** `Inputs` identifies `dataset_id`, `split_plan_id`, fold and 1–100 unique ordered existing features. Optional `market_dataset_id`, price/return/ADX/RSI/volatility columns provide original-unit diagnostics. Default price column is `close`. Missing/infinite feature values require upstream preparation.
2. **Existing C6 geometry.** Observation times, split row counts, source lineage and checksums must agree. D does not create a second split workflow. Training, validation and test have distinct roles.
3. **Preprocessing identity.** Saved C7/C8 fit bounds must be contained within the model's training window. Signature/checksum validation binds feature order and fitted parameters. D replay applies supported saved values without refitting.
4. **Experiment identity.** Schema version 1 records model, explicit parameters, state configuration, target/search settings, ordered inputs and seed. Default seed is 42; default statistical count is 3, candidate counts are 2/3/4, minimum occupancy is 0.01 and selection criterion is BIC where compatible. Unsupported parameter/criterion combinations are refused.
5. **Targets.** Source is column, rules or hybrid teacher. Objective is current or future; future horizon is 1–1,000 bars, availability lag 0–1,000. Boundary purging prevents unavailable labels entering model fitting or scoring. Future-shifted inputs are not generated by D.
6. **Search scope.** Manual fitting is default. Search spaces contain explicit parameter candidates and have bounded execution. The original `research_config` is saved so each walk-forward fold repeats the research policy on that fold's training/validation information.
7. **Teacher/student chronology.** HMM fitting precedes the student-target window; targets use filtered teacher states. Teacher agreement and economic validity are separate measurements.
8. **Validation/refits.** Chronological validation uses the saved fit. Expanding/rolling validation requires an existing C6 plan and, when necessary, `prepared_datasets` mapped by fold. Shorter-window robustness also enforces preprocessing fit bounds.
9. **Labels and approval.** Statistical state IDs never become forced economic labels. Manual mapping covers every original ID. Approval requires a validation artifact ID and rationale; the approved snapshot pins model version, contract, objective, fitted preprocessing and labels.
10. **Available observations.** D11 uses completed-bar availability before extraction/replay. Input asset, timeframe, feature identity, fitted preprocessing and full post-training history must agree. Changed or missing saved observations are refused.
11. **Prediction meaning.** Probabilities are native model outputs. K-Means distances and structural events are not probabilities. Filtering is causal; smoothing, fixed-lag smoothing and Viterbi remain retrospective. Future forecasts are distinguished from current-state estimates.
12. **Events and diagnostics.** Transition confirmation is timestamped at detection. Delay measures confirmation lag, not an independently observed economic regime-change time. Economic statistics use original units; absent evidence remains unavailable.
13. **Project isolation.** Requests use the active project context. `X-Project-Id` selects a project for HTTP clients; browser exports use `project_id`. No client-selected model path or arbitrary uploaded model deserialization is accepted.

## Artifacts, output and API

The project output directory contains `outputs/regimes/{ULID}/`. A manifest is anchored by a catalog checksum. An experiment stores `manifest.json`, `model.bin` and `assignments.parquet`; related immutable records store searches, labels, validation, robustness, approvals and inference.

Model manifests retain the fit configuration, original research/search policy, fitted model parameters, ordered feature contract, train bounds, source/preprocessing checksums, embedded fitted C7/C8 parameters, diagnostics, seed and runtime versions. Saved assignments retain observation times, raw states, partition/scope and probabilities/distances/events where applicable. Inference adds availability time, confirmed states, labels, prices where available, the request, model/approval identifiers and monitoring.

JSON exports contain metadata and complete history; CSV exports contain assignments. ZIP exports include the model and referenced approval/validation/experiment dependencies with embedded preprocessing. They do not include the whole source dataset or application code; preserve those plus the matching runtime for reproduction. There is no model-import endpoint for uploaded pickle files.

| API relative to `/api/regime` | Contract |
| --- | --- |
| `GET /options?dataset_id=…` | Registry availability, model parameters/metrics, numeric inputs, plans and preprocessing |
| `GET /plans/{id}` | Existing C6 spans |
| `POST /experiments` | Strict `Experiment`; fit/search job |
| `GET /artifacts?kind=…` | experiment/search/labels/validation/robustness/approval/inference records |
| `GET /artifacts/{id}` | Verified manifest |
| `GET /artifacts/{id}/history` | Offset/limit paging, default 2,500 and maximum 5,000 rows |
| `GET /artifacts/{id}/export?format=json\|csv` | Full result export |
| `GET /artifacts/{id}/bundle` | Dependency ZIP |
| `GET /experiments/{id}/evaluation` | Computed evaluation |
| `POST /experiments/{id}/labels` | Label mode, template and explicit mapping |
| `POST /experiments/{id}/validation` | Chronological/expanding/rolling mode, optional C6 plan and per-fold prepared dataset IDs |
| `POST /experiments/{id}/robustness` | Seeds, counts, feature sets, configurations, training fractions and initializations |
| `POST /experiments/{id}/approval` | Validation ID and researcher rationale |
| `POST /comparison` | Explicit experiment IDs |
| `POST /inference` | Approval, observations, prepared/replay mode, historical cutoff, decoding and confirmation/monitoring settings |

Long-running operations return HTTP 202 with the existing job ID/poll contract. Numerical work reports execution status and failure reasons. OpenAPI is served at `/docs`. Python integration uses `RegimeService` inside the existing project context; detailed examples are in [docs/module-d.md](docs/module-d.md).

## Dependencies and verification

From `backend/`, install `.[dev,regime-research]` to enable LightGBM, PyTorch and Optuna. XGBoost is a base backend dependency. The registry identifies missing optional dependencies and refuses unavailable execution.

Recorded implementation verification on **2026-11-04**:

| Verification | Result and scope |
| --- | --- |
| Full backend pytest suite | **1,416 passed, 8 optional skips**, including existing upstream regression tests |
| Frontend Vitest suite | **67 passed across 12 files** |
| Backend mypy | **67 source files; no issues** |
| Frontend TypeScript / production build | **Passed** |
| Ruff on regime implementation, tests and updated verification tools | **Passed** |
| Real headless Edge workflow | **D9 fit → D10 validate/approve → D11 filter passed**, on **2,131 BTCUSDT daily observations**, no browser errors |

The new computational/integration tests are `backend/tests/test_regimes_numerics.py`, `test_regimes_workflow.py` and `test_regimes_api.py`; frontend workflow tests are in `frontend/src/pages/regimes/Regimes.test.tsx`. Coverage includes all 18 registry models, HMM numerical ground truth and causal prefixes, PELT optimality, actual recurrent training, supervised horizon/availability purging, automatic state counts and search backends, fold-local walk-forward selection, C7 and C8 replay, future-observation perturbation, original-price diagnostics, project isolation, artifact integrity, approvals and inference.

`backend/tools/verify_regimes_browser.py` copies existing real BTCUSDT daily observations into `.tmp/regime-review`, prepares them through existing C6/C7 services, drives the built UI on its own temporary server, and stops that server afterward. It verifies bundle export and subsection selection as well as the workflow, original price units, light/dark themes and compact layout. The saved [verification report](artifacts/regime-rebuild/verification.json) and [screenshots](docs/module-d.md#installation-and-verification) document that run. The reference model is execution evidence, not economic validation.

Run pytest from `backend/` to use its async-test configuration; frontend commands are `npm test`, `npm run typecheck` and `npm run build`. These counts describe the recorded implementation tests; this documentation-only synchronization does not rerun them.

## Known limitations and remaining work

- Rolling C7 chains and complex C8 pre-transforms lack a general fitted replay interface. Supply verified prepared outputs from the same fitted artifacts; D refuses unsupported replay rather than duplicating preprocessing. C6/C7/C8 lineage and fit bounds must remain valid for every fold.
- PELT is offline L2 segmentation, capped at 20,000 observations, and is ineligible as a live state classifier. CUSUM segments are sequential change events; they are not recurring hidden economic states.
- Student-t degrees of freedom are fixed; Bayesian/Optuna search is discrete TPE; recurrent fitting is CPU-based. Other emission/optimizer/accelerator combinations are not represented as implemented.
- D11 replays complete post-training history. Incremental continuation checkpoints, external scheduling, market-data subscriptions and alert delivery remain future work.
- Automatic economic interpretation needs original-unit returns and related evidence. Missing ADX/RSI/volatility or realized labels remain unavailable. Custom labels may not provide trend/volatility descriptors.
- Real-browser coverage is the BTCUSDT Gaussian-HMM workflow. Other models have computational/integration tests; all real crypto/equity assets, timeframes and configurations have not been exercised. Repeated researcher selection on test reports does not preserve an untouched economic holdout.
- Eight optional existing end-to-end tests remain skipped in the ordinary suite. Existing large frontend bundle warnings remain. The parent-requested Graphify update was attempted but its CLI is unavailable.
- Preserve user datasets and unrelated working-tree changes. Historical D research is retained, but only new D10 approvals can be deployed through the replacement D11.

See [Decision 18](DECISIONS.md#decision-18--adopt-the-rebuilt-d9d11-regime-workflow-and-document-verified-capabilities), the [README capability inventory](README.md#module-d--regime-detection), the [technical operating reference](docs/module-d.md), and the [architecture diagram](docs/architecture.md).
