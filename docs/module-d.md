# Module D — Regime Detection

Implemented on 2026-11-04 from the D9–D11 master plan and [reference diagram](mermaid-diagram.png). The previous D implementation has been removed. Existing user datasets remain intact; historical D artifacts are not new-model approvals.

See the [implementation plan](regime-implementation-plan.md), [architecture diagram](architecture.md), and [current decisions](../DECISIONS.md).

**Remediation status, 2026-11-05:** F1–F13 are fixed and regression-tested; Phase 4
verification is complete. See the [final report](module-d-remediation-report.md)
for the authoritative current assessment: backend 1,535 passed / 8 opt-in skips,
frontend 76 passed, ten representative real-data family checks passed, and enabled
optional E2E 7 passed / 1 failed. The remaining E2E failure is unsupported legacy
D→E volatility conditioning; two pre-existing shared-file Ruff findings also remain.
The dated phase sections below describe their respective checkpoints.

## Operating the workflow

1. Prepare observations and features through A1–C5. Create chronological spans in C6; fit any C7 scaling or C8 PCA on training observations only.
2. Open **D9 — Model Selection & State Discovery** at `/regime/discovery`. Select a prepared dataset, compatible C6 plan/fold, and ordered feature columns. Review associated preprocessing. Supply original-unit return, price, ADX, RSI and volatility columns when available for economic diagnostics.
3. Select a model and configure its conditional settings, target where required, state count and optional optimization. Fit immediately or stage independent experiments. Results include actual diagnostics, assignments, fitted parameters, configurations, elapsed time and downloadable bundles.
4. Open the candidate in **D10 — Evaluation & Interpretation** at `/regime/evaluation`. Review train/validation/test metrics, profiles, transitions, economic labels, robustness and comparisons. Run chronological validation or select an existing C6 expanding/rolling plan with compatible prepared input per fold.
5. Save a state-ID-to-label mapping. Enter an approval rationale and approve the validated version. Approval pins the model and preprocessing contract, validation evidence and label mapping.
6. Open **D11 — Inference & Current Regime** at `/regime/inference`. Select an approved version, compatible observations and inference settings. Prepared mode verifies the fitted preprocessing signature; replay mode applies the saved supported C7/C8 parameters. Filtering is the default. Run inference to populate the dashboard, historical charts, monitoring and exports.

The asset, timeframe and feature order come from the approved contract. Use a separate experiment for another asset or timeframe. A pooled cross-asset estimator is not implicit.

## Model coverage

All 18 registry entries have executable implementations and computational tests. Availability is checked against installed dependencies.

| Family | Implemented methods | Configuration and output |
| --- | --- | --- |
| Baselines | Rule-based, moving-average trend, ADX trend, trend × volatility | Prepared indicator columns, configurable thresholds/rules, deterministic state IDs |
| HMM | Gaussian, multivariate Gaussian, Student-t, GMM-HMM | State count, initialization, covariance, start/transition probabilities, tolerance, iterations and seed; native filtered probabilities |
| Clustering | K-Means, Gaussian mixture | Counts, initialization, iterations and seed; distances for K-Means, native posterior component probabilities for GMM |
| Structural | PELT, CUSUM | Offline L2 segmentation with penalty/minimum segment length; sequential threshold/drift detection against a training-fitted reference |
| Supervised | Logistic Regression, XGBoost, LightGBM | Explicit label source, current/future objective, availability lag, prediction horizon and estimator parameters |
| Recurrent | LSTM, GRU | End-to-end CPU PyTorch training, sequence length, hidden size, layers, learning rate, epochs and batch size |
| Hybrid | HMM → XGBoost | Teacher fitting on an earlier subwindow; student targets from later causal teacher filtering; current imitation and future forecasting |

Univariate Gaussian HMM requires one selected feature. Multivariate Gaussian accepts multiple columns. Student-t and GMM emissions have their own supported covariance configurations; unavailable combinations are refused. Student-t degrees of freedom are configured and held fixed during EM.

Indicator periods are configured in existing C5 features, then selected in D. D never regenerates MA, ADX, RSI or volatility inputs. No probability is invented for deterministic baselines, K-Means or change-point detectors.

## Experiments and optimization

Fixed and automatic counts support 2–12 states/components. The 2-, 3-, 4- and 6-state templates choose counts without imposing semantic names on statistical states. Every successful candidate is saved independently. Candidates that violate minimum occupancy fail with a recorded reason and cannot be selected.

Manual, grid, seeded random, Bayesian and Optuna modes execute actual fits. Both Bayesian and Optuna use Optuna's seeded TPE sampler over model-specific candidate values. The user supplies a JSON search space, for example `{"max_iter": [50, 100], "covariance_type": ["diag", "full"]}` where the registry supports those parameters. Trial limits are explicit.

AIC/BIC are restricted to likelihood models; separation scores require valid cluster geometry. Supervised searches use validation accuracy or log loss. The test partition never selects hyperparameters. Walk-forward folds repeat the original research/search configuration using only each fold's training/validation span. State occupancy and economic profiles remain visible; stability is evaluated with independent refits and permutation-invariant ARI plus Hungarian state alignment.

## Evaluation and interpretation

D10 computes model diagnostics; counts/occupancy; mean/median returns; volatility, skewness, kurtosis, downside deviation and episode drawdown; supplied ADX/RSI characteristics; feature profiles; and durations. Missing economic inputs are explicitly unavailable.

Plots include state-colored original-unit prices, returns/volatility distributions, feature heatmaps, occupancy, transition matrices, durations and convergence. Diagnostic charts may be sampled for rendering; complete assignments remain exportable. Drawdown is measured within contiguous state episodes; boundary durations are marked censored.

Automatic/template labels use training-state statistics. Manual/custom labels preserve raw IDs in a separate mapping. Templates support trend, volatility, trend × volatility and stress. Teacher agreement is an imitation score, not evidence of independent economic validity.

Chronological validation evaluates frozen candidates. Expanding/rolling validation consumes existing C6 plans, refits the model per fold and verifies each fitted preprocessing span. Robustness varies seeds, features, counts, parameters, initializations and training windows; incompatible variants return their reason. Comparison identifies incompatible feature/target/partition contracts rather than ranking them together.

## Causal inference and monitoring

D11 cuts inputs at their completed-bar availability time **before** feature extraction or preprocessing replay. It verifies asset/timeframe, feature order, artifact checksums, preprocessing identity and the complete post-training history. Changed or omitted saved observations are refused.

F6 historical revision policy: overlapping **selected model features** must match the
experiment and earlier saved inference runs for that experiment, including accepted
appends. Comparison happens in the approved prepared coordinates with relative and
absolute tolerance `1e-12` for fitted replay roundoff. Changed inputs or omitted saved
post-training timestamps are refused; there is no revision override. Revised research
inputs require a separate experiment and approval, leaving previous bundles intact.
Optional realized labels and other non-model columns are not frozen by this check:
their absence must not block predictions. Saved inference source datasets must remain
available and pass checksum verification. Historical origins compare only the history
available to that request; later inputs are not inspected for feature agreement.

F7 saved inference context: opening a saved run restores its approval/version,
dataset, preprocessing mode, historical origin, decoder/lag, confirmation settings
and monitoring thresholds. The API example and retrospective result description
use the saved request even while controls are edited for a new run. Selecting
another approval clears the displayed result. Legacy records without request
metadata report that the saved API request is unavailable.

F12 transition confirmation requires consecutive eligible assigned observations.
An unassigned/invalid negative state resets the pending candidate and its counter;
the last confirmed state remains displayed (or `-1` before initial assignment).
For `[0, 1, -1, 1]` and persistence 2 there is no transition. Appending another `1`
confirms at index 4 with a delay of one bar from the new candidate run, without
backdating. Invalid observations do not count as persistence.

F13 moving-average `neutral_band=0` is supported at configuration and execution:
negative relative spread is state 0, positive spread is state 2, and exact equality
is neutral state 1. Positive bands retain inclusive neutral boundaries. Negative
or non-finite bands are rejected by parameter validation. General rules and
supervised rule targets still require strictly ordered lower/upper thresholds.

HMM forward filtering produces current probabilities. Forward-backward smoothing, fixed-lag smoothing and Viterbi are available as explicitly retrospective outputs. A future-state model produces a forecast, clearly distinguished from a live current-regime estimate.

Transition confirmation supports probability thresholds where available, minimum persistence and hysteresis. Events are recorded on confirmation bars, never backdated. Reported detection delay measures confirmation lag; it does not identify an unknowable true economic change time.

The dashboard includes raw/confirmed state, labels, available trend/volatility interpretation, duration, previous state, last transition, confidence, probabilities or distances, model version, observation and inference timestamps. Historical outputs contain assignments, timelines, probabilities, transition matrices, events, durations and frequencies.

Monitoring computes regime changes, low confidence, unusual observations, feature drift, state occupancy drift and optional CUSUM events. Performance deterioration is assessed only where realized targets are available. Monitoring runs on submitted observations; this module does not create a market-data subscription or an external alert-delivery service.

## Integration contracts

Implementation boundaries:

| Component | Location | Responsibility |
| --- | --- | --- |
| Input contracts | `backend/app/services/regimes/contracts.py` | Strict versioned configuration and request validation |
| Data adapter | `data.py` | Existing datasets, C6 boundaries, C7/C8 provenance and fitted replay |
| Model registry | `registry.py` | Capabilities, conditional parameters and dependency availability |
| Estimators | `estimators.py`, `hmm.py`, `structural.py`, `neural.py` | Fitting and method-compatible outputs |
| Experiments | `experiments.py`, `artifacts.py` | Search, diagnostics, immutable model/configuration bundles |
| Evaluation | `metrics.py`, `evaluation.py` | Analysis, labeling, chronological refits, comparison and approval |
| Inference | `inference.py`, `service.py` | Approved inference, transitions, monitoring and job orchestration |
| HTTP | `backend/app/routers/regimes.py` | Existing FastAPI application, `/api/regime` |
| Frontend | `frontend/src/pages/regimes/`, `frontend/src/api/regimes.ts` | Existing React, themes, shared grids, Plotly and polling |

Principal endpoints, relative to `/api/regime`:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/options?dataset_id=…` | Available methods, columns, plans and preprocessing |
| GET | `/plans/{id}` | Existing C6 spans |
| POST | `/experiments` | Submit fitting/search job |
| GET | `/artifacts?kind=experiment` | List saved records; also search/validation/labels/robustness/approval/inference |
| GET | `/artifacts/{id}` | Verified manifest |
| GET | `/artifacts/{id}/history?offset=0&limit=2500` | Paged assignments; limit ≤5,000 |
| GET | `/artifacts/{id}/export?format=json` | JSON metadata and complete history; `csv` for assignments |
| GET | `/artifacts/{id}/bundle` | ZIP of checksummed model/assignment files and dependency manifests |
| GET | `/experiments/{id}/evaluation` | Computed evaluation report |
| POST | `/experiments/{id}/labels` | Save explicit label mapping |
| POST | `/experiments/{id}/validation` | Chronological or walk-forward job |
| POST | `/experiments/{id}/robustness` | Independent robustness-refit job |
| POST | `/experiments/{id}/approval` | Approve with validation ID and rationale |
| POST | `/comparison` | Compare explicit experiment IDs |
| POST | `/inference` | Approved inference job |

Jobs return HTTP 202 and the existing job ID/poll contract. Requests use the application's existing project selection. Export links include `project_id` for project-scoped downloads.

Minimal fitting request (replace identifiers/columns with existing upstream outputs):

```json
{
  "name": "BTCUSDT multivariate regime experiment",
  "inputs": {
    "dataset_id": "PREPARED_DATASET_ID",
    "split_plan_id": "C6_PLAN_ID",
    "fold": 0,
    "features": ["returns", "volatility"]
  },
  "model": "multivariate_hmm",
  "parameters": {"covariance_type": "full", "max_iter": 100},
  "states": {"count": 3},
  "seed": 42
}
```

Minimal live inference request:

```json
{
  "approval_id": "D10_APPROVAL_ID",
  "dataset_id": "COMPATIBLE_OBSERVATIONS_ID",
  "input_mode": "prepared",
  "decoding": "filter",
  "minimum_persistence": 3
}
```

Python callers can use `RegimeService` inside the existing project context. Interactive API schemas are available through FastAPI `/docs`.

## Artifact persistence and reproducibility

Records are stored under the active project's `outputs/regimes/{ULID}/` with a catalog-anchored manifest checksum. Experiments include `manifest.json`, `model.bin`, and `assignments.parquet`. Manifests retain the explicit configuration, original optimization configuration, parameters, feature order, train bounds, data/preprocessing checksums, complete fitted preprocessing parameters, seed, diagnostics and runtime package versions.

Labels, validation, robustness, approval and inference are immutable related records. Approval references the exact experiment and validation. ZIP exports follow these dependencies and include model bytes, assignments and embedded preprocessing. Source market datasets and application source code are not copied into model ZIPs; retain the project dataset snapshot and matching runtime for reproduction. There is no arbitrary uploaded-pickle import endpoint.

## Installation and verification

From `backend/`, install `.[dev,regime-research]` to enable LightGBM, CPU PyTorch and Optuna:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,regime-research]"
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --disable-warnings
.\.venv\Scripts\python.exe -m mypy app
```

From `frontend/`: `npm test`, `npm run typecheck`, and `npm run build`.

Executed verification on 2026-11-04:

- Backend full suite: **1,416 passed, 8 skipped**. This includes upstream regression coverage and the three new regime test files.
- Backend type check: **67 source files**, no issues.
- Ruff checks pass for the regime implementation, its tests and updated verification tools.
- Frontend: **67 tests across 12 files** and a successful production build.
- Numerical tests cover HMM likelihood/posterior ground truth, causal prefixes, PELT optimality, supervised boundary purging, actual LSTM/GRU training and all registry models.
- Integration tests cover project isolation, tamper refusal, approval/inference, state-count search, optimization, fold-local walk-forward selection, C7 and C8 PCA replay, future-observation perturbation, and original-price diagnostics.
- A real headless Edge workflow used **2,131 local BTCUSDT daily observations**, prepared through existing C6/C7 services in isolated storage. D9 fitting → D10 validation/approval → D11 filtering passed with no browser errors. Dark/light themes and compact layout were exercised. This is execution evidence, not a claim that the reference feature configuration is economically useful.

Browser evidence: [verification report](../artifacts/regime-rebuild/verification.json), [D9](../artifacts/regime-rebuild/d9-results.png), [D10](../artifacts/regime-rebuild/d10-analysis.png), [D11](../artifacts/regime-rebuild/d11-dashboard.png), [light theme](../artifacts/regime-rebuild/d11-light.png), [compact layout](../artifacts/regime-rebuild/d11-compact.png).

The reproducible local browser driver is `backend/tools/verify_regimes_browser.py`.
Run from `backend/` after a frontend build, or enable its pytest entry with
`QM_E2E=1 python -m pytest tests/test_e2e_live.py::test_regime_endpoints -q`
(PowerShell: set `$env:QM_E2E='1'` first). It needs existing BTCUSDT daily data and
installed Edge. Each run copies observations into `.tmp/regime-review/<run-id>`,
writes reports to `artifacts/regime-rebuild/<run-id>`, starts its own temporary
server on port 8018, and terminates that server afterward. Prior reports and
datasets remain intact. `QM_REGIME_WEB_DIST` optionally selects an isolated
production build; otherwise the driver uses `frontend/dist`. The obsolete
`tools/e2e_regime.py` is not used. Other optional upstream E2E drivers retain
their separate shared-server fixture.

## Explicit limits and further work

### Phase 3 historical, UI and boundary corrections — F6, F7, F10, F12, F13

Verified on 2026-11-05. All five corrections are implemented; Phase 4 had not
started at this checkpoint. Regression tests were run against the pre-fix implementation:

| Finding | Original regression failures | Corrected coverage |
|---|---:|---|
| F6 | 6 | 7 backend cases: prepared/replayed historical changes, unchanged appends, saved appended inputs, refusal of revision overrides |
| F7 | 3 | 3 frontend cases: smoothing, fixed-lag and Viterbi saved context; API examples remain pinned while controls change |
| F10 | 2 | 2 backend entry-point cases; existing isolated browser driver enabled successfully |
| F12 | 6 | 10 backend cases: invalid gaps, warm-up, probabilities, confirmation delays and prefix invariance |
| F13 | 2 | 9 backend cases: zero/positive bands, invalid values, strict neighboring contracts and saved-model reopening |

New backend coverage is in `backend/tests/test_regimes_history.py`; frontend
coverage extends `frontend/src/pages/regimes/Regimes.test.tsx`. Existing tests
were retained. D9–D11 backend checkpoints passed 136 tests after F6, 136 after F7,
138 after F10, 148 after F12, and 157 after F13 (35 warnings at each checkpoint).
Final full backend: **1,529 passed, 8 skipped, 246 warnings** in 115.45 seconds.
The eight skips are opt-in live E2E entries; the D entry was separately enabled
and passed (final run: **1 passed in 33.67 seconds**). Other upstream live E2E
drivers were not enabled in this phase. Full frontend: **76 passed in 12 files**;
regime frontend: **13 passed**. Backend mypy passed for 67 source files, frontend
type checking and the production build passed, and Ruff passed on changed backend
files. The existing large JavaScript bundle warning remains.

The enabled browser test used 2,131 existing local BTCUSDT daily observations,
copied into isolated storage through the existing C6/C7 services. Gaussian HMM
fit, chronological validation, approval, filtering, price display, bundle export,
theme/layout checks and absence of browser errors were verified. The final report
is [verification.json](../artifacts/regime-rebuild/9a930fafcbb746a791bc7936d209d2f5/verification.json).
This is F10 workflow evidence, not complete C5→C8 integration, equity coverage,
all-family live verification or an assertion of economic validity. F6/F12/F13 have
computational/integration regression evidence; saved retrospective context F7 has
frontend regression evidence. The later Phase 4 results are in the final report.

HEAD remained `901fa080d8bb8c8d33a5238244e5154016119eea`. All 476 initial working-tree
entries were preserved, and no A1–C8 implementation was changed. The prior Phase 2
content-hash snapshot was unavailable after handoff, so preservation was checked
against a fresh pre-edit Phase 3 snapshot. All 1,056 durable files in the pre-E2E
data/report snapshot remained byte-identical. SQLite closed its pre-existing empty
WAL/shared-memory sidecars; persistent catalog bytes did not change. No commit,
migration, dependency change or artifact rewrite was performed. New E2E fixtures
and reports remain in their run-specific directories.

Historical revisions within an existing experiment remain unsupported. F6 checks
selected model coordinates with the documented replay tolerance, rather than
freezing optional target/diagnostic columns. It depends on retained source datasets
and scans prior saved runs; concurrent conflicting submissions and large archives
were not load-tested. Existing unsupported C7/C8 replay combinations remain
unsupported. F7 cannot reconstruct a missing legacy request. Browser verification
requires installed Edge, local BTCUSDT data, a production build and free port 8018.

### Phase 2 quantitative corrections — F5, F8, F9, F11

- **F5, return convention.** New experiments using `inputs.return_column` must also
  declare `inputs.return_type: "simple" | "log"`, including in the D9 form. Price-derived
  returns are always simple returns. Return moments and distributions stay in the
  declared units; unit-capital equity compounds as `cumprod(1 + r)` for simple returns
  and `exp(cumsum(r))` for log returns. Cumulative return is final equity minus one;
  episode drawdown uses equity divided by its running peak minus one. Two returns of
  `log(0.5)` therefore leave equity of 0.25 and a drawdown of -0.75. A simple return
  of -1 is total loss; values below -1, incomplete episodes, or nonfinite equity make
  that episode's drawdown unavailable. Disjoint state visits are not joined.
  Profiles and evaluation charts carry return-convention metadata. Old manifests
  remain readable without migration: an explicit return column with no declared
  convention yields unavailable return statistics and an explanatory note, while
  feature/model workflows remain usable. D never guesses from a column name or
  falls back to prices when an explicitly requested return column is unavailable.
  Create a new explicitly configured experiment to obtain unambiguous economics.
- **F8, occupancy.** All state proportions use assigned rows, with
  `0 <= state_id < state_count`, as their denominator. Unassigned/warm-up rows are
  excluded. Empty/all-unassigned spans yield zero proportions and unavailable
  occupancy drift. Profiles expose `occupancy_denominator`; training, validation
  and inference metadata explain the convention. For `[-1, -1, 0, 1, 1]`, counts are
  1 and 2, denominator 3, and occupancies 1/3 and 2/3. D10 charts use these backend
  proportions. D11's visible-history frequency chart explicitly shows assigned
  **counts**, not proportions. Full history/export files retain every row, including
  unassigned rows. Saved validation bundles preserve the denominator and note.
- **F9, PELT SSE.** Each active segment candidate maintains centered moments relative
  to its own first observation. Initial moments use centered two-pass arithmetic;
  subsequent rows use Welford's squared-deviation increment. This avoids differences
  of large cumulative squared sums and needs no negative-cost clipping. The L2
  objective, penalty, minimum segment length and delayed pruning are unchanged.
  Nonfinite costs are refused. Tests compare exhaustive small partitions, shifts of
  ±10^12, and large jumps followed by small-variance segments. Exactly representable
  shifted inputs use an accumulated-roundoff allowance of
  `128 * eps * n * max(1, expected_cost)`; random inputs also account separately for
  coordinate quantization (at most half an ULP per translated coordinate). These are
  binary64 test tolerances, not claims of exact arithmetic for arbitrary magnitudes.
- **F11, fold state alignment.** Alignment compares predictions at shared
  fold-training timestamps, requiring the same asset/timeframe and a shared primary
  observation ancestor. A shared auxiliary merge parent alone does not establish
  observation identity. Every branch's preprocessing is still validated by F1.
  Each model predicts its own full prepared history before timestamps are matched;
  fold coordinates are never fed into the reference model. Matching feature names
  or different scaler signatures do not establish a common coordinate space.
  Hungarian overlap and ARI use jointly assigned observations; fewer than two
  shared timestamps/assignments or unverifiable identity yields an explicit
  `state_alignment.unavailable` reason. This optional diagnostic does not block
  otherwise valid fold evaluation. Output records both preprocessing signatures,
  matched observation count and method. It is a retrospective label-switching
  diagnostic, never an OOS performance score, economic relabeling, or deployment map.

These four corrections are implemented and regression-tested in
`backend/tests/test_regimes_quantitative.py` and the regime frontend tests. Integration
tests execute the existing C6 and C7 services on temporary 160-row synthetic BTCUSDT
hourly fixtures: K-Means and Gaussian HMM use three separately scaled folds in both
rolling and expanding modes. Other tests cover persistence and exported bundles.
No A1–C8 implementation is changed, no fitted preprocessing is rebuilt inside D, and
no stored artifact is rewritten. Previously saved numerical reports are historical
outputs; regenerate validation to obtain corrected metrics. These tests do not verify
a fresh real-data C5→C6→C7→C8→D workflow, every PCA/preprocessing combination, numerical
behavior over every permitted dataset size, or new statistical/predictive validity.
At this Phase 2 checkpoint, Phase 3 and Phase 4 remained outstanding; the final
report records their subsequent completion and remaining verification limitations.

Phase 2 verification on 2026-11-05: 42 new backend cases and four new frontend
cases; the complete D suite passed 129 tests, the full backend passed 1,501 tests
with eight optional E2E skips, and the full frontend passed 73 tests. Backend mypy,
frontend type checking, targeted Ruff checks and the production build passed.
The existing large-bundle warning remains. Optional live E2E was not enabled:
repairing its obsolete entry point is F10 in Phase 3.

### Phase 1 critical corrections — F1–F4

- **F1, contributing lineage.** D traverses primary and additional catalog lineage
  parents, deduplicates shared ancestors, and rejects cycles, missing ancestors and
  unverifiable depth. Every encountered C7/C8 fit span is checked and its artifact
  identity is pinned. The shared upstream `Catalog.ancestors` behavior is unchanged.
  Secondary-branch transforms are recorded with `branch_only: true`; replay refuses
  such chains because D has no branch/merge column-mapping replay interface. Use the
  matching prepared output. Single-parent preprocessing signatures retain their shape.
- **F2, teacher cutoff.** Hybrid preprocessing must fit within the teacher's actual
  earlier training subwindow. The check runs before teacher fitting and when saved
  hybrids are reopened, using their recorded teacher-fit end index. The student's
  later targets still use causal teacher filtering. Supply an earlier fitted C7/C8
  artifact; D does not fit replacement preprocessing.
- **F3, predictive evidence.** Supervised and hybrid validation requires at least
  **two usable OOS prediction/target pairs** in the saved model's test span and in
  every requested walk-forward fold. This retains the former two-observation floor
  but counts realized, recognized-class labels with finite predictions after horizon,
  availability-lag and unassigned-state exclusions. `log_loss_rows` records this count.
  This is an operational minimum, not statistical significance or predictive quality.
  Poor accuracy alone does not imply insufficient evidence. Insufficient evidence
  raises an explicit refusal and creates no successful validation or approval record.
  Approval rechecks the saved model's current test evidence, and existing predictive
  approvals with insufficient recorded evidence cannot serve inference. HMM,
  clustering, deterministic and structural methods do not inherit this target-pair gate;
  PELT's existing live-approval refusal remains.
- **F4, predictions versus scoring.** Training requires only labels in its eligible
  historical target pairs; those required labels remain strict. Evaluation and
  monitoring select eligible label rows first, then exclude unavailable labels.
  Missing optional targets, including the final unrealized horizon, do not prevent
  approved inference. Monitoring scores the predictions actually returned using the
  saved training context. `monitoring.performance.available` describes scoring only;
  unavailable scoring has a reason and zero scored observations. Metrics also report
  `missing_target_count`. Numeric integer labels promoted to floats by missing values
  retain their class correspondence; textual class names remain exact. D11 displays
  prediction availability separately from realized-target scoring availability.

No artifact migration or deletion accompanies these corrections. Existing unsafe
contracts/teacher fits or evidence records may now be refused; retain them as research
records and create a corrected experiment/validation rather than rewriting them.
Regression coverage is in `tests/test_regimes_critical.py` and the regime frontend
tests, alongside the retained all-model, C7/C8 and causal-prefix tests. These checks
use temporary fixtures; they do not establish a fresh real-application-data workflow,
complete upstream causality. Later F5–F13 remediation and Phase 4 evidence are
recorded above and in the final report.

- Upstream replay integration is incomplete for rolling C7 chains and complex C8 pre-transforms. Supply verified prepared outputs from the same fitted artifacts; D refuses unsupported replay instead of fitting replacement preprocessing. Walk-forward folds need preprocessing fitted inside each training window.
- PELT currently supports L2 cost and at most 20,000 observations; it is retrospective and cannot be approved as a live state classifier. CUSUM segments are sequential events, not recurring semantic states.
- Student-t degrees of freedom are fixed; Bayesian optimization is TPE over discrete candidate values; recurrent training is CPU-based. Other emission/search/accelerator combinations are not claimed.
- Live inference replays complete post-training history. Incremental continuation checkpoints and externally scheduled/streaming inference are future extensions.
- Real-browser evidence covers the BTCUSDT Gaussian-HMM workflow. Other implemented methods have computational/integration tests; every real asset, timeframe and hyperparameter combination has not been exercised. Equity and additional crypto assets use the same dataset contracts.
- Statistical model approval remains a research decision. Test-set comparisons do not establish independent profitability or an untouched holdout after repeated researcher selection.
- Eight optional existing end-to-end tests remain skipped in the ordinary suite. Existing large frontend bundle warnings remain. The parent-requested Graphify command is unavailable in this environment.
