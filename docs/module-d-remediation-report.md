# D9–D11 remediation — final report

Date: 2026-11-05. HEAD: `901fa080d8bb8c8d33a5238244e5154016119eea`.
This report supersedes the earlier remediation-status statements in the dated
Phase 1–3 notes. The prior audit and implementation-era records remain historical.

## 1. Executive summary

**F1–F13: 13 fixed, 0 partially fixed, 0 open.** Each defect has regression
coverage. Phase 4 added verification only; no model, preprocessing implementation,
forecasting capability, API, database migration or dependency was introduced.
All 18 model implementations remain present and pass their existing training tests.

The final backend suite passed **1,535 tests**, with eight opt-in E2E skips. The
frontend passed **76 tests**. All **163 D backend cases** passed in the full suite.
Ten representative families passed applicable workflows using the saved BTCUSDT
C6/C7 artifacts. Type checks, the production build and D-scoped Ruff passed.

The complete optional E2E suite was enabled against isolated, owned servers:
**7 passed, 1 failed**. The remaining failure is the existing E1 volatility driver
calling the removed `/regime/detect` API and expecting legacy semantic probability
columns. Current D11 assignments are not an E1 conditioning adapter. That unsupported
integration was not rebuilt or disguised as a successful test. Repository-wide
application Ruff also reports two pre-existing errors in unchanged shared files.

**Readiness:** D9–D11 is ready for further controlled research testing within the
verified contracts. This does not endorse the fitted smoke-test models, economic
labels, predictive quality or profitability. The overall master-prompt assessment
is **Remediation partially complete — further work required**, because the full
verification surface is not green, despite completion of all thirteen audited fixes.

## 2. Defect remediation matrix

Paths below are relative to `backend/app/services/regimes/` unless qualified.
Test names are in `backend/tests/`; frontend tests are in
`frontend/src/pages/regimes/Regimes.test.tsx`.

| ID | Severity | Root cause and corrected behavior | Implementation files | Regression evidence | Status |
|---|---|---|---|---|---|
| F1 | P1 | Primary-only ancestry missed secondary merge transforms. D now traverses every contributing parent and validates each fit span; unsupported branch replay is refused. | `data.py` | `test_regimes_critical.py::test_f1_rejects_full_fit_on_every_merge_branch`; valid branches, cycles and incomplete lineage | Fixed |
| F2 | P1 | Hybrid preprocessing used the outer training cutoff. Fit and reopen now enforce the actual earlier teacher cutoff. | `estimators.py`, `evaluation.py`, `data.py` | `test_f2_teacher_rejects_outer_train_fitted_preprocessing`, accepted cutoff and unsafe legacy reopening tests | Fixed |
| F3 | P1 | Raw test-row counts allowed approval with no usable targets. Predictive/hybrid validation and approval require at least two usable recognized-class OOS pairs, including every fold. | `evaluation.py`, `inference.py` | `test_f3_predictive_approval_requires_usable_pairs`, empty legacy evidence, valid negative results, unsupervised exemptions | Fixed |
| F4 | P1 | Target validation blocked inference outside scoring-eligible rows. Prediction generation and optional realized-target scoring are separated. | `estimators.py`, `metrics.py`, `inference.py`; frontend `Inference.tsx` | `test_f4_predictions_survive_unavailable_monitoring_targets` and related horizon/training tests; frontend availability tests | Fixed |
| F5 | P2 | Log returns were compounded as simple returns. Explicit return conventions now control equity and drawdown; ambiguous legacy economics are unavailable. | `contracts.py`, `metrics.py`, `experiments.py`, `evaluation.py`; frontend API/configuration/reporting | `test_regimes_quantitative.py::test_f5_equity_and_drawdown_respect_return_type` and saved-contract tests | Fixed |
| F6 | P3 | Saved timestamps were checked without their model inputs. Revised overlapping selected features are refused, including earlier accepted appends. | `inference.py` | `test_regimes_history.py::test_f6_revised_saved_features_refused`, unchanged/appended history and revision-override refusal | Fixed |
| F7 | P3 | Saved output reused unrelated form selections. Opening a run restores its request; API examples and result descriptions retain saved context. | frontend `Inference.tsx` | Three saved smoothing/fixed-lag/Viterbi restoration cases | Fixed |
| F8 | P2 | Profiles and occupancy used different denominators. Assigned observations are the consistent denominator; invalid rows remain in history exports. | `metrics.py`, `experiments.py`, `evaluation.py`, `inference.py`; frontend evaluation/inference | `test_f8_profiles_and_occupancy_share_assigned_denominator`, saved validation and chart tests | Fixed |
| F9 | P2 | Cumulative-square subtraction cancelled at large offsets. PELT uses locally centered moments and Welford increments without clipping. | `structural.py` | `test_f9_segment_sse_survives_large_translation`, exhaustive partitions, ±1e12 offsets and quantization-aware tests | Fixed |
| F10 | P3 | Optional D verification invoked a deleted script. It now invokes the existing isolated D browser driver. Phase 4 also corrected three stale D title expectations in the general UI driver. | `backend/tests/test_e2e_live.py`; `backend/tools/verify_regimes_browser.py`, `e2e_ui.py` | Two entry-point regressions; enabled D browser test and general UI test pass | Fixed |
| F11 | P2 | Reference models received another fold's transformed coordinates. Each model predicts its own valid representation; assignments align only at verified shared timestamps. | `evaluation.py` | `test_f11_fold_scalers_never_enter_reference_model` and invalid identity/overlap cases; Phase 4 predictive folds | Fixed |
| F12 | P3 | Pending persistence survived invalid-state gaps. Negative states reset the pending counter; confirmation is never backdated. | `inference.py` | Ten `test_f12_invalid_observations_reset_pending_persistence` cases, including exact delays and causal prefixes | Fixed |
| F13 | P3 | Registry permitted a zero moving-average band but the estimator rejected equal thresholds. Zero now preserves exact equality as neutral; other rule thresholds remain strict. | `estimators.py` | Nine band-boundary, invalid-parameter, neighboring-contract and persisted-model cases | Fixed |

The regression files contain **43 critical**, **42 quantitative**, and **28 Phase 3**
backend cases. Six Phase 4 integration cases supplement them. No existing assertion
was removed or weakened. The title update retains the existing route/title checks
and changes only their expected text to the actual supported D page titles.

## 3. Phase-by-phase results

| Phase | Work and evidence | Results | Failures and limitations |
|---|---|---|---|
| 0 | Read-only inspection and baseline, reviewed by the user | Retained baseline: backend 1,416 passed / 8 skipped / 213 warnings; frontend 67 passed; mypy 67 files passed | Baseline Ruff recorded an unused import in shared `catalog.py`. Live E2E was not enabled. |
| 1 | F1–F4, reviewed and approved | 43 critical regression cases pass in final rerun; valid all-family training and target-independent inference retained | A separate historical Phase 1 completion log is not retained here; do not infer its full-suite count. Branch-specific merged replay remains unsupported. |
| 2 | F5/F8/F9/F11, reviewed and approved | Recorded checkpoint: 129 D tests; full backend 1,501 passed / 8 skipped; frontend 73 passed; type checks/build passed | Real-data all-family and full upstream-chain evidence were not claimed at this checkpoint. |
| 3 | F6/F7/F10/F12/F13, reviewed and approved | 157 D tests; full backend 1,529 passed / 8 skipped; frontend 76 passed; enabled D E2E passed | Missing legacy request metadata cannot be reconstructed; history revisions remain unsupported. |
| 4 | Current integration, quantitative, real-data, full-suite and optional E2E verification | Backend 1,535 passed / 8 skipped; frontend 76 passed; ten real-data families passed applicable checks; optional E2E 7 passed / 1 failed | Legacy D→E conditioning remains unsupported. Two pre-existing shared-file Ruff errors remain. |

Phase 3 demonstrated original failures before each correction: F6 six failures,
F7 three, F10 two, F12 six and F13 two. All subsequently passed.

Phase 4's first added integration run had two fixture failures because C7 correctly
rejects `fold=0` on a single-span date plan. The verification fixture was corrected
to omit that fold; no upstream code changed. Initial optional E2E had six passes
and two failures. Correcting the three obsolete D title strings resolved the UI
failure; the unsupported E1 conditioning failure remains. New verification-file
format/import/closure lint findings were corrected. No failing product assertion
was skipped to produce a green result.

## 4. Verification results

### Executed suites

| Verification | Final result | Evidence |
|---|---|---|
| Full backend | **1,535 passed, 8 skipped, 246 warnings**, 101.39 s | [backend.xml](../reports/module-d-phase4/backend.xml) |
| Approved D regressions | **157 passed, 35 warnings**, 36.59 s | [approved-regressions.xml](../reports/module-d-phase4/approved-regressions.xml) |
| Added integration checks | **6 passed**, final focused run 9.47 s | [integration.xml](../reports/module-d-phase4/integration.xml) |
| Full frontend | **76 passed**, 12 files; 13 regime cases | [frontend.json](../reports/module-d-phase4/frontend.json) |
| Backend mypy | Passed, **67 source files** | Executed `python -B -m mypy app` with temporary cache |
| Frontend type check | Passed | Executed `npm run typecheck` |
| Production build | Passed, **116 modules**, 14.65 s | Existing >500 kB JavaScript bundle warning; isolated build directory |
| D implementation/tests and verification-driver Ruff | Passed | Existing/new D files checked; no upstream lint edits |
| All-application Ruff | **2 pre-existing errors** | `app/storage/catalog.py:21` F401; `app/models/naming.py:42` E501; both files unchanged |
| Optional E2E, all eight enabled | **7 passed, 1 failed**, 133.25 s | [e2e-final.xml](../reports/module-d-phase4/e2e-final.xml) |
| Representative real-data workflows | **10 applicable family checks passed** | [reliability.json](../reports/module-d-phase4/reliability.json) |
| D real-browser workflow | Passed, no browser errors | [verification.json](../artifacts/regime-rebuild/49db09785e5941b4b34f8480217a1336/verification.json) |

The eight default pytest skips are precisely the opt-in live E2E entries. All eight
were separately enabled. Cleaning, resample/merge, features, D, splits, scaling and
general UI passed. Volatility conditioning failed with HTTP 404 at
`backend/tools/e2e_volatility.py:500`, calling `/regime/detect`. It also expects
`open_time`, `prob_crisis`, `prob_trending_up` and related legacy columns; current
D11 emits a saved inference artifact with `timestamp`, state IDs and numbered
probabilities. No semantic state mapping or D→E compatibility adapter was invented.

The E2E runner replaces only the server fixture in memory with isolated storage
and owned processes. All eight existing test functions execute their existing
drivers against real HTTP servers/browser. No existing application process is
stopped. Initial failure evidence remains in [e2e.xml](../reports/module-d-phase4/e2e.xml).

### Quantitative and causal checks

- **F5:** two `log(0.5)` returns yield equity **0.25**, cumulative return **−0.75**
  and maximum drawdown **−0.75**. Simple-return compounding, invalid domains,
  contiguous episodes and missing/legacy conventions are separately tested.
- **F8:** `[-1, -1, 0, 1, 1]` uses denominator **3**, with occupancies **1/3** and
  **2/3**. Backend profiles, saved validation and frontend chart behavior agree.
- **F9:** PELT segment SSE and objective agree with independent exhaustive small
  partitions. Translation by ±10^12 is tested. The documented binary64 tolerance
  is `128 * eps * n * max(1, expected_cost)` for representable translated inputs,
  with coordinate quantization separately accounted for on nondyadic inputs.
- **F11:** distinct fold scaler parameters never enter the reference model.
  Shared observation identity/timestamps are required; insufficient or invalid
  alignment is explicitly unavailable. Alignment is a retrospective diagnostic,
  not an OOS score or an economic/deployment mapping.
- HMM numerical filtering/smoothing/Viterbi are checked against enumerated
  likelihood/posterior/path ground truth. Real Gaussian, Student-t and GMM-HMM
  D11 pipelines passed saved-context causal prefix tests and future-price
  perturbations through approved C7 replay. Smoothing, fixed-lag and Viterbi
  outputs were explicitly marked retrospective.
- XGBoost and LSTM real-data inference remained identical with missing optional
  target columns; scoring was unavailable. Existing F4 tests also cover missing
  tails, entirely missing targets, missing columns and incomplete horizons.
- Empty/insufficient predictive evidence still refuses validation/approval.
  Negative performance remains distinguishable from insufficient evidence.
  HMM/clustering/structural methods do not inherit predictive-pair requirements.
- F12's invalid-gap sequence does not confirm across `-1`; appending the next
  valid state confirms at index 4 with delay **1**, as documented.

These checks establish correctness for the executed inputs and paths. They do not
prove universal upstream causality, numeric accuracy at every possible magnitude,
or economic validity from passing tests.

### Upstream contracts and walk-forward coverage

The read-only inventory found 946 default-catalog records and 20 BTCUSDT-project
records. The default catalog contains 37 feature and 13 PCA outputs, but its
datasets have no symbol; they were not relabeled into apparently valid D research
inputs. The BTCUSDT project contains existing C6 and C7 outputs, but no saved C5
or C8 output. No representative equity dataset was found. No new asset was fetched.

| Contract | Executed evidence | Limit |
|---|---|---|
| C5 selected features | Existing C5 service generates `normalized_high_low_range` and `high_low_range` on the synthetic fixture; exact ordered columns are consumed by D | Real BTCUSDT project has no saved C5 output; no claim of a real C5→C8 chain |
| C6 chronology | Saved real 60/20/20 and five-fold expanding plans consumed; synthetic rolling/expanding plans consumed | No replacement split implementation |
| C7 fitted artifacts | Exact saved BTCUSDT scaler replay matches prepared close values; synthetic folds use distinct actual C7 fits | Real expanding run reuses the saved early 355-row fit; it is not presented as five newly fitted real scalers |
| C8 PCA | Actual existing PCA service output, training-only scope and supported replay consumed on synthetic observations | Real saved C8 output unavailable; rolling/complex preprocessing replay remains unsupported |
| Multiple parents | Primary/secondary/nested lineage and leakage regression tests pass through existing merge services | General branch-specific merge replay remains unsupported |
| Appended history | Prepared/replay append tests preserve prefixes and fitted identities; revised accepted appends are refused | Real test extends the prediction origin over already available observations; it does not fabricate new real bars |
| Historical preprocessing | Historical-origin replay truncates inputs and preserves approved artifact identity; teacher cutoff refusals pass | End-to-end causality is bounded by verified upstream feature contracts |

New predictive walk-forward tests execute both rolling and expanding modes, with
fold-specific C7 parameters and actual refits. XGBoost runs a two-value grid
inside each fold's training/validation span; each fold saves its own search and
two successful trials. Hybrid folds use scalers fitted through each teacher's
earlier cutoff. All folds enforce horizon **3** plus availability lag **1**, and
the usable pair count is the test span minus four observations.

This verifies that specific fold-local selection design, not arbitrary nested
optimization. Complex nested searches, all preprocessing combinations and every
model/fold configuration were not executed.

### Representative BTCUSDT dataset and outcomes

| Field | Executed value |
|---|---|
| Raw dataset | `01M6E361KFA3DBYW74M1QY6KR2` |
| Prepared C7 dataset | `01M6E390XFZG56RXRCXDB8FJ6V` |
| Observations / timeframe | **2,131 / 1d**, BTCUSDT crypto spot |
| Observation dates | **2021-01-01 → 2026-11-01**, UTC |
| Selected model feature | **`close`**, in the saved C7 coordinates |
| C6 plan | `01M6E3J363D1ZX7HT8WDZBSGVT`, 60/20/20 |
| Training | **1,278**, 2021-01-01 → 2024-07-01 |
| Validation | **426**, 2024-07-02 → 2025-08-31 |
| Test | **427**, 2025-09-01 → 2026-11-01 |
| C7 fitting interval | **355 observations**, 2021-01-01 → 2021-12-21 |
| Preprocessing signature | `6acd7ea15a61da1731b5ffba9d9f1be904872d1c9516ffdf8f97e2e21e924f03` |
| Hybrid teacher cutoff | Index **638**, **2022-10-01**; C7 fit precedes it |
| Predictive target | XGBoost/LSTM: predefined rules on prepared volume, thresholds −0.5/+0.5; hybrid: causal teacher states |
| Predictive horizon / availability lag | **3 / 1 bars** |
| Usable predictive test pairs | **423**, for each predictive/hybrid model |
| D11 mode / output count | **filter / 853** post-training observations, except offline PELT |

All model configurations are fixed smoke-test configurations, with seed 42,
two states where applicable and minimum occupancy 0. Complete parameters, model,
validation/approval/inference IDs, diagnostics and fitted artifact identities are
in `reliability.json` and the isolated saved manifests.

| Model | Fixed key parameters | Applicable workflow outcome |
|---|---|---|
| Gaussian HMM | max_iter 12 | Fit, evaluate, validate, approve, filter, causal perturbation, all retrospective decoders, bundle check passed |
| Student-t HMM | max_iter 12; registered defaults otherwise | Same applicable HMM checks passed |
| GMM-HMM | max_iter 12; registered defaults otherwise | Same applicable HMM checks passed |
| K-Means | n_init 3 | Fit/evaluation/approval/native inference/prefix/bundle checks passed |
| Gaussian mixture | max_iter 30, n_init 2 | Applicable workflow passed; convergence warning retained |
| XGBoost | n_estimators 5 | 423 pairs; approval/filtering/missing targets/bundle checks passed; saved five-fold expanding plan also validated |
| LSTM | epochs 2, hidden_size 4, sequence_length 5 | 423 pairs; actual recurrent fit, approval/filtering/missing targets/bundle checks passed |
| PELT | penalty 50, minimum segment 30 | Offline fit/evaluation passed; live validation/approval correctly refused |
| CUSUM | threshold 50, minimum segment 50 | Sequential fit/evaluation/approval/inference/prefix/bundle checks passed |
| HMM→XGBoost | n_estimators 5, teacher_max_iter 12 | Teacher cutoff verified; 423 pairs; approval/filtering/prefix/bundle checks passed |

The saved real expanding plan `01M6E36BT9TB8XG13X90TVS3VH` produced predictive
pair counts **351, 351, 351, 351, 352** across its five XGBoost folds. Each train
end precedes its test start. The final test ends 2026-11-01.

The three HMM smoke fits and GMM reported **converged=false** at their deliberately
bounded iteration budgets. LSTM ran two epochs, not a convergence study. The hybrid
had **0.0 test teacher agreement despite 423 usable pairs**; this verifies a valid
negative evaluation rather than insufficient evidence. Neither it nor the other
smoke models is recommended for deployment. No hyperparameter was selected from
these real test results. All verification approvals are confined to isolated storage.

Saved assignments match reloaded model predictions. Bundles pass ZIP integrity
checks and retain exact model bytes. The existing HTTP integration test verifies
JSON/CSV export contents; the real browser test verifies original price display
and bundle delivery. User datasets, fitted artifacts and prior verification reports
were not rewritten.

## 5. Changed files

This inventory separates remediation files from the extensive pre-existing user
working tree. No A1–C8 implementation file was modified in this remediation.
Phase 4 itself changed no product/backend/frontend implementation.

| File | Necessity |
|---|---|
| `backend/app/services/regimes/data.py` | F1 complete lineage and preprocessing validation |
| `backend/app/services/regimes/contracts.py` | F5 explicit return convention with legacy readability |
| `backend/app/services/regimes/estimators.py` | F2 teacher cutoff, F4 eligible targets, F13 neutral band |
| `backend/app/services/regimes/experiments.py` | F5/F8 contract and reporting propagation |
| `backend/app/services/regimes/metrics.py` | F4 scoring, F5 equity/drawdown, F8 occupancy |
| `backend/app/services/regimes/evaluation.py` | F2 reopen, F3 evidence gates, F5/F8 reporting, F11 alignment |
| `backend/app/services/regimes/inference.py` | F3/F4 deployment/scoring, F6 history, F8 monitoring, F12 transitions |
| `backend/app/services/regimes/structural.py` | F9 stable segment SSE |
| `backend/tests/test_regimes_critical.py` | Added F1–F4 regression coverage |
| `backend/tests/test_regimes_workflow.py` | Existing hybrid workflow fixture now uses a valid teacher-period C7 fit |
| `backend/tests/test_regimes_quantitative.py` | Added F5/F8/F9/F11 coverage |
| `backend/tests/test_regimes_history.py` | Added F6/F10/F12/F13 coverage |
| `backend/tests/test_regimes_integration.py` | Added six Phase 4 upstream/predictive-fold integration checks |
| `backend/tests/test_e2e_live.py` | F10 current driver and isolated D lifecycle |
| `backend/tools/verify_regimes_browser.py` | Preserve previous E2E outputs with per-run directories; isolated build selection |
| `backend/tools/e2e_ui.py` | Three obsolete D title expectations updated to current exact title prefixes |
| `backend/tools/verify_regimes_reliability.py` | Saved-real-data, ten-family verification and evidence generation |
| `backend/tools/verify_regimes_e2e_suite.py` | Execute existing optional suite using owned servers and isolated storage |
| `frontend/src/api/regimes.ts` | F5 return convention types/metadata |
| `frontend/src/pages/regimes/Discovery.tsx` | F5 explicit return selection and validation |
| `frontend/src/pages/regimes/Evaluation.tsx` | F5/F8 accurate reporting context |
| `frontend/src/pages/regimes/Inference.tsx` | F4 availability, F7 saved context, F8 assigned-history chart |
| `frontend/src/pages/regimes/Regimes.test.tsx` | F4/F5/F7/F8 frontend regression coverage |
| `docs/module-d.md` | Corrected operating policies and dated verification checkpoints |
| `docs/module-d-remediation-report.md` | This final report and evidence/limitation inventory |

Generated reports are under `reports/module-d-phase4/`. New real-data verification
bundles remain under `.tmp/regime-reliability/`; optional-suite fixtures/logs under
`.tmp/regime-e2e-suite/`; D browser fixtures/reports under `.tmp/regime-review/`
and `artifacts/regime-rebuild/`, each with its own run ID. The generated-file
manifest is [changes-manifest.json](../reports/module-d-phase4/changes-manifest.json),
with [preservation results](../reports/module-d-phase4/preservation.json) and
the [starting hashes/status](../reports/module-d-phase4/preservation-baseline.json).
HEAD is unchanged, all **514** initial working-tree status entries remain, and all
**1,120** original data/artifact hashes match. Of the **522** files in the starting
source/visible-file snapshot, only `docs/module-d.md` and `backend/tools/e2e_ui.py`
changed during Phase 4. Existing deleted and untracked files were preserved.
Temporary production output and the temporary inventory script were removed;
pre-existing TypeScript build metadata was restored byte-for-byte.

## 6. Remaining risks and limitations

1. **The full optional E2E suite is not green.** E1 conditioning still relies on
   a removed legacy D contract. Repair requires a separately scoped D→E integration
   decision; mapping numbered states to legacy economic probabilities is not implicit.
2. **Two pre-existing global Ruff findings remain** in unchanged shared files.
   They are separate from the clean D-scoped lint and passing type checks.
3. **No real C5→C6→C7→C8→D chain or equity workflow is claimed.** Required saved
   real C5/C8 outputs and representative equity data were absent. Synthetic actual
   service integration is explicitly distinguished from real BTCUSDT C6/C7 evidence.
4. **Replay has deliberate limits:** branch-specific merge transforms, rolling C7
   replay and complex C8 pre-transforms lack supported general replay interfaces.
   D refuses them; upstream modules were not rebuilt.
5. **Approval is not a statistical-quality certificate.** The two-pair floor is
   operational, not statistical significance. The recorded smoke fits include
   nonconvergence and a zero-agreement hybrid. Repeated research comparisons do not
   preserve an untouched holdout or establish profitability.
6. **Historical consistency is scoped:** selected prepared model coordinates use
   relative/absolute tolerance 1e-12; optional targets/diagnostic columns are not
   frozen. Prior source datasets must remain available. Concurrent conflicting
   submissions and large archives were not load-tested.
7. **Coverage is representative:** all 18 entries pass computational training
   tests, ten families ran the real saved-data workflow, and the dedicated real
   browser workflow used Gaussian HMM. Every asset/configuration/search combination,
   arbitrary nested optimization and every numeric extreme were not verified.
8. Existing large JavaScript bundle warnings remain. PELT remains offline/L2-only;
   CUSUM IDs are sequential events. No streaming scheduler, incremental state
   checkpoint, model family, forecasting feature or deployment adapter was added.

## 7. Final readiness assessment

**Remediation partially complete — further work required.**

All thirteen confirmed defects are fixed and regression-tested. The authorized
Phase 4 verification work is complete, and D9–D11's tested workflows support
further controlled research. The overall classification remains partial because
one enabled legacy D→E E2E integration fails and the global lint surface contains
pre-existing findings. These are reported explicitly rather than excluded from
the verification verdict. No unresolved F1–F13 defect is being hidden by that distinction.

No further phase, integration redesign or unrelated cleanup was performed.
