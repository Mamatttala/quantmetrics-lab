# Quantmetrics Lab — Context

> **Master architectural and contextual guide.** The single source of truth for what this
> project is, how it is built, and which invariants may not be broken. Where this document
> and the code disagree, the code wins and this file is wrong — fix it.
>
> Companion documents: [`DECISIONS.md`](./DECISIONS.md) holds the numbered record of every
> engineering decision and deviation (122 entries). [`README.md`](./README.md) holds setup and
> day-to-day commands. This file synthesises both against the actual tree.
>
> **Appendix A lists every claim in the source brief that the codebase contradicts.** Read it
> before quoting Section 3 — one of the brief's rules was the exact inverse of this project's.

---

## 1. Project Overview

**Name.** Quantmetrics Lab.

**Purpose.** A local, single-user quantitative research platform for crypto and equity
markets. It ingests raw market data, prepares it, engineers features, splits and scales it,
infers market regime, and forecasts volatility — with each stage independently runnable against
any dataset that is already in the right state, rather than as one rigid pipeline that must be
executed start to finish.

**Deployment model.** It runs on one machine, for one user. There is no authentication, no
multi-tenancy, no cloud dependency, and no telemetry: no data leaves the machine. This is a
deliberate constraint, not an unfinished feature — it is what allows the tool to be trusted
with a research process whose results must be reproducible years later.

**Core philosophy.** Stated as an ordered priority list, because the order is what resolves
conflicts:

1. **Quantitative correctness**
2. **Reproducibility**
3. **Prevention of look-ahead bias and data leakage**
4. **Transparent calculations** — every number is traceable to a stated formula
5. **Modular architecture**
6. **Maintainability**
7. **Clear user experience**
8. **Reliable testing**

Where a requested feature would conflict with a higher entry, the conflict is resolved in
favour of the higher entry and recorded in `DECISIONS.md` rather than resolved quietly. This
is the project's most load-bearing convention: **a silent methodological choice is treated as
a defect, not as a convenience.**

---

## 2. Technology Stack

### Frontend

| Component | Choice | Notes |
|---|---|---|
| Framework | React 19 + TypeScript 5.7 | `strict`, `noUnusedLocals`, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`, `verbatimModuleSyntax` |
| Build | Vite 6 | dev server on 5173, proxying `/api` to 8000 so the browser stays same-origin and CORS is never load-bearing |
| Routing | react-router-dom 7 | |
| Styling | **Hand-written CSS with design tokens** | `src/styles/tokens.css` is the single source of colour/radius/type. No Tailwind, no CSS-in-JS |
| Data grid | ag-Grid Community 33 | editable, paged server-side |
| Charts | lightweight-charts 5 (candlesticks), Plotly 2.35 (analytics figures) | Plotly is reached only through the two lazily-loaded routes that need it (regime, volatility); the charts page uses lightweight-charts |

**Theme.** A light/dark palette defined by CSS variables in `src/styles/tokens.css`, toggled by
`ThemeToggle.tsx` and persisted under the `theme` key. Components reference the variables, never
literal hex, so a theme switch is one attribute change. See the *UI Design System* section below.

### Backend

| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ (developed on 3.14.3) | |
| API | FastAPI (async) + Pydantic v2 | |
| Server | uvicorn | |
| Analytical queries | DuckDB | reads the Parquet files directly |
| Metadata & lineage | SQLite | `data/catalog.db` via the stdlib `sqlite3` — hence absent from `pyproject.toml` |
| Canonical storage | Parquet | one file per dataset, named by ULID |
| Interchange | CSV / JSON | import and export only, never canonical |
| HTTP client | httpx | provider fetches |
| Settings | pydantic-settings | `QML_` env prefix |

### Quant / ML libraries

**Actually installed and declared:** `pandas`, `numpy`, `scipy`, `statsmodels`, `arch`,
`xgboost`, `scikit-learn`, `pyarrow`. Plus `plotly` (Python-side, for exported figures) and
`yfinance`.

**Deliberately not used.** `hmmlearn`, `ruptures` and `lightgbm` are *not* dependencies. This
is a design position, not an oversight:

- The **Gaussian HMM and Gaussian mixture** are implemented directly in
  `app/services/regime_models.py` (log-space forward–backward, `scipy.special.logsumexp`,
  ridge-regularised covariance). Hand-rolling them keeps the filtered-vs-smoothed distinction
  explicit, which is the whole leakage story in Module D.
- The **hybrid classifier** is `xgboost.XGBClassifier`, a recorded substitution for the
  specification's LightGBM (`DECISIONS §75`). LightGBM is not installed and cannot be assumed
  to be; XGBoost is the same family of model with the same inductive bias, and the
  substitution is stated in the model's docstring, in the report, and in the decision record.
- There is **no change-point detector**; see Section 4.

`scikit-learn` is a narrow exception to the pattern above, and the reason is worth stating.
It is used for exactly one class — `neural_network.MLPRegressor`, the non-linear leg of Module
E's `tdnn_gjr_garch` — for the same reason `arch` is used for the GARCH family: a tapped-delay
MLP is not a model this project has an opinion about the *maths* of, and a hand-rolled
backprop would add a second gradient implementation to audit for no methodological gain. The
models the project does have an opinion about (the HMM, the GMM, Diebold–Mariano) are still
hand-rolled. §103 records the dependency-gap repair that declared both `scikit-learn` and
`xgboost` explicitly; before it, Module D imported XGBoost without declaring it.

---

## 3. Core Architectural Invariants (the Golden Rules)

These are enforced by test, not by convention. Breaking one is a defect even if every number
still looks plausible.

### 3.1 Causality (point-in-time correctness)

> For any row *t*, its value depends only on rows at or before *t*.

The operational form, and the one the suite asserts: **truncate the input and recompute; every
row the two runs share must be identical, bit for bit.** This is checked for every entry in
the feature catalogue, so a whole-sample statistic added later fails a unit test rather than
quietly inflating a backtest.

Consequences: no `center=True` in any rolling window (enforced by a source-level guard test
that greps the module), no backward fill, no `nearest` alignment, no scaler fitted on data a
model will be scored on.

### 3.2 Immutability and lineage

> Raw and uploaded data are read-only. Every transformation produces a new dataset carrying
> `parent_dataset_id`.

- Writing to an existing `raw_data` or `uploaded_data` path **raises**.
- Deriving leaves the parent's checksum unchanged.
- Lineage points at a real parent and is acyclic.
- `dataset_id` is a **ULID** and storage paths derive from it, so renaming never moves a file.
- `processed_data` is the one category that may be rewritten in place — it is the working
  tier, not the record.

The four categories are `raw_data`, `uploaded_data`, `processed_data`, `outputs`. `uploaded_data`
is deliberately distinct from `raw_data`: both are immutable, but a raw dataset has verifiable
provider provenance and an uploaded one does not.

### 3.3 Train-only scaling

> A scaler is fitted **only** on the training span of a split plan, and applied to everything.

Min-Max, Standard and Robust each estimate a fixed pair of numbers over a span of history, so
they require a split plan and are refused without one. The fitted pair is stored as its own
`outputs` dataset, so the transform is reproducible from data rather than from memory.

For a walk-forward (TSCV) plan, a **fold must be named**. Fitting on the union of every fold's
training rows would compute fold 1's statistics from rows that lie *after* fold 1's test span —
look-ahead relative to that fold (`DECISIONS §93`).

The rolling z-score is the exception that proves the rule: it estimates a fresh mean and
standard deviation from the trailing window at each row, so it needs no fit, cannot leak, and
cannot be given a plan.

### 3.4 Timestamp integrity

> Every timestamp is UTC-aware, strictly increasing, and unique — checked on the
> **observation** column, not on `close_time`.

This rule was stated incorrectly in the source brief (see Appendix A, item 2), and the
correction is safety-critical, so it is spelled out here.

- The timeline a split, a resample or a feature window is cut on is the **observation instant**
  — the bar's open. `quality.TIME_COLUMN_CANDIDATES` is `("open_time", "create_time",
  "calc_time", "Date", "date", "timestamp")`.
- **`close_time` is deliberately absent from that list.** `close_time` is the *end* of a bar,
  not the moment its values became knowable. Splitting or labelling on bar-close instants
  dates every span one bar late, and a "strictly sorted `close_time`" check — which the
  original specification asked for — would have validated the column that carries the leak
  while leaving the column that matters unchecked.
- The absence is asserted mechanically: `tests/test_leakage_guards.py` fails if `close_time`
  appears where an observation time is meant, and `splits.py` is checked to be free of it.
- Timestamps from uploaded CSVs arrive as **text** (`pandas.read_csv` types nothing) and are
  coerced exactly once, via `pd.to_datetime(..., utc=True, errors="coerce")`, with a refusal
  on any unparseable value. Reading them straight through is worse than it sounds:
  `is_monotonic_increasing` over strings compares lexicographically, which agrees with the
  clock only while every row is spelled identically. See `DECISIONS §101`.

---

## 4. Module Breakdown (current state)

Layout is `app/services/<module>.py` + `app/routers/<module>.py`. There is **no `app/modules/`**
anywhere in this codebase — a parallel layout for two of them would be a second convention to
learn for no gain (`DECISIONS §84`).

### Module A — Data collection

| Stage | What it does |
|---|---|
| A1 · Fetch | Binance (klines, funding, metrics) and Yahoo Finance ingest. Strict schema normalisation to UTC, deterministic checksums, retry with backoff, `SpanTooWide` refusal when a provider silently returns less history than requested |
| A2 · View | Candlestick view with quality overlays |

Binance funding and metrics are fetched from the `data.binance.vision` public dumps rather than
REST, because the specification's column names are the dumps' header rows, not REST field names
(`DECISIONS §4`).

Yahoo **daily** bars keep the **exchange-local session date** and record the timezone rather
than being coerced to UTC midnight: a session that closes at 16:00 New York is not a UTC
midnight boundary, and coercing it would misalign Yahoo daily bars against Binance daily bars
by a day.

### Module B — Data preparation

| Stage | What it does |
|---|---|
| B3 · Cleaning | Quality report; duplicate, missing-value and outlier policies; lineage |
| B4 · Resample / Align / Merge | Timeframe resampling, backward/exact alignment, left/inner joins, column rename/drop, 5+5 preview, export |

Two policy decisions here are load-bearing and non-obvious:

- **Forward fill and backward fill are separate options, never one "fill" switch.** A forward
  fill carries the last *past* observation into a gap; a backward fill carries the next
  *future* one backwards. On a price series that is the difference between a defensible
  assumption and a look-ahead.
- **Only `backward` and `exact` alignment are offered.** A `forward` fill would stamp a bar
  with a funding rate that had not been published yet, and `nearest` can do the same whenever
  the next observation is closer than the last — a leak invisible in the merged frame that
  would surface only as an unexplained backtest edge.

### Module C — Preprocessing

**C5 · Features.** A parameterised catalogue of **seven groups** — `price`, `volume`,
`returns`, `volatility`, `indicators`, `time`, `exogenous` — computed from a chosen price
source (`close` or `weighted`). Two structural distinctions matter:

- **`bars` vs `duration` windows.** A candle-count window is available for every dataset; a
  duration window ("15m", "1d") is meaningful only when the dataset declares a timeframe to
  divide it by. The two cannot be one control.
- **`computed` vs `carried`.** Exogenous entries are not calculated at all — they are columns
  another dataset already holds (a funding rate, an open interest) and selecting one keeps it.
  Nothing is invented, so no methodology can be invented for it.

Availability is decided per dataset and reported with a reason, not silently dropped.
Clock-time features (`hour_of_day`, `day_of_week`, …) are unavailable on bars of a day or
longer — every bar in such a dataset has the same time of day, so there is nothing to record —
and `annualized_vol` requires a declared timeframe, because √(365/252) is about 20% and
guessing the factor would be a silent multiplicative error rather than a missing value.

**C6 · Chronological splits.** Six procedures, each its own enum member rather than three with
options: `date_3way`, `date_2way`, `percentage_70_15_15`, `percentage_60_20_20`,
`tscv_expanding`, `tscv_rolling`.

- Spans carry **row indices** as well as timestamps, plus a `dataset_checksum` so a plan
  refuses to apply to a frame it was not computed on.
- **Boundary-leakage controls** for percentage and walk-forward splits: `gap_size` (an embargo
  dropping `[c−gap, c)`), `purge_flag` with a required `label_horizon_bars` (dropping every row
  whose label reaches the cut, `[c−h, c)`). They are **not additive** — the effective training
  cut is `c − max(gap, h)`, and the report states how many rows each rule actually removed. With
  `gap ≥ h` the purge removes nothing, and the report says `0` rather than implying otherwise.
- Gap and purge are **not** offered on date splits. Not because a date split is safe — the leak
  is identical — but because the user names that boundary themselves, and silently shrinking a
  span they typed would be worse than having no knob at all (`DECISIONS §88`).
- Walk-forward derives its window length from the fold count (`block = ⌊N / (n_folds + 1)⌋`)
  and prints the derivation rather than leaving it implicit.

**C7 · Scaling.** `minmax`, `standard`, `robust` (sklearn-free, reduced to one affine
`center_`/`scale_` pair per feature so a single inversion undoes all four), and
`rolling_zscore` — causal, `center=False`, enforced by a source-level guard test. The scaled
frame is the **whole frame** with the selected features replaced and every other column
untouched: no rows dropped, no columns invented. Which rows were train and which were test is
already in the plan, and both the report and the provenance say so.

### Module D — Regime detection

Deliberately **parallel to C, not downstream of it** — it can run on any dataset carrying the
observation columns, and its outputs feed back as features. Four detectors:

| Detector | Notes |
|---|---|
| `hmm` | Gaussian HMM, hand-implemented; log-space forward–backward |
| `gmm` | Gaussian mixture |
| `hybrid` | XGBoost classifier trained to reproduce the HMM's labels from faster indicators (`DECISIONS §75`) |
| `rules` | Volatility-breach risk control — no statistical model, works on a close column alone |

There is **no change-point/`ruptures` method** (see Appendix A, item 5).

Outputs are **soft probabilities** over four states (`trending_up`, `trending_down`, `ranging`,
`crisis`) plus a transition matrix. `transition` is a fifth *named* state but not a fifth
probability column: it is derived from the shape of the other four (no clear winner), so giving
it its own column would double-count the same probability mass.

Two controls exist because naive regime labelling produces unusable output:

- **Anti-thrashing** (`min_state_duration`): a state must persist for a minimum number of bars
  before it is committed. Without it, a model flips labels every few bars on noise.
- **Crisis priority override**: a volatility breach forces the crisis label regardless of the
  statistical model's posterior. This is a risk control and is kept distinct from estimation —
  conflating the two would hide that `rules` and `hmm` answer different questions.

Detector availability is gated on **which C5 features the dataset carries**, and each
unavailable option states what it needs: `hybrid` requires `adx` and `rsi`, which this module
deliberately does not compute itself — they are methodology-laden indicators the feature
catalogue already defines and documents, and a second implementation here would be a second
formula for one concept.

### Module E — Volatility forecasting

The first module whose subject is a bar that **has not happened yet**. That changes what
"correct" means: everything before it can be checked by re-reading its input, and a forecast
can only be checked by a protocol — a walk-forward loop, a loss function, and a test that says
whether one model's advantage is real or noise.

| Model | Family | Notes |
|---|---|---|
| `garch(p, q)` | A — arch | The baseline |
| `egarch(p, o, q)` | A — arch | Log-variance; the leverage effect is in the sign |
| `gjr_garch(p, o=1, q)` | A — arch | Asymmetry as a separate term on negative shocks |
| `arma_egarch_xgboost` | B — hybrid | EGARCH × an XGBoost correction learned on in-sample `ẑ²` |
| `tdnn_gjr_garch` | B — hybrid | GJR × a `MLPRegressor` on a tapped delay line |

Four shock distributions (`normal`, `students_t`, `skew_student`, `ged`), mapped to arch's own
names in one table. Two structural facts about the module:

- **Every model refits at every scored step.** This is not a performance choice — it is what
  makes the causality property possible, and it is what the truncation test asserts: fit and
  walk-forward over the whole series, and again over the first 80%, and the shared 80% prefix
  of every forecast series must be identical. A whole-sample fit passes every eyeball check and
  fails that one. A hybrid refits a *second* model inside the loop, which is why it is the
  slowest option on the page by a wide margin.
- **There are two forecasts, and only one of them is an observation.** The *scored* forecast is
  one-step-ahead at each walk-forward position — what MQLIKE, Mincer–Zarnowitz and
  Diebold–Mariano are defined over, and the only thing written to the store. The *forward path*
  is `forecast_horizon` steps past the last bar — what the fan chart draws, scored against
  nothing, and deliberately not a dataset row (`DECISIONS §104`). The page draws both on one
  time axis and labels them separately.

Evaluation is in **variance**, not volatility. `mqlike` is Patton's robust ratio form
(`mean(σ²/σ̂² − log(σ²/σ̂²) − 1)`, zero at a perfect forecast); `diebold_mariano` is hand-rolled
with a Newey–West Bartlett long-run variance and the Harvey–Laybourne–Newbold small-sample
correction, because statsmodels has no DM test and the two-line substitute over-rejects
(`DECISIONS §110`). The realised proxy is the **squared log return**, and the report states the
consequence rather than leaving it to be misread: a Mincer–Zarnowitz R² near zero is what a
correctly specified model looks like against a proxy whose variation is mostly unpredictable,
so the page ranks on MQLIKE and treats the R² as a calibration diagnostic.

Four further guards, all of them the same principle — a wrong number that would look right:

- Hybrid features are **lagged by construction** (`delayed_design` has no offset parameter), so
  a same-bar return cannot reach the feature matrix that predicts it; the exogenous block is
  shifted the same way, and a source-level guard pins the lag enumeration at 1.
- arch is fitted on **percentage returns** and the variance rescaled back, with the factor
  stated in the report — a conditioning transform, asserted by a round trip rather than assumed.
- A **NaN is counted, never filled**: each row carries `finite`, each model's score carries
  `n_finite`/`n_nan`/`n_inf`, and a model with too few finite steps gets no metric and a
  sentence explaining which of the two possible causes it was (`DECISIONS §113`).
- A **diverged step is refused rather than scored** (`DECISIONS §118`). A one-step variance
  forecast that is non-finite, not strictly positive, more than **100×** the largest squared
  return in its own training window, or below **1e-30×** that window's mean squared return is
  recorded as a failure, counted as non-finite, and left out of the frame and the forward path —
  never written as a number. Both bounds are *relative* because no fixed ceiling can be right
  across decimal returns and annualised basis points. The ceiling is chosen from the recursion's
  own shape plus measurement: over thirty windows of this project's fixture the legitimate ratio
  tops out at 15.2, and an unidentified EGARCH fit produced 3 495. The floor guards the other
  tail — a recursion underflows as well as overshoots — and is measured the same way: the
  legitimate ratio runs 0.35 to 7.9, under two orders centred on one, so thirty orders is a
  floating-point boundary rather than an estimate. Both tails cost the same thing: one runaway
  step had turned a run's headline MQLIKE into `4.7e293` and its RMSE into `inf` (which cannot
  even leave the process — the response renders with `allow_nan=False`), and a collapsed step is
  scored without complaint on a forecast the data never supported.

A variance forecast of exactly zero is dropped rather than floored, for the same reason: lifting
it to `1e-300` does not remove QLIKE's infinity, it moves it, making one step's loss larger than
every other step's combined. All four metrics align on the same surviving pairs, so a model's
MQLIKE and its RMSE are read off one sample (`DECISIONS §119`); `rmse` and `mae` subtract after
scaling by the largest magnitude present, so a forecast too large to square returns a finite
number instead of `inf`.

Every 422 now leaves in the application's own `error` envelope, whoever raised it — a
`RequestValidationError` handler sits beside the `QuantmetricsError` one (`DECISIONS §117`).
Before it, the four request models that refuse inside a `model_validator` answered FastAPI's
default `{"detail": [...]}`, which `src/api/client.ts` cannot read: the page degraded to a
generic failure for exactly the refusals written to be specific.

Fan bands are the **empirical quantiles of the walk-forward forecast-error ratio**, not a band
derived from the fitted distribution: the honest question is how wrong this model has actually
been on this series, and a distributional band would be narrower than the model has earned.

---

## 5. Key Engineering Decisions

A summary of the load-bearing entries in `DECISIONS.md`. The full record is 122 numbered
entries; these are the ones a new contributor is most likely to trip over.

| # | Decision | Why |
|---|---|---|
| §14, §86 | Splits validate the **observation** column, not `close_time` | `close_time` dates every span one bar late. See Section 3.4 |
| §75 | **XGBoost replaces LightGBM** in the hybrid detector | LightGBM is not installed and cannot be assumed; XGBoost is the same family, and an `ImportError` on a user's machine would be worse |
| §88 | Gap and purge are **not** offered on date splits | The user names that boundary; silently shrinking a typed span is worse than no knob |
| §95 | The split timeline is positioned by **row index**, not wall-clock time | A wall-clock Gantt renders an hourly and a daily series identically and cannot show an embargo at all; by index, every gap — asked for or not — appears as empty space exactly where it is. Plotly is several MB and lazily loaded on another route, so using it here would pull it into this page's chunk to draw rectangles |
| §93 | A walk-forward plan **requires a fold** in C7 | Pooled statistics would be computed from rows after an early fold's test span — look-ahead relative to that fold |
| §96 | A scaled frame lands in `processed_data`, and the plan is a **second parent** | It is still a frame of bars feeding the next module; the plan decides which rows produced the numbers, so it is recorded as a genuine input |
| §92 | Fitted scaler parameters are an **`outputs` dataset**, not a file beside the frame | Checksummed, lineage-linked, visible in the Explorer, exportable as JSON — one source of truth |
| §76 | Adding a table does **not** bump `SCHEMA_VERSION` | The version moves only for a change requiring user action; `CREATE TABLE IF NOT EXISTS` already handled it |
| §101 | An uploaded CSV's time column is **text** and must be coerced before any arithmetic | `is_monotonic_increasing` over strings compares lexicographically — spans would have been ordered, contiguous, non-overlapping, and covering the wrong bars |
| — | Yahoo daily bars keep the **exchange-local session date** | Coercing to UTC midnight misaligns them against Binance daily bars by a day, and 16:00 New York is not a UTC midnight boundary |
| — | `time_idx` carries a **loud caution** | Under a chronological split every test value lies outside the training range: a tree cannot split on it and a linear model extrapolates a trend that does not exist. It is offered because the specification asks for it, and discouraged in the report |
| §1, §13 | A registered-but-unbuilt route renders an explicit **"not implemented"** bar | An absent feature must not be mistakable for a broken one. A driver check fails if any route renders it while claiming to be built |
| §104 | A forecast run produces **two** objects and stores **one** | The scored one-step walk-forward is what the metrics are defined over; the forward path is a projection with no realised value to score it against, and a dataset row in this application is an observation, not a claim about the future |
| §105 | The forecast's join key is `open_time`, and `regime_engine.py` was renamed to say so | `close_time` is the instant a bar *finished*; a forecast joined on it aligns with the bar it predicts, one bar late, with both frames individually correct |
| §110 | Diebold–Mariano is hand-rolled, with the HAC variance and the HLN correction | statsmodels has no DM test, and the t-test substitute ignores the autocorrelation loss differentials always have — it over-rejects, which is exactly the wrong error when two volatility models usually *are* indistinguishable |
| §116 | Which metric is a ranking, and which direction is better, comes from the server | Crowning the highest MQLIKE is one character and inverts the report's conclusion; the page must not be able to make that mistake independently of the engine |
| §117 | Every 422 leaves in the `error` envelope, whoever raised it | Four request models refuse inside a `model_validator`, so FastAPI's default `{"detail": …}` was the shape most malformed requests actually got — and the client reads `error.code`, so the page showed a generic failure |
| §118 | A diverged fit is a **step that is refused**, not a value that is scored | One runaway EGARCH step (β pinned at 0.999, `convergence_flag` still 0) turned a headline MQLIKE into `4.7e293`, made RMSE `inf`, and wrote `8.3e213` into the output frame as though it were an observation. The bounds are relative — 100× the window's own largest squared return above, 1e-30× its mean below — because no fixed threshold works across return scales, and a recursion underflows as well as overshoots |
| §122 | What the module's own verification found | The module's tests, not the module, held the last several defects — including a report that dated the scored span one bar early while the frame it described was correct, invisible to any check that reads one of the two alone |

---

## 6. Current Status & Next Steps

### Completed and verified

| Module | State |
|---|---|
| **M0** scaffold, design tokens, three-pane shell, sidebar registry | Verified |
| **M1** storage layer, catalog, Data Management page, editable grid | Verified |
| **M2** A1 raw data fetch (Binance + Yahoo) | Verified |
| **M3** A2 raw data view (candlestick + quality overlays) | Verified |
| **B3** cleaning | Verified |
| **B4** resample / align / merge | Verified |
| **C5** features engineering | Verified |
| **C6** chronological splits | Verified |
| **C7** scaling | Verified |
| **D** regime detection | Verified |
| **E** volatility forecasting | Verified |

"Verified" means the full gate passes *and* the endpoint or page has been exercised against a
live server and a real browser — not that every input has been tried.

**Gate status — all green:**

| Command | Result |
|---|---|
| `ruff check .` | clean |
| `mypy app` | clean, 47 source files |
| `pytest` | 1284 passed, 8 skipped |
| `npm run typecheck` | clean |
| `npm run build` | built |
| `tools/verify_all.py` | all gates passed (includes all 8 e2e drivers) |

**Known limits, stated rather than implied:**

- **Yahoo is unrun.** Binance klines, funding and metrics have each been fetched live; no Yahoo
  fetch has been made from this repository, so the equity paths and the `SpanTooWide` guard
  have never executed.
- The UI has been driven **headless, in Edge only**, at one emulated viewport. Keyboard and
  focus order are not covered — every interaction is a synthetic `.click()`.
- **EGARCH is unidentified on a short window.** At the module's own `MIN_TRAIN_WINDOW` of 50
  bars its β pins at ≈ 0.999 on a GARCH-generated series — a random walk in log-variance that
  50 points cannot pin down — and the guard refuses the resulting forecast rather than scoring
  it (`DECISIONS §118`, §121). The floor is a floor on what may be *attempted*, not a promise
  that a four-parameter asymmetric recursion is estimable on it; the report says so and names a
  longer window as the fix. Measured on this repository's four synthetic series, the collapse is
  present at 100 bars on all four and clears at **150** (the causality test's fixture), **180**
  (the API test's fixture and the driver's four-hour frame) and **400** (the driver's hourly
  upload). Each fixture and each driver schedule therefore runs at the shortest window on which
  every model it fits estimates, which is why the numbers differ between them (`DECISIONS §122`).
- The dev store holds **559 datasets** (246 `processed_data`, 217 `outputs`, 91 `uploaded_data`,
  5 `raw_data`), most of them written by the e2e drivers. They are safe to prune.

### Pending

| Module | Scope |
|---|---|
| **F · Return prediction** | |
| **G · Signal engine aggregator** | |
| **H · Risk engine** | |

---

## 7. Developer Quick-Start

**Start the app** (kills any running `uvicorn app.main` and `vite`, relaunches both, waits for
health, and writes `backend/uvicorn.log` + `frontend/vite.log`, both gitignored):

```powershell
cd backend
python tools\serve.py
```

There is no `start_quant_app.bat` — it was named in the source brief but does not exist in this
repository. `tools\serve.py` is the one command. To run the two processes by hand:

```powershell
cd backend;  uvicorn app.main:app --reload --port 8000
cd frontend; npm run dev
```

Then open <http://localhost:5173>.

**Run full verification:**

```powershell
cd backend
python tools\verify_all.py
```

That runs lint → types → tests → typecheck → build → servers → all eight e2e drivers, and
prints one summary. `--no-lint` skips the first five when only the drivers are in question.

**The drivers** (`backend/tools/`) need both processes up. They are scripts rather than tests
because they require running servers; they are also reachable from `pytest` via
`$env:QM_E2E = "1"; python -m pytest -m e2e -q`.

| Driver | Covers |
|---|---|
| `e2e_part_b.py` | Cleaning policies and refusals, export formats |
| `e2e_part_b_merge.py` | Resample, alignment, joins, prefixes, tolerances |
| `e2e_features.py` | Catalogue, every feature at once, refusals, availability against real uploads |
| `e2e_splits.py` | Every split type, plan geometry, purge/embargo, refusals |
| `e2e_scaling.py` | All four scalers, per-fold fitting, refusal without a plan |
| `e2e_regime.py` | All four detectors, job handshake, diagnostics |
| `e2e_volatility.py` | Five models end to end, walk-forward geometry, the metrics and comparison tables, determinism across two runs, the diverged-fit refusal |
| `e2e_ui.py` | A real browser over the DevTools Protocol: every sidebar entry, form, chart, grid, and page |

**First-time setup:**

```powershell
cd backend;  python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -e ".[dev]"
cd ..\frontend; npm install
```

## 8. UI Design System

### Theme Infrastructure
- Uses CSS variables in `:root` (dark) and `[data-theme="light"]` (light).
- Persistence: `localStorage` key `theme`. Applied before first paint via inline script in `index.html`.
- Toggle Component: `ThemeToggle.tsx` (32x32 rounded-square sun/moon), rendered in the sidebar and the Data Explorer header.

### Color Palette (CSS Variables)
- **Backgrounds**: Main `--bg-main` (#111111/#fff), Sidebar `--bg-sidebar`, Explorer `--bg-explorer` (#111827/#f1f5f9), Cards `--bg-card`, Inputs `--bg-input` (#0f172a/#f8f9fa).
- **Accents**: Primary Gold `--accent-gold` (#d4a843/#b8860b), Secondary Purple `--accent-purple` (#a855f7/#9333ea).
- **Text**: Primary `--text-primary`, Secondary `--text-secondary`, Muted `--text-muted`, Code `--text-code`.
- **Borders**: Cards `--border-card`, Inputs `--border-input`, Explorer `--border-explorer`.
- **Shadows/Overlays**: `--tooltip-shadow`, `--modal-shadow`, `--overlay` (with backdrop-filter blur).

### Micro-UI Elements
- **Scrollbars**: Global 8px (Explorer 6px). Track `var(--bg-main)`, Thumb `var(--border-input)`, Hover `var(--accent-gold)`.
- **Tooltips/Popovers**: `.infotip__panel`, `.ctxmenu`. Z-index 9999, `var(--bg-input)`, `var(--tooltip-shadow)`.
- **Modals**: `.modal-backdrop` (overlay + blur), `.modal` (12px radius, `--modal-shadow`). Note: Native `window.confirm` cannot be styled.
- **Selection**: `::selection` uses `rgba(212, 168, 67, 0.3)`.

### Charts & Tables
- **Charts**: Plotly uses `chartTheme()` and lightweight-charts uses `candleTheme()` (both in `theme.tsx`). Bg `#0b0f19`/`#fff`, Grid `--chart-grid`.
- **Tables**: ag-Grid uses `ag-theme-quartz` / `ag-theme-quartz-dark`. Headers use Gold text.

---

## Appendix A — Corrections to the source brief

This document was commissioned from an outline. Eleven of its claims are contradicted by the
codebase. They are listed here rather than silently applied, because several would have made
this file actively misleading — and one would have inverted a safety invariant.

| # | The brief said | The codebase says |
|---|---|---|
| 1 | "overwriting the old fragmented `context.md`" | **No `context.md` existed** anywhere in the project. This file is new, not a rewrite |
| 2 | "**`close_time` is the absolute source of truth.** It must be strictly unique, ascending, and timezone-aware" | **Inverted.** `close_time` is deliberately *excluded* from `quality.TIME_COLUMN_CANDIDATES`. The observation column is the timeline; ordering on bar-close instants dates every span one bar late. A "strictly sorted `close_time`" check would validate the leaking column and leave the real one unchecked. See Section 3.4 |
| 3 | "Tailwind CSS" | No Tailwind in `package.json`, no config file. Hand-written CSS over design tokens in `src/styles/tokens.css` |
| 4 | Quant libs include "hmmlearn, ruptures, scikit-learn, lightgbm" | None is a dependency. The HMM and GMM are hand-implemented in `regime_models.py`; the hybrid is XGBoost (`DECISIONS §75`); there is no `ruptures` usage at all |
| 5 | Module D offers "HMM, GMM, Change-Point, Rules-based, Hybrid" (5 methods) | **Four** detectors: `hmm`, `gmm`, `hybrid`, `rules`. There is no change-point method |
| 6 | "Hybrid **HMM-LightGBM**" | Hybrid is HMM-labelled + **XGBoost** |
| 7 | "Why **`bounds_inclusive=true`** was added for date splits (to prevent DST timezone gaps)" | **No such flag exists**, and neither does the DST rationale. There is no DST handling in the project. The real recorded decision is `§88` (gap/purge are not offered on date splits, because the user names that boundary); date spans are resolved inclusive at both ends via `searchsorted(..., 'right')` because a date picker means inclusive dates |
| 8 | CSS Gantt chosen over Plotly for **performance** | The recorded reason (`§95`) is row-index proportionality — it is the only rendering that shows an embargo as empty space — with Plotly's bundle size as a secondary argument |
| 9 | `time_idx` caution | **Correct.** The catalogue entry carries exactly this caution (`feature_catalog.py`) |
| 10 | Yahoo daily bars keep exchange-local metadata | **Correct**, and the reason is recorded: UTC-midnight coercion would misalign them against Binance daily bars |
| 11 | Start via "`python tools/serve.py` (or the generated `start_quant_app.bat`)" | `serve.py` is correct; **no `.bat` file exists** in the repository |
| 12 | C5 groups: "Volatility, Momentum, Oscillators, Time, Exogenous" (5) | **Seven** groups: `price`, `volume`, `returns`, `volatility`, `indicators`, `time`, `exogenous` |

Two further notes, neither an error in the brief so much as an omission worth stating:

- **`README.md` is stale.** Its status table still shows C5 as "Built, unverified" and C6–C7 as
  "Not built", and its test count as 250. Section 6 of this file supersedes it; the README's
  setup and command sections remain accurate.
- **`app/modules/` does not exist.** The brief's instruction to describe `app/services/` and
  `app/routers/` is followed throughout; no module in this project deviates from that layout.
