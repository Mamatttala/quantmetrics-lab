# C8 — Exploratory Analysis

Open `/exploratory` or `/exploratory/:datasetId`. Every catalog category is
available, including outputs and split tables. The selected table is analyzed
directly. Correlation and PCA results remain independent until Clear Results.

## API

- `GET /api/exploratory/options[?dataset_id=…]`: all datasets; with a selection,
  numeric columns, grouping/carried columns, observation column and split plans.
- `POST /api/exploratory/correlation`: returns HTTP 202 with `job_id` and `poll`.
- `POST /api/exploratory/pca`: returns the same job contract.
- `GET /api/jobs/{job_id}`: completed diagnostics in `result`, or a typed failure
  in `error`. Read-only diagnostics create no datasets or parameter artifacts;
  their job record follows the existing durable job-registry behavior.

Both requests take `dataset_id`, `features`, and `scope` (default
`full_dataset`). `custom_date_range` takes inclusive `date_start`/`date_end`
boundaries; unzoned dates are UTC. `train_span` takes `split_plan_id` and, for
walk-forward plans, one `fold`. Plans must match the source ID, checksum and
observation timeline. `close_time` is never a feature, causal timeline or join key.
Full-sample diagnostics also work on tables without a recognized observation column.
`rolling` scope applies only to correlation. Legacy `start_time`/`end_time`, `target`,
`controls`, `group_column`, `strict_causal`, `name` and `keep_original_features` remain
accepted aliases. New callers can use the names in the expanded C8 brief.

## Correlation

`method` accepts `pearson`, `spearman`, `kendall`, `partial_pearson` and
`partial_spearman`. Optional `target` joins the matrix. Partial methods require
`controls` separate from features/target. `group_column` adds conditional tables
(up to 50 groups, with optional explicit numeric `group_bins`). Up to 250 features
and 30 controls can be selected. Every pair reports `n_used`; `min_obs=30` is
configurable, and insufficient pairs return null coefficients and inference.

Non-finite values are missing. Correlations delete rows pairwise, including
controls; VIF deletes rows across all selected features. Partial Spearman ranks
the complete observations for each pair before residualizing. VIF reports exact
redundancy as `status=infinite`, `vif=1e30`, and constants separately. VIF supports
an optional intercept (default on). Redundancy groups are deterministic connected
components above `redundancy_threshold` (default 0.9), with lowest-VIF/alphabetical
representatives and intra-group minimum/maximum/mean correlations. They suggest
changes without dropping features.

Pearson methods provide Fisher-z intervals at the requested confidence level.
Partial methods use effective control rank, `df=n-rank(controls)-2`. Spearman
p-values use a t approximation; Kendall uses SciPy tau-b. Rank-method intervals
are null unless fixed-seed paired bootstrap is enabled. Bonferroni or BH FDR can
correct p-values independently in each matrix/group. Inference assumes independent
rows and does not correct time-series autocorrelation. Group sizes and small-group
warnings accompany each matrix. Future-derived grouping remains diagnostic.

`enable_ccf` with `lag_min`/`lag_max` (default -10/+10, hard bounds -120/+120)
computes `corr(x_t, y_{t+L})`. Positive L means x
leads y and uses future targets. Lags count observed rows, preserve missing-row
positions, and are diagnostics only. CCF is never saved as row-level features.

`scope=rolling` computes correlations over preceding rows only, excluding the
current row. It is exclusive: static matrices, VIF, CCF, conditional groups and
redundancy are run under full/custom/train scopes instead. Window limit is 5,000
bars; minimum periods and step are explicit. Fitted preprocessing is refitted
within each window, with no backward filling. All-pairs mode returns summaries
only; explicit-pairs mode also returns row-level series. Full-sample top-N pair
selection is refused because future-aware selection would invalidate the causal
label. Output is bounded to 250,000 pair/time points. Summary stability is
`1/(1+sample_std_corr)`. Temporal diagnostics require a strictly increasing
observation timeline. Rolling PCA is deferred.

Differences, fractional percent changes, absolute returns and squared returns use
prior rows without fitting. Scaling, winsorization and mean/median imputation fit
only selected rows. Defaults are pairwise-complete missingness and exclusion of
fit-constant columns; alternatives are explicit. Rank transforms are diagnostic-only
and cannot be saved as transformed datasets. Price-level correlations produce a warning.

## Static PCA and reuse

`standardizer` accepts `none`, `standard` (sample standard deviation), `robust`
(median/IQR) or `minmax`. PCA always mean-centers after standardization. Fitting
uses complete finite rows, with at least two observations (warn below `min_obs`). Zero scales become
one. Component choices are `manual` with `n_components`, `variance_threshold`
with a ratio in (0, 1], `eigenvalue_gt_1`, deterministic `scree`, or fixed-seed
`parallel` analysis (maximum 200 permutations). If Kaiser selects nothing, retain
one component with a warning. Whitening refuses zero-variance components.
SVD signs make the largest absolute loading positive, breaking exact
ties by input feature order. Repeated eigenvalues can have rotated bases across
numerical-library versions; reproducibility is defined for identical inputs and stack.

The result includes the full eigenvalue spectrum, ratios, selected unit-eigenvector
loadings (features × components), component table and up to 500 fit scores for
the biplot. Biplot lines show unit loading directions alongside scores.

`write_dataset=false` is the default. With `write_dataset=true`, C8 creates:

- An `outputs` dataset for full/custom fitting, or `processed_data` for train fitting
  (destination can be overridden), operation `pca_dimensionality_reduction`, containing
  the observation column and `pc_1…pc_k`. `keep_original_features` and
  `carry_columns` optionally retain source columns. All source rows are transformed
  using the fixed fitted parameters, preserving row alignment and null scores
  for incomplete rows.
  Fitted imputation can supply missing inputs. A blank name uses the catalog's
  unique canonical source/scope/operation/ID naming; duplicate custom names are
  refused. The parameter artifact records fit,
  transform, missing and NaN score counts, preprocessing parameters, feature order,
  standardizer parameters and fixed sign flips.
- An `outputs` dataset, operation `pca_parameters`, with per-feature centers,
  scales, minima, maxima, PCA centers and loadings. Its provenance and lineage
  details also contain the complete parameter/report JSON, component selection,
  eigenvalues, explained variance, fit scope, fit-row count and warnings.

Both have direct source lineage and an additional split-plan parent for train
scope. Cross-references identify the transformed dataset and parameter artifact.
Both appear in Data Explorer through normal catalog storage. The reusable transform is:

```text
scores = (((preprocessed_X - center) / scale) - pca_center) @ loadings
# if whiten: divide scores by sqrt(retained eigenvalues)
```

`causal_for_downstream_training` is true for train-span PCA fitting and for the
exclusive causal rolling-correlation scope. Full
and custom-scope fits are allowed and labeled false. `strict_causal_mode=true` blocks
only saving such non-causal PCA outputs; diagnostics remain available. The flag
describes C8 fitting only and does not certify the upstream/downstream pipeline.
Rolling scope contains no full-sample fitted diagnostics, so its top-level causal
label is not diluted by a second non-causal procedure.

## Verification

`tests/test_exploratory_stats.py`, `tests/test_exploratory_causality.py`,
`tests/test_exploratory.py` and `tests/test_exploratory_api.py` cover arithmetic,
determinism, scope, immutability, lineage and jobs. `tools/verify_all.py` includes
the HTTP driver. The broader UI driver includes C8, or run only C8 with:

```text
python tools/e2e_exploratory.py --web http://127.0.0.1:5173
```

Use `--base` for a different API, `--catalog` to verify SQLite lineage edges,
and `--screenshot` for an optional browser capture. The driver creates synthetic
datasets and artifacts in the selected API's default project. Browser checks use
the existing Playwright dev dependency. Rolling PCA is outside this module.
