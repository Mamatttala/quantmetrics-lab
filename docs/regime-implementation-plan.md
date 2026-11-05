# D9–D11 replacement plan

The specification dated 2026-11-04 and `mermaid-diagram.png` replace the previous Module D design.

## Inspection

- React 19, TypeScript, React Router, handwritten theme CSS, Plotly, and ag-Grid.
- FastAPI, Pydantic, NumPy/SciPy/scikit-learn, SQLite catalog, Parquet datasets, project-scoped settings and background jobs.
- C5 supplies feature columns; C6 stores immutable chronological spans; C7 stores fitted affine parameters in lineage; C8 supplies PCA parameters and fit scope. D consumes these contracts without creating new cleaning/scaling/PCA workflows.
- Old D9 setup / D10 execution / D11 evaluation is incompatible with the requested flow and will be removed.
- Shared test fixtures currently live in an old D test. Move those fixtures before deleting it.
- Preserve existing A–C and E code and all user datasets. Historical D datasets are research records, not executable new-model approvals.
- Graphify was requested by parent repository guidance, but its executable is unavailable.

## Incremental work

1. Replace D-only implementation, routes, contracts and pages. Establish strict input validation and immutable, project-scoped experiment artifacts.
2. Implement baseline, K-Means, GMM, Gaussian/multivariate HMM; then Student-t/GMM emissions, PELT/CUSUM, and numerical tests.
3. Implement D10 diagnostics, state statistics, independent label mappings, transition statistics, validation, robustness, comparison and approval.
4. Implement D11 approved model deployment, causal inference, retrospective decoding, transition confirmation, monitoring, plots and JSON/CSV export.
5. Add supervised classification/forecasting, end-to-end recurrent networks, HMM–XGBoost teacher–student fitting, and real optimization backends. Report missing dependencies explicitly.
6. Verify BTCUSDT contract using deterministic numerical fixtures and available local prepared observations. Run upstream regressions, frontend tests/build, and browser checks where available.

## Methodological boundaries

- No imputation, sorting, resampling or feature generation inside D. Incomplete/misaligned inputs fail with the upstream integration requirement.
- Model fitting may use its training window; historical predictions are out of sample only after that window ends.
- Forecast labels are purged at train/validation boundaries. Teacher fitting and student validation have chronological separation.
- Preprocessing fitted after a fold's training boundary is refused. Reuse verified C7/C8 artifacts; never silently fit another scaler.
- PELT is offline segmentation, CUSUM is sequential alerting, and neither has fabricated class probabilities.
- Approval is an explicit application action following genuine chronological evaluation; it pins model, data, preprocessing, feature order and label mapping.
- Train/test metrics are distinct. Hyperparameter selection uses train/validation only. Test results are not optimization objectives.

## Completion record

All six phases have executable implementations. Verification includes 1,416 passing backend tests (8 optional skips), 67 passing frontend tests, backend/frontend type checks, targeted Ruff checks and a production build. Headless Edge completed the real-BTCUSDT D9 fit → D10 validation/approval → D11 causal inference workflow, including export and navigation checks, with no browser errors. The old D routes and dedicated implementation are removed; the shared browser driver now uses the replacement routes.

See [module-d.md](module-d.md) for operating instructions, detailed evidence and explicit model/replay limitations. The requested Graphify update was attempted; the executable is unavailable.
