# Quantmetrics Lab architecture

The D9–D11 replacement follows the supplied [reference diagram](mermaid-diagram.png).

```mermaid
flowchart TB
  subgraph Upstream[Existing upstream application]
    A1[A1 Raw Data Fetch] --> A2[A2 Raw Data View]
    A2 --> B3[B3 Cleaning] --> B4[B4 Resampling / Merging]
    B4 --> C5[C5 Feature Selection] --> C6[C6 Chronological Splits]
    C6 --> C7[C7 Fitted Scaling] --> C8[C8 Analysis / PCA]
  end
  C8 --> Inputs[Prepared observations + ordered features + split plan + fitted artifacts]
  subgraph D9[D9 — Model Selection and State Discovery]
    Inputs --> Registry[Registry and conditional configuration]
    Registry --> Baseline[Rules / MA / ADX / Trend x Volatility]
    Registry --> Unsupervised[HMM emissions / K-Means / GMM]
    Registry --> Structural[PELT offline / CUSUM sequential]
    Registry --> Supervised[Logistic / Boosting / LSTM / GRU]
    Registry --> Hybrid[HMM teacher to XGBoost student]
    Baseline & Unsupervised & Structural & Supervised & Hybrid --> Search[State counts / manual / grid / random / TPE]
    Search --> Fit[Independent experiment jobs]
    Fit --> Artifacts[Versioned models + assignments + diagnostics]
  end
  subgraph D10[D10 — Evaluation and Interpretation]
    Artifacts --> Diagnostics[Diagnostics / state profiles]
    Diagnostics --> Labels[Explicit state-label mapping]
    Labels --> Robustness[Transitions / persistence / stability / alignment]
    Robustness --> Validation[Chronological / rolling / expanding validation]
    Validation --> Comparison[Model comparison]
    Comparison --> Approval[Researcher approval pins complete artifact contract]
  end
  subgraph D11[D11 — Inference and Current Regime]
    Approval --> Deploy[Approved version and preprocessing]
    New[Completed upstream observations] --> Cut[Availability-time cutoff]
    Cut --> Replay[Verify prepared inputs or replay saved parameters]
    Deploy --> Replay --> Filter[Causal filtering / native prediction]
    Replay -. retrospective only .-> Smooth[Smoothing / fixed lag / Viterbi]
    Filter --> Confirm[Persistence and hysteresis confirmation]
    Confirm --> Dashboard[Current state / probabilities / distances / events]
    Confirm --> Monitor[Drift / confidence / unusual observations / CUSUM]
    Dashboard & Monitor & Smooth --> Output[History / charts / JSON / CSV / downstream API]
  end
  Jobs[Existing background jobs] -. execution .-> Fit
  Jobs -. execution .-> Validation
  Store[Existing project catalog and Parquet storage] -. lineage .-> Inputs
  Store -. immutable bundles .-> Artifacts
```

Data access, models, experiments, evaluation, interpretation, inference, visualization and export remain separate modules within the existing application architecture.
