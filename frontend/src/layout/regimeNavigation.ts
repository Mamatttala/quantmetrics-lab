export const D_SECTIONS = {
  discovery: { badge: "D9", label: "Model Selection & State Discovery", sections: [
    ["baseline", "Baseline Models"], ["unsupervised", "Unsupervised Models"], ["supervised", "Supervised Models"],
    ["hybrid", "Hybrid Models"], ["states", "State Configuration & Discovery"], ["optimization", "Hyperparameter Optimization"],
    ["training", "Model Fitting & Training"], ["results", "Training Results"],
  ] },
  evaluation: { badge: "D10", label: "Evaluation & Interpretation", sections: [
    ["diagnostics", "Model Diagnostics"], ["analysis", "State Analysis"], ["labeling", "Regime Labeling"],
    ["transitions", "Transition & Persistence"], ["stability", "Stability & Robustness"], ["validation", "Walk-Forward Validation"], ["comparison", "Model Comparison"],
  ] },
  inference: { badge: "D11", label: "Inference & Current Regime", sections: [
    ["deployment", "Model Deployment"], ["inference", "Regime Inference"], ["filtering", "Filtering & Smoothing"],
    ["detection", "Transition Detection"], ["dashboard", "Current Regime Dashboard"], ["history", "Historical Visualization"],
    ["monitoring", "Monitoring & Alerts"], ["output", "Output & Integration"],
  ] },
} as const;
