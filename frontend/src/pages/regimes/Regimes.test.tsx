import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { Discovery } from "./Discovery";
import { Evaluation } from "./Evaluation";
import { Inference } from "./Inference";
import { regimes } from "@/api/regimes";
import { waitForJob } from "@/api/jobs";
import type { Artifact, EvaluationResult, Options } from "@/api/regimes";

vi.mock("./charts", () => ({ Chart: ({ title, data }: { title: string; data: unknown }) => <div aria-label={title} data-series={JSON.stringify(data)}>Computed chart</div>, Matrix: () => null, ProbabilityChart: () => null, StateChart: () => null }));
vi.mock("@/api/jobs", () => ({ waitForJob: vi.fn(), failureMessage: vi.fn() }));
vi.mock("@/components/ResultGrid", () => ({ ResultGrid: ({ caption }: { caption: string }) => <div>{caption}</div> }));
vi.mock("@/api/regimes", () => ({ regimes: { options: vi.fn(), plan: vi.fn(), list: vi.fn(), get: vi.fn(), fit: vi.fn(), evaluate: vi.fn(), label: vi.fn(), approve: vi.fn(), validate: vi.fn(), infer: vi.fn(), history: vi.fn(), exportUrl: () => "export", bundleUrl: () => "bundle" } }));

const options = {
  models: [
    { id: "gaussian_hmm", name: "Gaussian HMM", family: "hmm", available: true, probabilities: true, state_count: true, decoding: true, criteria: ["bic"], parameters: { covariance_type: { type: "select", default: "full", options: ["full", "diag"] } } },
    { id: "kmeans", name: "K-Means", family: "clustering", available: true, probabilities: false, state_count: true, decoding: false, criteria: ["silhouette"], parameters: { n_init: { type: "number", default: 10 } } },
    { id: "logistic", name: "Logistic Regression", family: "supervised", available: true, probabilities: true, state_count: false, decoding: false, criteria: ["validation_accuracy"], parameters: { C: { type: "number", default: 1 } } },
  ],
  datasets: [{ dataset_id: "prepared", name: "BTC features", symbol: "BTCUSDT", timeframe: "1h" }],
  plans: [{ dataset_id: "plan", name: "Chronological plan" }], features: ["returns", "volatility"], columns: ["returns", "volatility", "label"], preprocessing: [],
  optimization: { manual: true, grid: true, random: true, bayesian: true, optuna: true },
} as unknown as Options;
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(regimes.options).mockResolvedValue(options);
  vi.mocked(regimes.list).mockResolvedValue([]);
  vi.mocked(regimes.plan).mockResolvedValue([{ fold: 0, role: "train", start_time: "2024-01-01", end_time: "2024-02-01", row_count: 20 }]);
});
function mount(page: React.ReactNode) { render(<MemoryRouter>{page}</MemoryRouter>); }

describe("rebuilt D workflow", () => {
  it.each(["smooth", "fixed_lag", "viterbi"])("F7 restores saved %s context and keeps examples tied to the saved request", async decoding => {
    const request = { approval_id: "saved-model", dataset_id: "saved-data", input_mode: "replay",
      as_of: "2024-06-01T00:00:00Z", decoding, lag: 8, minimum_persistence: 4,
      probability_threshold: 0.7, hysteresis: 0.15, low_confidence: 0.3, drift_threshold: 0.4, cusum_threshold: 6 };
    const inference = { id: "saved-run", kind: "inference", name: "Saved retrospective run", created_at: "now",
      approval_id: "saved-model", request, snapshot: { retrospective: true, model: "gaussian_hmm", version: "v2" } } as unknown as Artifact;
    vi.mocked(regimes.options).mockResolvedValue({ ...options, datasets: [...options.datasets,
      { ...options.datasets[0]!, dataset_id: "saved-data", name: "Saved observations" }] });
    vi.mocked(regimes.list).mockImplementation(async kind => kind === "inference" ? [inference] : [
      { id: "unrelated", kind: "approval", name: "Unrelated clustering", created_at: "now", model: "kmeans" },
      { id: "saved-model", kind: "approval", name: "Saved HMM version", created_at: "now", model: "gaussian_hmm", version: "v2" },
    ]);
    vi.mocked(regimes.get).mockResolvedValue(inference);
    vi.mocked(regimes.history).mockResolvedValue({ rows: [], total: 0, offset: 0 });
    mount(<Inference />);
    await screen.findByRole("option", { name: /Unrelated clustering/ });
    fireEvent.change(screen.getByLabelText("Approved model version"), { target: { value: "unrelated" } });
    fireEvent.change(screen.getByLabelText("Market observations"), { target: { value: "prepared" } });
    fireEvent.change(screen.getByLabelText("Saved inference history"), { target: { value: "saved-run" } });
    await screen.findByText(`Retrospective research result · ${decoding}`);
    const control = (label: string) => screen.getByLabelText(label) as HTMLInputElement;
    expect(control("Approved model version").value).toBe("saved-model");
    expect(control("Market observations").value).toBe("saved-data");
    expect(control("Preprocessing mode").value).toBe("replay");
    expect(control("Historical prediction origin (optional)").value).toBe(request.as_of);
    expect(control("Decoding method").value).toBe(decoding);
    if (decoding === "fixed_lag") expect(control("Future lag (bars)").value).toBe("8");
    for (const [label, expected] of Object.entries({ "Minimum persistence (bars)": "4",
      "Confirmation probability threshold": "0.7", "Probability hysteresis": "0.15",
      "Low-confidence threshold": "0.3", "Distribution drift threshold": "0.4", "CUSUM alert threshold (optional)": "6" })) {
      expect(control(label).value).toBe(expected);
    }
    const example = () => JSON.parse(screen.getByText("Reusable inference API request").parentElement!.querySelector("pre")!.textContent!);
    expect(example().body).toEqual(request);
    fireEvent.change(control("Decoding method"), { target: { value: "filter" } });
    fireEvent.change(control("Market observations"), { target: { value: "prepared" } });
    expect(example().body).toEqual(request);
    expect(screen.getByText(`Retrospective research result · ${decoding}`)).toBeTruthy();
    fireEvent.change(control("Approved model version"), { target: { value: "unrelated" } });
    expect(screen.queryByText("Reusable inference API request")).toBeNull();
  });
  it("F8 displays backend occupancy with its assigned-observation denominator", async () => {
    const note = "Occupancy excludes unassigned observations; denominator: 3.";
    vi.mocked(regimes.evaluate).mockResolvedValue({
      diagnostics: {}, partitions: { test: { observations: 5, occupancy_note: note, predictive: {},
        profiles: [{ state_id: 0, occupancy: 1 / 3, feature_means: {} }, { state_id: 1, occupancy: 2 / 3, feature_means: {} }],
        transitions: { probabilities: [], expected_duration: [], empirical_duration: [], durations: [], events: [] } } },
      labels: {}, charts: {}, state_distribution_drift: {}, economic_note: "Simple returns.",
    } as unknown as EvaluationResult);
    render(<MemoryRouter initialEntries={["/?experiment=example"]}><Evaluation /></MemoryRouter>);
    await screen.findByText(note);
    const chart = screen.getByLabelText("State occupancy · assigned observations");
    expect(JSON.parse(chart.getAttribute("data-series")!)[0].y).toEqual([1 / 3, 2 / 3]);
  });
  it("F8 shows assigned counts in D11 while preserving all history rows", async () => {
    const inference: Artifact = { id: "saved", kind: "inference", name: "Saved history", created_at: "now",
      transitions: { probabilities: [], switching_frequency: 0, expected_duration: [], empirical_duration: [], events: [], durations: [] } };
    vi.mocked(regimes.list).mockImplementation(async kind => kind === "inference" ? [inference] : []);
    vi.mocked(regimes.get).mockResolvedValue(inference);
    vi.mocked(regimes.history).mockResolvedValue({ rows: [-1, -1, 0, 1, 1].map((state, index) => ({ timestamp: String(index), confirmed_state_id: state })), total: 5, offset: 0 });
    mount(<Inference />);
    await screen.findByRole("option", { name: /Saved history/ });
    fireEvent.change(screen.getByLabelText("Saved inference history"), { target: { value: "saved" } });
    const chart = await screen.findByLabelText("Regime counts · assigned visible history");
    expect(JSON.parse(chart.getAttribute("data-series")!)[0]).toMatchObject({ x: ["0", "1"], y: [1, 2] });
    expect(screen.getByText(/Showing rows 1–5 of 5/)).toBeTruthy();
  });
  it.each(["simple", "log"])("F5 requires and submits the declared %s return convention", async convention => {
    mount(<Discovery />);
    await screen.findByRole("button", { name: /Gaussian HMM/ });
    fireEvent.change(screen.getByLabelText("Prepared dataset"), { target: { value: "prepared" } });
    fireEvent.change(screen.getByLabelText("C6 chronological split plan"), { target: { value: "plan" } });
    fireEvent.click(screen.getByLabelText("returns"));
    fireEvent.click(screen.getByText("Economic diagnostics inputs"));
    fireEvent.change(screen.getByLabelText("return column"), { target: { value: "returns" } });
    fireEvent.click(screen.getByRole("button", { name: "Stage independent experiment" }));
    expect(screen.getAllByText("Error: Select simple or log returns for the return column.").length).toBeGreaterThan(0);
    expect(regimes.fit).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Return convention"), { target: { value: convention } });
    fireEvent.click(screen.getByRole("button", { name: "Stage independent experiment" }));
    vi.mocked(regimes.fit).mockResolvedValue({ id: "search", kind: "search", name: "test", created_at: "now" });
    fireEvent.click(screen.getByRole("button", { name: "Run 1 staged experiments" }));
    await waitFor(() => expect(regimes.fit).toHaveBeenCalled());
    expect(vi.mocked(regimes.fit).mock.calls[0]?.[0].inputs).toMatchObject({ return_column: "returns", return_type: convention });
  });
  it.each([false, true])("F4 distinguishes prediction availability from scoring (%s)", async available => {
    vi.mocked(regimes.list).mockImplementation(async kind => kind === "approval" ? [{ id: "approval", kind: "approval", name: "Predictive reference", created_at: "now", model: "logistic" }] : []);
    const inference: Artifact = {
      id: "inference", kind: "inference", name: "Predictions", created_at: "now", observation_count: 48,
      monitoring: { warnings: [], feature_drift: [], state_occupancy_drift: null, cusum_events: [],
        performance: { available, ...(available ? { metrics: { observations: 43, accuracy: 0.6 } } : { reason: "No complete prediction/target pairs in this span." }) } },
    };
    vi.mocked(regimes.infer).mockResolvedValue({ job_id: "job", status: "running", poll: "/jobs/job" });
    vi.mocked(waitForJob).mockResolvedValue({ job_id: "job", kind: "regime_inference", status: "succeeded", created_at: "now", started_at: "now", finished_at: "now", dataset_id: "prepared", output_dataset_id: "inference", message: null, error: null, result: inference });
    vi.mocked(regimes.history).mockResolvedValue({ rows: [{ timestamp: "2024-01-07", confirmed_state_id: 0 }], total: 1, offset: 0 });
    mount(<Inference />);
    await screen.findByRole("option", { name: /Predictive reference/ });
    fireEvent.change(screen.getByLabelText("Approved model version"), { target: { value: "approval" } });
    fireEvent.change(screen.getByLabelText("Market observations"), { target: { value: "prepared" } });
    fireEvent.click(screen.getByRole("button", { name: "Run approved model" }));
    await screen.findByText("Predictions available for 48 completed observations.");
    expect(screen.getByText(available ? "Realized-target scoring is available." : /Realized-target scoring is unavailable/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Historical assignments CSV" })).toBeTruthy();
  });
  it("renders model-specific configuration and supervised target controls", async () => {
    mount(<Discovery />);
    await screen.findByRole("button", { name: /K-Means/ });
    expect(screen.getByLabelText("covariance type")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /K-Means/ }));
    expect(screen.queryByLabelText("covariance type")).toBeNull();
    expect(screen.getByLabelText("n init")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Logistic Regression/ }));
    expect(screen.getByLabelText("Target source")).toBeTruthy();
    expect(screen.getByLabelText("Objective")).toBeTruthy();
  });
  it("stages independent configurations and sends the prepared-data contract", async () => {
    vi.mocked(regimes.fit).mockResolvedValue({ id: "search", kind: "search", name: "test", created_at: "now", trials: [] });
    mount(<Discovery />);
    await screen.findByRole("button", { name: /Gaussian HMM/ });
    fireEvent.change(screen.getByLabelText("Prepared dataset"), { target: { value: "prepared" } });
    fireEvent.change(screen.getByLabelText("C6 chronological split plan"), { target: { value: "plan" } });
    fireEvent.click(screen.getByLabelText("returns"));
    fireEvent.click(screen.getByRole("button", { name: "Stage independent experiment" }));
    fireEvent.click(screen.getByRole("button", { name: /K-Means/ }));
    fireEvent.click(screen.getByRole("button", { name: "Run 1 staged experiments" }));
    await waitFor(() => expect(regimes.fit).toHaveBeenCalled());
    expect(vi.mocked(regimes.fit).mock.calls[0]?.[0]).toMatchObject({ model: "gaussian_hmm", inputs: { dataset_id: "prepared", split_plan_id: "plan", features: ["returns"] } });
  });
  it("keeps approval unavailable without chronological validation", async () => {
    mount(<Evaluation />);
    await screen.findByText("D10.7 · Model Comparison & Final Selection");
    expect((screen.getByRole("button", { name: "Approve validated model for D11" }) as HTMLButtonElement).disabled).toBe(true);
    expect(regimes.approve).not.toHaveBeenCalled();
  });
  it("defaults to causal inference and hides incompatible probability/decoding controls", async () => {
    vi.mocked(regimes.list).mockImplementation(async kind => kind === "approval" ? [{ id: "approval", kind: "approval", name: "Validated clustering", created_at: "now", model: "kmeans" }] : []);
    mount(<Inference />);
    await screen.findByRole("option", { name: /Validated clustering/ });
    fireEvent.change(screen.getByLabelText("Approved model version"), { target: { value: "approval" } });
    expect((screen.getByLabelText("Decoding method") as HTMLSelectElement).value).toBe("filter");
    expect(screen.queryByRole("option", { name: /Viterbi/ })).toBeNull();
    expect(screen.queryByLabelText("Confirmation probability threshold")).toBeNull();
    expect(screen.getByText(/current dashboard requires an inference result/)).toBeTruthy();
  });
});
