import { Suspense, lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { Shell } from "@/layout/Shell";

import { Cleaning } from "./pages/Cleaning";
import { DataManagement } from "./pages/DataManagement";
import { FeatureEngineering } from "./pages/FeatureEngineering";
import { RawDataFetch } from "./pages/RawDataFetch";
import { RawDataView } from "./pages/RawDataView";
import { ResampleMerge } from "./pages/ResampleMerge";
import { ScalingPage } from "./pages/ScalingPage";
import { SplitsPage } from "./pages/SplitsPage";

/**
 * D1 and E1 are the only lazily-loaded routes, and the reason is specific rather than a
 * general preference for code splitting: they are the only pages that import Plotly. That
 * library is several times the size of the whole rest of the application, and a user who
 * never opens a chart page should not pay for it on first paint. Every other page is a
 * static import, because splitting pages that share nearly all their dependencies buys a
 * round trip and nothing else.
 *
 * The fallback is a plain panel rather than a spinner: the wait is a local chunk fetch
 * measured in milliseconds, and an animated indicator for that is noise.
 */
const RegimePage = lazy(() =>
  import("./pages/RegimePage").then((module) => ({ default: module.RegimePage })),
);
const VolatilityPage = lazy(() =>
  import("./pages/VolatilityPage").then((module) => ({ default: module.VolatilityPage })),
);

export function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route index element={<Navigate to="/fetch" replace />} />

        {/* A · Data Collection */}
        <Route path="/fetch" element={<RawDataFetch />} />
        <Route path="/view" element={<RawDataView />} />
        <Route path="/view/:datasetId" element={<RawDataView />} />

        {/* Utility */}
        <Route path="/data" element={<DataManagement />} />
        <Route path="/data/:datasetId" element={<DataManagement />} />

        {/* B · Data Preparation */}
        <Route path="/cleaning" element={<Cleaning />} />
        <Route path="/cleaning/:datasetId" element={<Cleaning />} />
        <Route path="/resample" element={<ResampleMerge />} />
        <Route path="/resample/:datasetId" element={<ResampleMerge />} />

        {/* C · Preprocessing */}
        <Route path="/features" element={<FeatureEngineering />} />
        <Route path="/features/:datasetId" element={<FeatureEngineering />} />
        <Route path="/splits" element={<SplitsPage />} />
        <Route path="/splits/:datasetId" element={<SplitsPage />} />
        <Route path="/scaling" element={<ScalingPage />} />
        <Route path="/scaling/:datasetId" element={<ScalingPage />} />

        {/* D · Regime Detection */}
        <Route
          path="/regime"
          element={
            <Suspense fallback={<div className="page panel muted">Loading charts…</div>}>
              <RegimePage />
            </Suspense>
          }
        />
        <Route
          path="/regime/:datasetId"
          element={
            <Suspense fallback={<div className="page panel muted">Loading charts…</div>}>
              <RegimePage />
            </Suspense>
          }
        />

        {/* E · Volatility Forecasting */}
        <Route
          path="/volatility"
          element={
            <Suspense fallback={<div className="page panel muted">Loading charts…</div>}>
              <VolatilityPage />
            </Suspense>
          }
        />
        <Route
          path="/volatility/:datasetId"
          element={
            <Suspense fallback={<div className="page panel muted">Loading charts…</div>}>
              <VolatilityPage />
            </Suspense>
          }
        />

        <Route
          path="*"
          element={
            <div className="page">
              <h1 className="page__title">Not found</h1>
              <p className="page__lead">No route matches this address.</p>
            </div>
          }
        />
      </Route>
    </Routes>
  );
}
