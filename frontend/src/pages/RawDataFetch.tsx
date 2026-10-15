import { useMemo, useState } from "react";

import { ApiError, api } from "@/api/client";
import { notifyDatasetsChanged } from "@/api/datasetsChanged";
import {
  MARKET_OPTIONS,
  timeframesFor,
  type AssetClass,
  type CryptoDataType,
  type FetchResult,
  type Timeframe,
} from "@/api/types";
import { PageHeader } from "@/components/PageHeader";

const CRYPTO_TYPES: { value: CryptoDataType; label: string; hint: string }[] = [
  { value: "klines", label: "Klines (OHLCV)", hint: "Price bars, from the REST API." },
  {
    value: "funding",
    label: "Funding rate",
    hint: "Futures only. From the public data dump.",
  },
  {
    value: "metrics",
    label: "Metrics (OI, L/S ratio)",
    hint: "Futures only. From the public data dump; 5m at source.",
  },
];

/** Local date string for a date input, offset from today. */
function isoDate(offsetDays = 0): string {
  const date = new Date();
  date.setDate(date.getDate() + offsetDays);
  return date.toISOString().slice(0, 10);
}

export function RawDataFetch() {
  const [assetClass, setAssetClass] = useState<AssetClass>("crypto_spot");
  const [cryptoType, setCryptoType] = useState<CryptoDataType>("klines");
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [timeframe, setTimeframe] = useState<Timeframe>("1h");
  const [start, setStart] = useState(isoDate(-30));
  const [end, setEnd] = useState(isoDate(-1));

  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<FetchResult | null>(null);
  const [error, setError] = useState<ApiError | Error | null>(null);

  const market = MARKET_OPTIONS.find((m) => m.value === assetClass);
  const isCrypto = market?.provider === "binance";

  // Funding and metrics exist only for futures, so under spot the data-type select
  // offers klines alone.
  const offeredTypes = useMemo(
    () => CRYPTO_TYPES.filter((t) => t.value === "klines" || assetClass === "crypto_futures"),
    [assetClass],
  );

  // Same guard as `effectiveTimeframe` below, and needed for the same reason. Switching
  // futures → spot leaves `cryptoType` on "funding", which spot does not offer, so the
  // select's `value` names no option it has: the browser then displays the first option
  // ("Klines") while React still holds "funding". Submitting would call the
  // futures-only funding endpoint while the user believes they fetched spot klines —
  // two different datasets, one of them not the one on screen.
  const effectiveCryptoType: CryptoDataType = offeredTypes.some((t) => t.value === cryptoType)
    ? cryptoType
    : "klines";

  const availableTimeframes = useMemo(
    () => timeframesFor(assetClass, effectiveCryptoType),
    [assetClass, effectiveCryptoType],
  );

  // Keep the selected timeframe valid when the market or data type changes.
  // Leaving a stale value selected would send a request the provider rejects.
  const effectiveTimeframe = availableTimeframes.includes(timeframe)
    ? timeframe
    : availableTimeframes[0];

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();

    // A date input can be cleared, and `new Date("T00:00:00Z")` is an Invalid Date on
    // which `.toISOString()` throws a RangeError. Because those conversions used to sit
    // above the `try`, that throw escaped the handler entirely — no error shown, and
    // `busy` left true, so the button stayed on "Fetching…" until a reload.
    if (start === "" || end === "") {
      setError(new Error("Pick both a start date and an end date."));
      return;
    }
    if (start > end) {
      // Both are `YYYY-MM-DD` from a date input, so lexicographic order is
      // chronological order and no parsing is needed to compare them.
      setError(new Error(`The start date ${start} is after the end date ${end}.`));
      return;
    }

    setBusy(true);
    setError(null);
    setResult(null);

    // Send an explicit instant, not a bare date. A date-only string is interpreted
    // in the browser's local zone by some clients and as UTC by others, which would
    // shift the fetched window by a day depending on where the user is.
    const startIso = new Date(`${start}T00:00:00Z`).toISOString();
    const endIso = new Date(`${end}T23:59:59Z`).toISOString();

    try {
      let response: FetchResult;
      if (!isCrypto) {
        response = await api.fetchYahoo({
          asset_class: assetClass,
          symbol,
          timeframe: effectiveTimeframe as Timeframe,
          start: startIso,
          end: endIso,
        });
      } else if (effectiveCryptoType === "klines") {
        response = await api.fetchKlines({
          asset_class: assetClass,
          symbol,
          timeframe: effectiveTimeframe as Timeframe,
          start: startIso,
          end: endIso,
        });
      } else if (effectiveCryptoType === "funding") {
        response = await api.fetchFunding({ symbol, start: startIso, end: endIso });
      } else {
        response = await api.fetchMetrics({
          symbol,
          timeframe: effectiveTimeframe as Timeframe,
          start: startIso,
          end: endIso,
        });
      }
      setResult(response);
      // A fetch lands a new catalog row. The explorer is mounted by the shell and does
      // not re-render on its own, so it has to be told.
      notifyDatasetsChanged();
    } catch (cause) {
      setError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      setBusy(false);
    }
  };

  const showTimeframe = isCrypto
    ? effectiveCryptoType !== "funding" && availableTimeframes.length > 0
    : true;

  return (
    <div className="page">
      <PageHeader
        title="Raw Data Fetch"
        lead={
          <>
            Download raw market data into immutable <strong>Parquet</strong> datasets from{" "}
            <strong>Binance</strong> and <strong>Yahoo Finance</strong>.
          </>
        }
      />

      <form onSubmit={submit} className="stack">
        <div className="row">
          <div className="field">
            <label htmlFor="market">Market</label>
            <select
              id="market"
              value={assetClass}
              onChange={(e) => setAssetClass(e.target.value as AssetClass)}
            >
              {MARKET_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {isCrypto && (
            <div className="field">
              <label htmlFor="dtype">Data type</label>
              <select
                id="dtype"
                value={effectiveCryptoType}
                onChange={(e) => setCryptoType(e.target.value as CryptoDataType)}
              >
                {offeredTypes.map((type) => (
                  <option key={type.value} value={type.value}>
                    {type.label}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="field">
            <label htmlFor="symbol">Symbol</label>
            <input
              id="symbol"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              placeholder={isCrypto ? "BTCUSDT" : "AAPL, ES=F"}
              spellCheck={false}
              autoComplete="off"
            />
          </div>

          {showTimeframe && (
            <div className="field">
              <label htmlFor="tf">Timeframe</label>
              <select
                id="tf"
                value={effectiveTimeframe}
                onChange={(e) => setTimeframe(e.target.value as Timeframe)}
              >
                {availableTimeframes.map((tf) => (
                  <option key={tf} value={tf}>
                    {tf}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        <div className="row">
          <div className="field">
            <label htmlFor="start">Start</label>
            <input
              id="start"
              type="date"
              value={start}
              onChange={(e) => setStart(e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="end">End</label>
            <input id="end" type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
          </div>
        </div>

        <div className="row">
          <button type="submit" className="btn" disabled={busy || symbol.trim() === ""}>
            {busy ? "Fetching…" : "Fetch"}
          </button>
          <span className="muted" style={{ fontSize: 12 }}>
            {isCrypto
              ? CRYPTO_TYPES.find((t) => t.value === effectiveCryptoType)?.hint
              : assetClass === "equity_futures"
                ? "Yahoo serves no settlement price or open interest; those columns are declared and left null."
                : "Raw Close and Adj Close are both kept."}
          </span>
        </div>
      </form>

      {error !== null && (
        <div className="result__bar result__bar--error">
          <span>{"code" in error ? (error as ApiError).code : "error"}</span>
          <span>·</span>
          <span>{error.message}</span>
        </div>
      )}

      {result !== null && (
        <div className="result__bar result__bar--ok">
          <span>{result.dataset.dataset_id}</span>
          <span>·</span>
          <span>{result.dataset.name}</span>
          <span>·</span>
          <span>{result.dataset.row_count.toLocaleString()} rows</span>
          <span>·</span>
          <span>{result.dataset.category}</span>
          {result.deduplicated && (
            <>
              <span>·</span>
              <span>duplicate of existing content — nothing new written</span>
            </>
          )}
        </div>
      )}

      {result !== null && result.warnings.length > 0 && (
        <div className="result__bar">
          {result.warnings.map((warning) => (
            <div key={warning}>{warning}</div>
          ))}
        </div>
      )}

      {result !== null && result.dataset.unpopulated_fields.length > 0 && (
        <div className="result__bar">
          <span>declared but unpopulated:</span>
          <span>{result.dataset.unpopulated_fields.join(", ")}</span>
        </div>
      )}
    </div>
  );
}
