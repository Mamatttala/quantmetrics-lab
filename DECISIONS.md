# Decision Record

Every point where the specification was ambiguous, silent, or in conflict with a
provider's actual behaviour is recorded here with the reasoning. The specification
requires that such choices be made deliberately and documented rather than invented
silently in code.

Format: **decision → why → where it lives**.

---

## 1. Repository root

**Decision.** The project is its own git repository, initialised inside
`Quantmetrics-lab/`, independent of any repository above it.

**Why.** `git rev-parse --show-toplevel` from the project directory originally
returned `C:\Users\Asus` — a repository rooted at the user's home directory. Every
file in the user profile (`.ssh/`, `.aws/`, `.config/`, browser profiles) showed as
untracked. A single `git add -A` there would stage credentials and personal data.
A separate repository removes that failure mode entirely.

**Where.** `.git`, `.gitignore`.

---

## 2. Equity futures: `settlement` and `open_interest`

**Decision.** Both columns are created and populated with nulls. The dataset's
`unpopulated_fields` records `["settlement", "open_interest"]` and the UI badges the
dataset as having known-empty columns.

**Why.** The specification asks for these fields in the equity-futures branch with
the note that they are daily-only. Yahoo Finance does not publish either one for
futures contracts. There is no free, local, reliable source, and this application
must not "silently invent financial methodology" — fabricating a proxy (for example
using the front-month contract's close as a settlement proxy) would produce a column
that *looks* like settlement data and is not.

A null column with explicit provenance is honest and keeps the schema stable, so a
future real source can backfill into the same shape without a migration.

**Where.** `backend/app/services/yahoo_fetch.py`, `DatasetMeta.unpopulated_fields`.

---

## 3. Data grid: ag-Grid Community with hand-rolled menu and clipboard

**Decision.** ag-Grid Community, plus a custom React context-menu overlay and a
custom clipboard layer.

**Why.** The specification asks for ag-Grid (Community) or MUI X *and* for a
right-click context menu and cut/copy/paste. Those two requirements conflict:
context menu and clipboard are **Enterprise** (ag-Grid) and **Pro** (MUI X) features.
Both cannot be satisfied with the free tier as written.

Rather than quietly shipping a grid missing half the requested features, or
depending on a paid licence, the missing pieces are implemented directly.
Virtualization, server-side pagination, cell editing, and column drag-reorder — the
parts that are genuinely hard to write correctly — come from ag-Grid Community.

**Trade-off.** Paste is text/TSV only, not Excel-native range paste, and the
clipboard layer is our code to test.

**Where.** `frontend/src/components/DataGrid.tsx`, `ContextMenu.tsx`, `clipboard.ts`.

---

## 4. Data source: Binance public data dumps, not only the REST API

**Decision.** Two different Binance sources, chosen per dataset:

| Data | Source | Why |
|---|---|---|
| Klines (spot + futures) | REST `/api/v3/klines`, `/fapi/v1/klines` | Current to the minute, arbitrary date ranges, no publication lag |
| Funding rate | `data.binance.vision` monthly/daily ZIP dumps | The only source that carries `funding_interval_hours` |
| Metrics | `data.binance.vision` daily ZIP dumps | Full history; REST caps at ~30 days per request |

**Why.** The specification names funding columns `calc_time`, `funding_interval_hours`,
`last_funding_rate` and metrics columns `create_time`, `symbol`, `sum_open_interest`,
`count_long_short_ratio`, `sum_taker_long_short_vol_ratio`. Those are **not** REST field
names — the REST funding endpoint returns `fundingTime` / `fundingRate`, and the REST
metrics endpoint is capped at 30 days. They are the exact header rows of Binance's public
data dumps, verified by fetching the files:

```
fundingRate  → calc_time,funding_interval_hours,last_funding_rate
metrics      → create_time,symbol,sum_open_interest,sum_open_interest_value,
               count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,
               count_long_short_ratio,sum_taker_long_short_vol_ratio
futures 1h   → open_time,open,high,low,close,volume,close_time,quote_volume,
               count,taker_buy_volume,taker_buy_quote_volume,ignore
spot 1h      → (no header row — see §5)
```

This supersedes an earlier entry in this document which claimed
`funding_interval_hours` must be derived from consecutive `calc_time` deltas. That claim
was wrong: the dump supplies it directly, and deriving it would have discarded a
provided field in favour of a computed one.

**Trade-off.** Dumps lag — daily files appear the following day, monthly files after the
month closes. For funding and metrics that is acceptable, since neither is needed
intraday. Klines, which are, come from REST.

**Where.** `backend/app/services/binance_fetch.py`.

---

## 5. Binance CSV header detection

**Decision.** The dump reader sniffs the first row: if the first field parses as a
number, the file is headerless and canonical column names are supplied positionally.

**Why.** Binance's dump files are not uniform. The `fundingRate` and `metrics` files
fetched during development carry header rows; the **spot klines** file does not — its
first line is data. Binance added headers to some families at some point in their
history, so a reader that assumes a header either loses its first bar or crashes,
depending on the family and date. Sniffing handles both without a per-family, per-date
table that would rot.

Verified by direct inspection of `BTCUSDT-1h-2024-01.csv` from the spot dump: the first
line is `1704067200000,42283.58,...`, not a header.

**Where.** `backend/app/services/binance_fetch.py`.

---

## 6. Canonical column naming

**Decision.** The three kline families are normalised to one naming scheme before
storage, and the source's own name is recorded in `provenance`.

| Canonical | Spot REST | Futures REST | Dump |
|---|---|---|---|
| `quote_volume` | `quoteVolume` | `quoteVolume` | `quote_volume` |
| `num_trades` | `numberOfTrades` | `numberOfTrades` | `count` |

**Why.** The specification names the futures columns `quote_asset_volume` and
`number_of_trades` (REST names) while naming the spot columns `quote_volume` and
`num_trades` (dump names). Persisting that inconsistency would mean every downstream
feature has to know which family it is looking at before it can name a column — and a
merge across spot and futures would silently produce two half-empty columns instead of
one.

**Where.** `backend/app/services/binance_fetch.py`.


---

## 14. Timestamp semantics

**Decision.** Binance `open_time` and `close_time` are both normalised to
`datetime64[ns, UTC]` and are *never* used as substitutes for one another. Yahoo
daily bars are stamped with the **exchange-local session date**, and the dataset's
`timezone` field records the exchange zone when one is known — and is `null` when it
is not.

**Why.** Two distinct hazards:

1. `open_time` is bar-open and `close_time` is bar-close. Using `close_time` as an
   observation timestamp while using `open_time` elsewhere silently shifts every
   feature by one bar — a look-ahead error that produces plausible, wrong results.
2. Coercing Yahoo daily bars to UTC midnight misaligns them against Binance daily
   bars by up to a day, because a US equity session starting 09:30 America/New_York
   is not a UTC midnight boundary. Storing the session date *with* its timezone keeps
   the meaning recoverable; converting to UTC is a deliberate downstream operation,
   not a silent ingest default.

**Correction (implementation).** An earlier version of this entry asserted that the
timezone is recorded "with an explicit `timezone` field". It is not, in the Yahoo
path: `routers/fetch.py` passes `timezone=None` and the API surfaces it as "not
stated". Deriving the exchange zone from the ticker suffix (`.L` → Europe/London,
`=F` → the underlying exchange) would be a guess presented as metadata, and asking
Yahoo for it is a second network call whose failure would have to be handled anyway.
An explicit `null` is the honest record of "we do not know", and the UI renders it as
such. Binance datasets do record `"UTC"`, which is a fact about the source.

**Where.** `backend/app/services/ingest.py`, `backend/app/routers/fetch.py`,
enforced by `backend/tests/test_leakage_guards.py`.

---

## 15. Adjusted close is additive, never a replacement

**Decision.** Yahoo's `Adj Close` is stored alongside raw `Close`, not instead of it.

**Why.** Adjustment factors are revised retroactively whenever a split or dividend
occurs, so a series of adjusted prices is not reproducible point-in-time: the value
recorded for a past date today is not the value that existed then. Building features
or backtests on adjusted prices alone leaks future corporate actions into the past.
Keeping both lets a researcher choose, knowingly.

**Where.** `backend/app/services/yahoo_fetch.py`.

---

## 7. Provider limits raise errors rather than truncating

**Decision.** A request outside a provider's permitted window raises
`SpanTooWide` (HTTP 422) naming the permitted range. It never returns a shortened
frame.

**Why.** A silently truncated result is the most dangerous possible outcome: the
fetch reports success, the row count looks plausible, and every downstream statistic
is computed on less data than the researcher believes. The frontend branches on the
error code to offer a corrected range.

**Where.** `backend/app/core/errors.py`, `backend/app/services/*_fetch.py`.

---

## 8. Metadata fields beyond the specified fourteen

**Decision.** Three fields are added to the dataset metadata schema:

| Field | Purpose |
|---|---|
| `category` | The Data Explorer's four buckets. The spec describes this grouping in prose but names no field for it. |
| `unpopulated_fields` | Columns present but known-empty (see §2), so nulls are not misread as missing data. |
| `provenance` | Per-column note for values this application derived rather than received (see §4). |

**Why.** Each carries information that would otherwise exist only in this document
and be invisible to the application at runtime.

**Where.** `backend/app/models/dataset.py`.

---

## 9. Parquet canonical, CSV export-only

**Decision.** Parquet is the canonical internal format. CSV is import/export only.
JSON is portable result/config export only.

**Why.** CSV has no type information and no timezone semantics, so round-tripping a
timestamped dataset through it loses exactly the properties §14 depends on. Parquet
preserves Arrow types including `timestamp[ns, tz=UTC]`, and DuckDB and PyArrow both
read it without a parse step.

**Where.** `backend/app/storage/parquet_store.py`.

---

## 10. Uploaded datasets are immutable too

**Decision.** `uploaded_data` is immutable exactly as `raw_data` is. Editing an
upload derives a new dataset in `processed_data`.

**Why.** They differ in provenance, not in mutability. Making uploads editable in
place would mean the application's guarantee — "the dataset you looked at last week
is the dataset you are looking at now" — holds for fetched data and not for
uploaded data, which is worse than not having the guarantee at all.

**Where.** `backend/app/storage/parquet_store.py`, `backend/app/services/datasets.py`.

---

## 11. Storage paths derive from `dataset_id`, not from the name

**Decision.** A dataset's file path is built from its ULID and category. Renaming
never moves a file.

**Why.** It keeps rename as pure metadata (no partial-failure window where the
catalog and disk disagree), lets names collide harmlessly, and keeps the path
injective.

**Where.** `backend/app/storage/parquet_store.py`.

---

## 12. One SQLite connection guarded by a lock

**Decision.** A single `sqlite3` connection, `check_same_thread=False`, serialised by
an `RLock`, in WAL mode.

**Why.** This is a single-user local application; a connection pool buys concurrency
there is no work for while making the write path harder to reason about. WAL keeps a
long explorer read from blocking a fetch commit.

**Where.** `backend/app/storage/catalog.py`.

---

## 13. Verification status of unbuilt pages

**Decision.** Routes that are registered but unbuilt render an explicit
"not implemented — arrives in <milestone>" bar.

**Why.** A plausible-looking mock is worse than a blank page: it makes an
unimplemented feature indistinguishable from a broken one, and invites a bug report
against something never claimed to work.

**Where.** `frontend/src/pages/Pending.tsx`.

---

## 16. Grid pagination is a manual page fetch, not ag-Grid's infinite row model

**Decision.** The grid fetches one page at a time (`GET /datasets/{id}/page`) and holds
only that page. ag-Grid's `clientSide` row model virtualises within it. Page size is
selectable (50/100/250/500) and capped server-side by `max_page_size`.

**Why.** Both satisfy the specification's requirement that the backend send only 50–100
rows per page, so the deciding factor is testability. With an explicit offset/limit, a
test can assert that page *n* and page *n+1* are disjoint and that their union covers
exactly `row_count` — which is what "server-side pagination" actually promises. The
infinite row model would push that arithmetic into the grid's internals, where the same
assertion is much harder to make, and would re-request blocks on every scroll.

**Consequence.** Scrolling past the end of a page does not continue into the next one;
the pager is the only way across. That is a real usability cost and is accepted
deliberately. Switching to `infinite` later is a contained change to `DataGrid.tsx`
alone, because the paging contract already lives on the server.

**Where.** `frontend/src/components/DataGrid.tsx`, `backend/app/services/datasets.py`.

---

## 17. Sorting is server-side

**Decision.** Clicking a column header re-queries the backend with `sort=column:asc|desc`
rather than sorting the loaded rows.

**Why.** Sorting only the current page presents that page's ordering as the dataset's.
The result is not merely incomplete — it is confidently wrong, and it looks exactly like
a correct answer. Only one column sorts at a time, because the server contract is a
single `ORDER BY`.

**Enforcement.** ag-Grid multi-sorts on shift-click by default, and that default was
left in place: `handleSortChanged` resolved the sort key as the first sorted column in
*display* order, while ag-Grid ordered its own rows by *sort priority*. Shift-clicking a
column to the left of the current one made the two disagree — the server chose which
rows to send by one key and the grid displayed them by another, yielding a page sorted by
neither. `suppressMultiSort` is now set on the grid, so the UI cannot enter a state the
server contract cannot express.

**Where.** `frontend/src/components/DataGrid.tsx`, `backend/app/storage/duckdb_engine.py`.

---

## 18. Quality markers may carry a null timestamp

**Decision.** A quality marker's `time` is UNIX seconds, or `null` when the row's
timestamp cannot be parsed. The API emits `null`; the chart drops those markers while
the marker list still shows them.

**Why.** Three of the five checks (non-positive price, `high < low`, extreme return) do
not depend on the timestamp being valid, so a row can be genuinely defective *and* have
an unreadable timestamp — which is arguably the most interesting row in the file.
Substituting a sentinel (zero, or the epoch) would place a marker at the start of the
chart corresponding to nothing; dropping the marker silently would hide the defect.
`null` states the fact, and the two views present it appropriately.

This was found by inspection, not by a failing test: the original `_epoch` called
`int(value)`, which raises on `NaN`. Reading a row with a `NaT` timestamp through the
price-sanity check would have returned HTTP 500.

**Where.** `backend/app/services/quality.py`, `frontend/src/charts/Candles.tsx`.

---

## 19. Cell editing is locked while a sort is applied

**Decision.** While the grid is sorted (§17), cell editing and the two mutating
clipboard commands (Cut, Paste) are disabled, and `Ctrl+X` / `Ctrl+V` are suppressed.
A toolbar notice states why, and a **Clear sort** button returns the grid to file
order. Copying is unaffected — it only reads.

**Why.** A cell edit is sent as a row *number*, and both ends of that number are
positional:

* the grid sends `offset + index-within-page`, and the index is a position in the
  window the server returned;
* `DatasetService._apply_cell_edits` resolves it with
  `frame.iat[row, frame.columns.get_loc(column)] = value` against the dataset in its
  own file order.

Those agree only when the server returned rows in file order — which is exactly what
a sort stops being true. With `sort=close:desc` applied, the row displayed at the top
of the grid is the dataset's largest `close`, while row `0` of the file is its first
bar. An edit made in that state would write the user's value to a different row than
the one they clicked, with no error and no visible sign: the grid would even redraw
correctly, because the page is re-fetched sorted.

This is a silent-corruption path, so it is closed rather than mitigated. The
alternative — sending the row's identity (a timestamp, or an explicit file row index
alongside every page) — would need a stable key, and a raw dataset cannot be assumed
to have one: duplicate timestamps are one of the quality checks precisely because they
occur. Adding a synthetic row index column to every dataset would be the larger
change, and it is not needed to make the common case correct.

**Consequence.** A user cannot fix a value in the middle of a sorted view without
clearing the sort first. That is a real cost and is accepted deliberately: it is
strictly better than an edit that lands somewhere the user did not choose.

**Where.** `frontend/src/components/DataGrid.tsx` (the `editable` gate),
`backend/tests/test_paging.py::test_unsorted_pages_are_in_the_file_row_order_that_edits_address`
(the invariant that makes the unsorted case safe).

---

## 20. The missing-bar check runs only on continuously-traded instruments

**Decision.** `quality.analyse` takes the dataset's `asset_class`. The missing-bar
detector runs for `crypto_spot` and `crypto_futures`, and reports itself as *not run*
for equities and equity futures, with the reason "no exchange session calendar is
modelled". The other four checks are unaffected.

**Why.** The detector's premise is that consecutive bars are one bar length apart. That
is true of a market that never closes and false of one that does: a daily equity series
contains no Saturday or Sunday bar, and an intraday one contains no bar overnight or on
a holiday.

Applied to an equity frame, the check therefore flags every session boundary. A two-year
daily series produces roughly two hundred "missing bar: 2 bar(s) absent" markers — each
one a *correct* observation about the data and a *worthless* one about the market. They
are indistinguishable in the marker list from a genuine hole, so the check does not
merely add noise; it destroys the signal it exists to provide, and it does so on exactly
the datasets where a real gap is most worth finding.

The fix that suggests itself — skip weekends, and maybe a list of US market holidays —
is worse than skipping the check. Weekends are only the largest closure: half-days,
ad-hoc closures, and non-US exchanges all produce the same false positive, and a
hardcoded holiday table silently becomes wrong the moment it is not maintained. Deriving
sessions from the data instead would make the check circular: a gap large enough to
matter is precisely what would be read as a market closure. Every one of these is
inventing a trading calendar, which §2's rule against silently inventing financial
methodology forbids.

**Consequence.** Real gaps in equity data go unreported by this check. They remain
visible as `duplicate_timestamp` and as discontinuities in the chart, and the report's
`skipped` list states plainly that the check did not run — the distinction the module's
own docstring is built around. A dataset with no `asset_class` is treated as
continuously traded, which is the pre-existing behaviour and the right default for the
crypto-first use case.

**Where.** `backend/app/services/quality.py` (`CONTINUOUS_MARKETS`, `analyse`),
`backend/app/services/datasets.py::quality`,
`backend/tests/test_quality.py::test_missing_bar_check_is_skipped_for_equity_instruments`.

---

## 21. A partly-failed dump walk is fatal, not stored with holes

**Decision.** When walking Binance's funding or metrics dumps over a date range, a file
that returns **404** is treated as a normal terminal result — there is no data for that
period, and no file will ever appear. Any *other* failure (network error, a truncated
download, an unparseable file) is retried, and if it still fails the whole fetch raises
`UpstreamError` naming how many files failed and the first error, even when the other
files arrived successfully.

**Why.** The two outcomes look identical in the stored dataset and are opposite in
meaning. A 404 is a fact about the market: no funding was charged in that window. A
failed download is a fact about the network, and storing the rest of the range turns it
into a hole in the data that is indistinguishable — both in the frame and in every
downstream statistic — from a period when the market genuinely had no data. The row
count will not look wrong; the gap will simply be absent from a series where absence is
already expected and already meaningful.

The earlier behaviour (fatal only when *every* file failed) made this worse rather than
better, because the failure that matters most is the rare one: a single failed daily
file among eight hundred produces a dataset that passes every check the application can
run, and the researcher has no way to learn that one day's data is missing.

**Consequence.** A long fetch now fails outright on a transient network error instead of
returning most of the range, and the user retries. That is the intended trade: a fetch
that visibly fails costs a retry, while a fetch that silently succeeds with a hole costs
a wrong result that is never detected. Retries before the failure absorb the common
transient case.

**Where.** `backend/app/services/binance_fetch.py` (`_collect_dump`, `_download_zip_csv`).

---

## 22. Column reordering is saved, as a set rather than as a sequence of moves

**Decision.** Dragging a column in the grid produces `reorder` column operations, one per
column, each carrying a 0-based place in the **final** schema. The backend applies them
all at once, after every drop, rename, and add, by sorting the columns on the requested
index rather than by performing a move per operation. A column the caller did not mention
sorts after every column that was placed, keeping its existing position among its peers.

**Why saved at all.** The grid has always *shown* the new order — `onColumnMoved` fed the
page's column model — but `columnOps` emitted only rename/drop/add, so Save discarded the
ordering without saying so. The user was looking at one schema and storing another, and
nothing in the interface could reveal the difference.

**Why a set rather than a sequence.** `columnOps` is a *diff* computed in the browser, not
a log of drags, and a diff has no canonical serialisation order. Applying `reorder(a,3)`
then `reorder(b,1)` gives a different result from the reverse, so a move-based backend
would quietly depend on the order the frontend happened to emit them in. Sorting on the
requested index is order-independent, which is a contract the next caller cannot break by
accident.

**Why positions are in the final schema.** The frontend can rename a column and drag it in
the same save, and it can drag a column it just added. Positioning by the name the column
*will* have, and running after the other three phases, means both cases are expressed by
the same operation instead of needing a second one.

**Consequence.** The reorder phase must run last, which is why `_apply_column_ops` now
orders its phases explicitly (drop → rename → add → reorder) instead of trusting the order
the operations arrive in. Hiding a column is *not* part of this: hidden columns keep their
existing slots in the model, because hiding is a view action and letting it also decide
the saved order would record a rearrangement the user never performed.

**Where.** `backend/app/services/datasets.py` (`_apply_column_ops`),
`frontend/src/pages/DataManagement.tsx` (`columnOps`),
`frontend/src/components/DataGrid.tsx` (`handleColumnMoved`).

---

## 23. A zone-less timestamp is resolved as UTC, by the chart and the quality checks alike

**Decision.** Both the candlestick endpoint and the quality checks read a timestamp that
carries no offset as UTC. Neither reads it as the machine's local time.

**Why.** These two layers used to disagree. `datasets.ohlcv` used
`pd.Timestamp(value).timestamp()`, which reads a naive value as the *local* time of
whatever machine the backend is on; `quality._time_seconds` used
`pd.to_datetime(series, utc=True)`, which reads it as UTC. For a fetch that difference is
invisible, because both providers emit tz-aware timestamps. An uploaded CSV is the case
that has no zone: its timestamp column is whatever text the file held, and it stays text.
There the two layers placed the same instant at different points on the axis, by the
machine's UTC offset — so an anomaly marker landed on the wrong candle, on every machine
that is not on UTC, and on no machine that is. A result that depends on which computer
ran it is exactly what this project's reproducibility rule rules out.

UTC is the choice because it is the one the rest of the schema already speaks and the one
the chart renders in. It is *not* a claim that the source data was recorded in UTC — see
§24.

**Consequence.** A row whose timestamp cannot be parsed is skipped rather than raising.
One broken cell in a large file used to turn the whole viewer into a 500; the quality
overlay is where a defective timestamp is meant to surface.

**Where.** `backend/app/services/datasets.py` (`_epoch_seconds`),
`backend/app/services/quality.py` (`_time_seconds`).

---

## 24. An uploaded CSV keeps its timestamp column as text

**Decision.** Uploads are parsed with `pd.read_csv` and nothing more. A timestamp column
is **not** converted to `datetime64`, so `start_timestamp` and `end_timestamp` are null
for every upload, and sorting that column in the grid is lexicographic.

**Why.** A CSV carries no timezone. Parsing the column would be a decision about what zone
those instants are in, and the only zones available to assume are UTC — which is a
fabrication if the file holds exchange-local times — or the machine's, which is not
reproducible. §23 already covers the reading that is *forced* (a zone-less value has to be
placed somewhere on a UTC axis), but that is a rendering decision on data whose metadata
still says "timezone: not stated". Promoting the column to a real timestamp would move the
same unverifiable assumption into the stored dataset, where it would look like a fact and
where any later resampling (§B) would build on it. The column names and dtypes are also
not ours to reinterpret: `open_time` in a user's file need not mean what `open_time` means
in a Binance frame.

**Consequence.** The Data Management metadata panel shows "Start —" for uploads, which is
honest but less useful than for a fetch. Lexicographic ordering is correct for the ISO-8601
spelling most tools emit and wrong for anything else, and the grid does not say which it
got. If a researcher needs the range, the correct move is to declare the zone — which is a
feature not yet built, and is the reason this is recorded as a limitation rather than
patched around.

**Where.** `backend/app/routers/datasets.py` (`upload_csv`),
`backend/app/storage/parquet_store.py` (`_detect_time_range`).

---

## 25. An upload's `asset_class` is a placeholder, and that is a known defect

**Decision (provisional).** `upload_csv` stores `asset_class = crypto_spot` for every
uploaded CSV. The value is a placeholder required by the storage layer, not a claim
about the file.

**Why it is not simply null.** `DatasetMeta.asset_class` is typed `AssetClass` and the
catalog column is `TEXT NOT NULL`, so a null cannot be stored without a schema change.
§20 already describes the *desired* behaviour — "a dataset with no `asset_class` is
treated as continuously traded" — and `quality.analyse` accepts `AssetClass | None` for
exactly that case, so the design anticipated this and the model does not permit it.

**What it costs.** Two things, both small but real:

1. An uploaded CSV displays `crypto_spot` in the Data Management metadata panel and in
   the explorer, which reads as a statement about data whose market is unknown.
2. It would enable the missing-bar detector on a frame whose market is unknown, if a
   timeframe were ever set for it. Today it cannot be — `timeframe` is also null for
   uploads, so `analyse` skips the check on the timeframe branch and reports that reason
   (§20) — so there is no false-marker output in practice. The exposure is latent, not
   live.

**Recommended fix (not applied).** Add an `UNKNOWN = "unknown"` member to `AssetClass`
and store that for uploads. This needs no migration, no schema-version bump, and no
catalog change — the column stays `NOT NULL` — and it makes the stored value honest. It
was left unapplied because it is a behaviour change to a model shared with the fetch
routes, and the test suite had not been executed at the time of writing; changing a
model's vocabulary on top of an unrun suite trades a cosmetic wrong label for an
untested behavioural one. `AssetClass.CRYPTO_SPOT` is never *read* as a claim anywhere
in the code, so nothing downstream depends on the placeholder being crypto.

**Where.** `backend/app/routers/datasets.py` (`upload_csv`),
`backend/app/models/dataset.py`, `backend/app/models/enums.py`.

---

## 26. Renames are applied simultaneously, not one at a time

**Decision.** `_apply_column_ops` collects every `rename` in a save into a single
mapping and applies it in one `DataFrame.rename` call, rather than applying one rename
per operation in sequence. Collisions are checked against the *final* set of names.

**Why.** The operations arriving at `/derive` are a diff — screen versus parent — not a
log of the user's clicks, so no canonical order exists among them. Applying renames
sequentially made the result depend on which order they happened to arrive in, and one
of those orders is a normal thing to do:

> Rename `b` to `c`, then rename `a` to `b`.

The grid permits it: `renameColumn` checks the new name against the columns *on screen*,
and by the second rename `b` is gone, so nothing conflicts. The backend refused it —
`a`→`b` ran while `b` was still occupied by the column whose own rename had not been
applied yet — and the user saw *"A column named 'b' already exists."* while looking at a
grid containing no column called `b`. A refusal the user cannot connect to anything
visible is worse than a wrong answer, because there is no way to work around it.

Check-the-target-first cannot fix this: neither target is occupied until the other
lands, so a per-operation check either rejects a valid pair or accepts a pair that
produces a duplicate. Validating the final name set handles both — two renames onto one
name are refused, and a shift along the schema is allowed.

**Consequence.** The rename phase is order-independent, joining the reorder phase (§22).
Drops still run first and adds last, because a drop can never depend on a rename and an
add may occupy a name the other two free.

**Where.** `backend/app/services/datasets.py` (`_apply_column_ops`),
`backend/tests/test_paging.py` (`test_two_renames_can_shift_names_along_in_one_save`,
`test_shifting_renames_does_not_depend_on_the_order_they_arrive_in`,
`test_two_renames_onto_the_same_name_are_refused`).

---

## 27. A backend `hint` is folded into the message the UI shows

**Decision.** When an error payload carries `details.hint`, `api/client.ts` appends it to
the `ApiError` message it throws, so the sentence reaches the screen. The structured
`details` object is still populated and unchanged.

**Why.** `QuantmetricsError.__init__` takes `**details`, and several rejections in
`_fit_to_dtype` and in the upload route put their *actionable* half there:

> `"1.5" is not a whole number, and column "close" holds whole numbers (int64).`
> — hint: *Storing it would round the value without saying so.*

Every display site in this application — the Data Management notice bar, the Data
Explorer's alerts, the grid's `onNotice`, the fetch page's error bar — renders
`error.message` and nothing else. None of them reads `details`. So each hint was
composed, serialised, sent over the wire, and stored on the exception without ever being
displayed once: the more useful half of the message was the half nobody could see.

**Consequence.** The message shown to the user is no longer byte-identical to the
backend's `message` field. Anything wanting the hint separately reads `details.hint`,
which is untouched; the appended copy is for display.

**Where.** `frontend/src/api/client.ts` (`request`).

---

## 28. Every fetch endpoint writes a run record, not just klines

**Decision.** `fetch_binance_funding`, `fetch_binance_metrics`, and `fetch_yahoo` each
call `IngestService.record_run` after a successful `register`, as `fetch_binance_klines`
already did.

**Why.** Only one of the four fetch handlers wrote to `fetch_runs`. The table — and
`GET /fetch/runs`, which reads it — therefore reported a history in which funding,
metrics, and Yahoo fetches had *never happened*, while klines fetches were all there.
That is the worst shape a gap can take: not an obviously empty log, but a plausible and
incomplete one. An absent entry in an audit log reads as "this did not run", not as
"this was not recorded", and the two are indistinguishable to whoever reads it.

The run is written *after* `register` commits, and carries the dataset id, for the
reason the klines handler records: a run marked succeeded before the dataset exists
would put a success in the log for a fetch that then failed to store anything.

**Not done here.** Failures are still not recorded — a fetch that raises leaves no row
at all. That is a real gap in the same table, and it is left open deliberately rather
than half-fixed: recording failures means wrapping each provider call, deciding what to
store for an exception that may carry a partial frame, and adding a `failed` status the
schema already allows but nothing writes. Doing it properly is a change to all four
handlers plus the catalog tests, and it should be made when the suite can be run.

**Where.** `backend/app/routers/fetch.py`.

---

## 29. Test helpers are imported as `tests.conftest`, not `conftest`

**Decision.** Every test module imports the shared helpers with
`from tests.conftest import ...`. `backend/tests/__init__.py` stays, and no
`pythonpath` setting is added.

**Why.** Seven of the eight test files used a bare `from conftest import ...`. That
import only resolves if `backend/tests` is on `sys.path`, and with a `tests/__init__.py`
present it is not:

- pytest's default `prepend` import mode resolves the *package* root of a module by
  walking up while `__init__.py` exists. For `backend/tests/test_paging.py` that walk
  stops at `backend/`, so `backend/` is what gets inserted and the module is imported as
  `tests.test_paging`.
- `backend/tests` is therefore never inserted, and there is no `backend/conftest.py` for
  a bare `conftest` to find.
- Running `python -m pytest` puts the *current directory* on `sys.path`, which is
  `backend/` — the same directory, not the one that would help.

The result is `ModuleNotFoundError: No module named 'conftest'` at **collection** time,
in seven files at once. Not a failing assertion: the suite never starts, so `pytest`
would have reported nothing about the behaviour it exists to check. That is the failure
mode this project can least afford, because the entire verification story rests on the
suite running.

The fix is the import, not the layout. `backend/tests/` is a real package directory, so
`tests.conftest` names it unambiguously and imports it once — pytest has already loaded
it under that name by the time a test module is imported, so the two agree on one module
object.

**The alternative that was rejected.** `pythonpath = ["tests"]` in `pyproject.toml`
would also make the bare import resolve, in one edit rather than seven. It was rejected
because it leaves `conftest` importable under two names: pytest's `tests.conftest` and a
second, independent `conftest` module created by the test's own import. The helpers here
are pure functions and the duplication is currently harmless, which is exactly what makes
it a trap — the day one of them gains module-level state, the fixtures and the tests
would be reading different copies and nothing would say so.

Removing `tests/__init__.py` would work too, and is the more conventional pytest layout,
but it makes the test directory's importability depend on a *missing* file. The explicit
`tests.conftest` spelling makes it depend on a present one.

**Consequence.** Ruff's import sorter is satisfied: `tests` resolves to a directory under
`backend/`, so it is first-party and sorts after `app.*`, which is where these imports
already sat. The blank line that separated them was removed, since it would have split
one section in two.

**Confirmed.** Originally recorded as unverified reasoning from pytest's documented
`prepend` import behaviour and `_pytest.pathlib.resolve_package_path`. The suite has since
been run and all eight modules collect and pass, so the spelling is correct in practice
and not only on paper.

**Where.** `backend/tests/test_api.py`, `test_cell_edits.py`, `test_leakage_guards.py`,
`test_lineage.py`, `test_paging.py`, `test_quality.py`, `test_storage.py`.

---

## Part B — Data preparation

The specification lists the cleaning and merge options as short phrases in a UI mock. Each
one below is a place where the phrase admits more than one implementation and the choice
changes the numbers in the output dataset.

### 30. Two timestamp-duplicate policies, not one dedupe switch

**Decision.** Duplicate handling offers four members: `keep`, `drop_exact`,
`keep_first_by_time`, `keep_last_by_time`.

**Why.** The specification says only "remove exact duplicate rows", which is *not* the same
defect as a repeated timestamp. Two rows can carry the same `open_time` and differ in every
other column — a provider correction that was appended rather than substituted, which is
common in Binance dumps around re-published bars. `drop_exact` leaves that pair intact, and
the result is a frame with two bars at one instant: every return, lag, and rolling window
computed on it silently pairs the *later* row with the *earlier* one. Both defects are
offered, separately, because a user who wants one does not necessarily want the other.

**Consequence.** `keep_first_by_time` / `keep_last_by_time` are order-dependent, which is
why the sort step precedes them (see §34).

**Where.** `backend/app/models/enums.py` (`DuplicatePolicy`), `services/cleaning.py`
(`_handle_duplicates`).

---

### 31. Forward and backward fill are separate choices

**Decision.** `ffill` and `bfill` are two members of `MissingValuePolicy`, not one
"fill gaps" option.

**Why.** They differ in exactly the way that matters to a quantitative application. A
forward fill carries the last *past* observation into a gap. A backward fill carries the
next *future* observation backwards into it — on a price series that is a look-ahead, and it
is invisible in the output frame. A single checkbox would make the two indistinguishable at
the point of choosing, and the leak would only ever show up as an unexplained edge in a
backtest. The UI labels the backward option with that consequence on screen.

**Consequence.** `interpolate_time` is a third distinct choice: it draws a line between the
known values on either side and is the only policy that uses *both* neighbours. It is
restricted to `limit_area="inside"`, so it never extrapolates beyond the first or last
observation — pandas' default would invent values outside the data instead of between it.

**Where.** `backend/app/models/enums.py` (`MissingValuePolicy`), `services/cleaning.py`
(`_handle_missing`, `_interpolate`).

---

### 32. Alignment is as-of only: `backward` and `exact`

**Decision.** `AlignMethod` has two members. A forward or nearest match is not offered.

**Why.** Merging a lower-frequency series (funding at 8h, metrics at 5m) onto a klines
timeline must not read the future. `backward` takes the most recent value *at or before*
each bar — the only direction that cannot. `forward` stamps a bar with a value that had not
been published yet at that bar's open; `nearest` does the same whenever the next
observation happens to be closer than the last. Neither is visible in the merged frame, and
both would surface only as an unexplained edge in a backtest. `exact` is offered alongside
because a user merging two series already on the same grid legitimately wants "no
observation, no row".

**Consequence.** The staleness cap is expressed in seconds and is a separate input, so a
user can bound how far back a carried-forward value may reach rather than accepting an
unbounded as-of join.

**Where.** `backend/app/models/enums.py` (`AlignMethod`, `MergeHow`), `services/merge.py`.

---

### 33. Resampling keeps levels as `last`, and labels by interval open

**Decision.** OHLCV columns aggregate by their conventional rules (first/max/min/last/sum);
every other column aggregates as `last`. The resample uses `label="left", closed="left"`.

**Why.** Open interest, a funding rate, and a long/short ratio are *levels* observed at an
instant. Averaging them over an hour reports a value that was never observed at any time,
and `mean` is pandas' own default — the mistake is available by doing nothing. `label="left",
closed="left"` matches how Binance and this application stamp bars: the timestamp is the
opening instant of the interval, and pandas' default would label each bar with its close,
shifting every timestamp by one bar and every subsequent join by the same amount.

**Consequence.** An interval with no source rows is dropped rather than emitted as a bar.
The test is a per-bin row count, not `dropna(how="all")`: `sum` over an empty group is `0`
rather than `NaN`, so a volume column keeps a fabricated row alive with no prices and a zero
volume. The count is also the right strength in the other direction — a bin whose rows are
all empty is a real hole and must stay visible to the missing-value detector.

**Where.** `backend/app/services/merge.py` (`resample_frame`).

---

### 34. The cleaning step order is fixed, and the plan is applied in it

**Decision.** `sort` → `duplicates` → `missing` → `outliers`, always, whenever the
corresponding policy is not `keep`.

**Why.** The order is observable in the output, so leaving it to the request would make two
requests with identical options and different key order produce different datasets. Each
position is forced by the meaning of the steps:

- `sort` first, because `keep_first_by_time` and `keep_last_by_time` mean "first and last
  *in time*". On an unsorted frame those words describe the file's row order, so the sort is
  a precondition of the dedupe, not a cosmetic tidy-up.
- `duplicates` before `missing`, because a row that is about to be deleted should not first
  be filled — the fill would consume and then discard a neighbouring observation.
- `outliers` last, so a row dropped for being an outlier is not first interpolated into its
  neighbours and does not drag a fabricated value into the fence statistics.

**Consequence.** A step that was not applied is not reported: an empty `steps` list means
"nothing was changed", not "nothing was considered".

**Where.** `backend/app/services/cleaning.py` (`clean_frame`).

---

### 35. Outlier fences are computed over the whole series

**Decision.** The winsorize/remove quantile fence and the IQR fence are computed from the
entire column, not from a trailing window ending at each observation.

**Why.** This is a deliberate and limited exception to the no-look-ahead rule, and it is
recorded as one. A trailing-window fence answers "is this value extreme *given what was
known at the time*", which is the backtest-correct question. A whole-sample fence answers
"is this value extreme *relative to the dataset*", which is the data-quality question this
step is asked. The specification's outlier options are a cleaning step applied to a fixed
historical dataset that the user is about to inspect; a rolling fence would let a real
data-feed error survive simply because it persisted for longer than the window.

**Consequence.** A dataset cleaned this way carries a threshold that depends on rows which
did not exist at the time the earliest row was recorded. The fence actually applied is
recorded per column in the step report **and** in the child dataset's provenance, so the
value is auditable and the assumption is not silent.

**Where.** `backend/app/services/cleaning.py` (`_fences`).

---

### 36. A cleaned or merged dataset lands in `PROCESSED`

**Decision.** Both Part B steps write their output as a new dataset in
`DatasetCategory.PROCESSED`, with `parent_dataset_id` set.

**Why.** Neither is a fetch and neither is an upload, so `raw_data` and `uploaded_data` are
both wrong by their own definitions — and both of those categories are write-once, which a
step that rewrites rows cannot honour. `PROCESSED` is the category that exists for derived
frames, and the lineage edge is what makes the step reproducible: the parent's checksum is
untouched and the child records what was done to it.

**Consequence.** "Any modification creates a new dataset version and is recorded in
lineage" holds without exception. Nothing in Part B overwrites its input.

**Where.** `backend/app/routers/datasets.py`, `services/datasets.py`.

---

### 37. A naive fetch bound means UTC; an offset bound is converted

**Decision.** `_as_utc` reads a bound with `tzinfo=None` as UTC, and converts a bound that
carries an offset with `astimezone(UTC)` rather than discarding the offset. It is applied
before the dump files are listed as well as before the rows are filtered.

**Why.** The dump files carry UTC timestamps, and pandas does **not** coerce a tz-aware
column against a naive bound — it raises `TypeError: Invalid comparison between
dtype=datetime64[ns, UTC] and datetime`. Neither `fetch_funding` nor `fetch_metrics` was
covered by a test, so both endpoints answered HTTP 500 for *every* request that reached
them; the missing-timestamp path was only ever exercised through the parser, never through
the filter.

Reading a naive bound as local time was rejected: Binance klines are UTC and every stored
timestamp in this application is UTC, so treating a naive bound as the machine's local zone
would shift the requested range by the host's offset, and the resulting dataset would look
plausible. Discarding an offset instead of converting it was rejected too, for the same
reason in reverse — `02:00+02:00` is midnight UTC, and reading it as 02:00 UTC silently
narrows the range by two hours at each end.

Ordering matters for the same reason: a bound near midnight can fall on the previous UTC
day, so the conversion has to happen before file selection or the wrong monthly dump is
listed and the result is an empty frame with no error.

**Consequence.** A test asserts both the recorded bound and the surviving rows for an
offset-aware request, so the "convert, don't discard" behaviour fails loudly if it regresses.

**Where.** `backend/app/services/binance_fetch.py` (`_as_utc`), `tests/test_binance_dump_ranges.py`.

---

### 38. `calc_time` is cast to a number before a `unit` is applied

**Decision.** The funding dump's `calc_time` is read with `pd.to_numeric(..., errors="coerce")`
before `pd.to_datetime(..., unit="ms", utc=True)`; rows that coerce to `NaT` are dropped.

**Why.** The CSV supplies `calc_time` as text. `to_datetime` with a `unit` on *strings* is
deprecated, and the deprecation is raised as an error by `filterwarnings`. Under pandas 3
the string would instead be parsed as a datetime string, at which point `"1704067200000"`
becomes `NaT` and every funding row is dropped **without an error** — a silently empty
dataset from a fetch that reports success.

**Where.** `backend/app/services/binance_fetch.py` (`fetch_funding.assemble`).

---

### 39. A quality finding is not a skipped check

**Decision.** `QualityResponse` carries three lists, and they make three different claims:
`counts` (a present count of 0 = "checked, nothing found"; an absent count = "not checked"),
`skipped` ("a check could not run"), and the new `notes` ("a check ran and this is what it
found").

**Why.** The invalid-numeric finding — *"Numeric values: level (2 of 12)"* — was being put
in `skipped`, which the UI renders under "not checked". That reports the check as not
having run at the exact moment it produced a number, and it is the more dangerous direction
of the two errors: it teaches the reader to distrust a panel that is working correctly.

**Consequence.** `notes` is a new field on the wire. The frontend renders it under "also
found:", separately from "not checked:". `invalid_dtype` remains in `skipped` because a
column this application cannot build a feature from genuinely means the *other* checks on
that column could not run.

**Where.** `backend/app/services/quality.py`, `frontend/src/pages/RawDataView.tsx`,
`frontend/src/pages/Cleaning.tsx`.

---

### 40. `invalid_dtype` and `invalid_numeric` count columns, not values

**Decision.** Both counts are per-column, and a mostly-text column is not invalid.

**Why.** "Invalid data type" for a canonical Parquet store cannot mean "a corrupt value" —
Parquet has no corrupt scalar. It means a *valid* type this application cannot build a
feature from, which is a property of the column, so counting values would report a
thousand-row binary column as a thousand defects.

`invalid_numeric` needs the mirror-image guard. A column of venue labels is mostly
non-numeric, which is what keeps it out of the count; flagging it would make the check
useless on any dataset carrying a symbol or notes column. The count fires on a numeric
column contaminated by a few non-numbers — the CSV-upload case it exists for.

**Where.** `backend/app/services/quality.py`, `tests/test_quality.py`.

---

### 41. A resample-only run is allowed; a no-op request is not

**Decision.** `DatasetService.merge` refuses a request only when it carries *both* an
empty join list and no target timeframe. A timeframe with no joins resamples the base and
writes a child named `… (resampled)` with `operation="resample"`.

**Why.** The guard used to be `if not joins: raise ValidationFailure("Nothing to merge:
no dataset was selected.")`, and the frontend mirrored it with
`canRun = joins.length > 0 && …`. Together they made **resampling impossible**: the first
half of what the step is named for could not be performed at all, in either layer.

That is not a cosmetic omission. Changing the observation frequency — 1-minute bars to
1-hour bars — is a data-preparation decision in its own right, it is the one operation in
this step that needs no second dataset, and it is the operation the page's most prominent
control (the timeframe selector) exists to request. Requiring a join to reach it forced a
user who wanted only to resample to attach an unrelated dataset to the request, which
would then have written that dataset's columns into the output.

The first-run screen made the defect visible rather than hiding it: the join list starts
empty, so the page opened on a timeframe selector above two permanently dead buttons,
with nothing on screen to say why.

**Consequence.** An empty join list is no longer a defect by itself, so the "nothing to
do" rejection moved to the pair. A resample-only child omits `merge_inputs` and
`merge_methods` from its provenance rather than writing them as empty strings — an empty
`merge_inputs` reads as a merge whose inputs were lost, which is the opposite of what
happened. Two tests cover the newly-legal path, and `test_merging_nothing_is_refused`
now asserts the narrowed guard.

**Where.** `backend/app/services/datasets.py` (`merge`),
`frontend/src/pages/ResampleMerge.tsx` (`problems`, `canRun`), `tests/test_merge.py`.

---

## Part C — Preprocessing

Part C is where the pipeline acquires its most dangerous property. Every choice in it can
introduce look-ahead that is *invisible in the resulting frame*: a feature computed over
the whole series, a split whose boundary lets a label horizon straddle it, a scaler fitted
before the cut. Each produces a dataset that looks correct and backtests too well. So the
whole part is written around one invariant, held by test rather than by intent — **every
transformation is causal and point-in-time**: for any row *t*, its value depends only on
rows at or before *t*, and computing the same series over a shorter history leaves every
row it shares unchanged.

This pass builds **C5 (feature engineering)** to completion. C6 and C7 are fixed at the
interface level only (§56).

### 42. The C5/C6/C7 badges were corrected once, and are frozen from here

**Decision.** The registry's third section now reads C5 = Features Engineering, C6 =
Chronological Splits, C7 = Scaling. The registry ids (`feature_selection`,
`chronological_splits`, `scaling`) are unchanged.

**Why.** When Part C was still unspecified, the three entries were given guessed labels —
C6 "Scaling", C7 "Chronological Splits", C5 "Feature Selection" — and the specification
turned out to number them the other way round. The registry's own rule is that badges never
drift, because they are how documentation and conversation refer to a page ("C5 is
broken"); that rule exists to protect codes that something already depends on. Nothing did:
none of the three had ever rendered, no decision entry, no test, and no driver named them.
Leaving the wrong labels in place to honour a rule about stability would have made every
future reference to "C6" mean the wrong page.

**Consequence.** This is the only badge to have moved, it is recorded in the file itself
(`layout/registry.ts`), and the codes are frozen now that the specification names them.
Dropping the labels entirely was the alternative and was rejected: the sidebar's numbering
is the only thing tying the three unbuilt pages to the specification's items.

**Where.** `frontend/src/layout/registry.ts`.

---

### 43. `weighted_price` is one series with a disclosed resolution, not a formula

**Decision.** `PriceSource.weighted` resolves per dataset: the exact bar VWAP
`quote_volume / volume` when both columns exist, otherwise the typical price
`(high + low + close) / 3`. Which one was used, and why, is reported in the run's
`price_source.definition`/`basis`, written to the saved dataset's provenance as
`feature_price_source`, and rendered in the report panel.

**Why.** Nothing in the specification or in a Binance kline is called "weighted price".
Both readings are defensible and they are not the same number: the first is the volume-
weighted average price actually traded in the bar, the second is a midpoint estimate that
ignores volume entirely. Picking one silently would make two datasets with the same
`price_source: "weighted"` provenance incomparable — the difference would appear only as
a slightly different column, with nothing anywhere saying why. The two are also not equally
good, so the fallback is stated rather than hidden.

**Consequence.** The `basis` string distinguishes the branches in the report; on a dataset
without `quote_volume`, the run's notes also say so in a sentence. A user who needs the
exact price must use a dataset that carries quote volume.

**Where.** `backend/app/services/features.py` (`_price_series`, the report's
`price_source`), `frontend/src/api/types.ts` (`PRICE_SOURCE_OPTIONS`).

---

### 44. A time-denominated window must divide exactly into the dataset's bar length

**Decision.** A duration window (`"4h"`, `"7d"`) is offered only when it is an exact
non-zero multiple of the dataset's timeframe. A window that resolves to fewer than two bars
is also refused for the features whose arithmetic needs two. Every refused tick is returned
with its reason and rendered disabled rather than hidden.

**Why.** The specification asks for "15m, 1h, 4h, 1d, 7d, 21d" windows and separately
notes that a daily dataset should have the intraday ones off. A rule is needed to make
"intraday" precise, and exact divisibility is the only one that does not silently round:
15 minutes on an hourly dataset is a quarter of a bar, and rounding it to one bar turns a
quarter-hour return into an hourly one — a 4× error in the return and a 16× error in its
variance, with nothing in the column name to show it.

**Consequence.** `realized_vol` carries a second, separate guard: it needs at least two
squared returns, so a window of one bar is refused with "at least 2". On a 1h dataset a
"1h realized volatility" would otherwise be `|r|` — an absolute return wearing a
volatility's name, and one that a model would use as though it were a second moment.

**Where.** `backend/app/services/feature_catalog.py` (`duration_bars`, `availability`),
`tests/test_features_catalog.py` (`test_a_daily_dataset_cannot_support_an_intraday_window`,
`test_realized_volatility_needs_two_bars`).

---

### 45. `annualized_volatility` is refused when no timeframe is declared

**Decision.** The annualisation factor is derived from the dataset's declared timeframe
and asset class (365 days for crypto, 252 trading days for equity, and the session length
for intraday equity bars), is stated in the info icon, and the feature is **unavailable**
on a dataset with no declared timeframe.

**Why.** Forgetting to declare a timeframe would otherwise not produce a missing column —
it would produce a wrong one. Annualising an hourly series with the daily factor gives a
volatility that is √24 ≈ 4.9× too small; the crypto/equity mix-up is √(365/252) ≈ 1.20. A
silent multiplicative error in a volatility is indistinguishable from a market that was
quieter than it was, which is exactly the kind of error that survives to a backtest.

**Consequence.** The factor and its source are reported per dataset, so two datasets can be
compared knowing both were annualised the same way. Uploaded CSVs declare no timeframe and
therefore cannot use this feature — stated in the catalogue's reason, and in the run's
notes (§54).

**Where.** `backend/app/services/feature_catalog.py` (`NEEDS_TIMEFRAME`,
`annualisation_factor`), `features.py` (`annualized_vol` computer).

---

### 46. Conventional defaults are baked into the column name, not hidden in the code

**Decision.** Where the specification gives no parameter, a conventional default is chosen
and named in the output column: Bollinger 20/2 → `bb_upper_20_2`, MACD 12/26/9 →
`macd_signal_12_26_9`, Williams %R 14 → `williams_r_14`, Aroon 14 → `aroon_up_14`,
`obv_slope_10`, `sma_20_slope_5`. The formula string in the info icon states the same
constants.

**Why.** A parameter has to be chosen for these; the specification gives none. The choice is
unavoidable, the invisible choice is not. `bb_upper` names a column whose value depends on
two numbers the reader cannot see, and a later reader (or the same person in six months)
cannot tell whether it was 20/2 or 14/1.5 without reading the implementation. A test
enforces the rule: any catalogue entry with a parameter must name that parameter in at
least one of its column templates.

**Consequence.** The name is longer and the schema is self-describing. `ADX(1)` is allowed
— Wilder smoothing at period 1 is no smoothing, so ADX(1) equals DX — and named `adx_1` with
a caution on the option rather than refused, because it is degenerate but defined.

Naming the constants forced the *warm-up* to be generalised too, and that exposed two defects
on either side of the same declaration — neither one visible in a column name. ADX smooths
twice: DI± need `n` bars before DX exists, and ADX is a further `n` bars of smoothing on top of
DX, so its first defined row is `2n − 1`. It was declared as `n + 13`, which is `2n − 1` at
exactly `n = 14` — the period the formula was first written for — and wrong at every other tick
the specification lists (ADX(1) is defined at row 1, not row 14; ADX(21) at row 41, not row
34). That is the claim side, found by reading the declaration against the implementation while
the gates could not be run. The value side was worse and surfaced only when the tests were
first executed: `_dx` took its window from a *default* argument of 14, and `_adx` called it
without passing `args.bars`, so every tick computed `Wilder(DX(14), n)`. `adx_14` was therefore
right by arithmetic coincidence — the coincidence that hid the defect — while `adx_1`, `adx_7`
and `adx_21` carried a 14-period DX underneath a column name, a formula and a warm-up that all
said otherwise, and `adx_1` was not the raw DX its caution promises. What the two together
produced was not a wrong number in a column nobody reads: it was a wrong *claim about where
the numbers start*, which `usable_from` reports to the user and the B3 missing-value policy
acts on, plus four columns whose stated period was not their period. `_dx` now has no default
window, so the omission cannot recur silently.

**Where.** `backend/app/services/feature_catalog.py` (`adx`),
`backend/app/services/features.py` (`_dx`, `_adx`),
`tests/test_features_catalog.py` (`test_the_column_name_carries_its_constants`),
`tests/test_features_compute.py` (`test_the_warm_up_is_exactly_as_declared`).

---

### 47. Features only ever add columns; rows are never dropped

**Decision.** `rows_before == rows_after` for every feature run. The leading rows a window
has not filled are left **empty**, not truncated, not filled, and not zeroed.

**Why.** Two reasons, and they point the same way. Deleting the warm-up rows would make
this step a second, hidden way to lose the timeline — B3's missing-value policy already
owns that decision, is recorded in provenance, and is reported to the user; a silent second
deleter would make a dataset's row count depend on which features happened to be selected.
And filling the gap would fabricate observations: a 20-bar average's first value does not
exist, and `0` is not a neutral stand-in for it on any price-derived column — it is a
−100% return.

**Consequence.** The gap is visible in the grid as empty cells and is visible to B3's
missing-value check, which is where a user can decide what to do about it. The report
states the exact warm-up (`warmup_bars`, the longest of the selections, not their sum) and
the timestamp of the first fully-warmed row (`usable_from`). A test asserts the empty cells
at the right indices, including the boundary: row 18 empty, row 19 present for a 20-bar
window.

**Where.** `backend/app/services/features.py`, `tests/test_features_compute.py`
(`test_rows_are_never_dropped`, `test_the_warm_up_is_exactly_as_declared`),
`tests/test_features_api.py` (`test_the_warm_up_rows_are_visible_as_empty_in_the_stored_frame`).

---

### 48. `month_end` is a calendar fact, not "the last bar of the month"

**Decision.** `month_end` is 1 when the bar's own calendar date is the last day of its
month. `days_to_month_end` and `days_to_quarter_end` are offered alongside.

**Why.** "The last bar of the month" — the natural reading of the specification's phrase —
cannot be known at bar *t*: it requires seeing that no further bar arrives, which is
information from the future. A column built that way is a look-ahead that survives every
causal test applied to the feature itself, because the leak is not in the calculation, it
is in the definition. The calendar form is knowable at *t* by definition.

**Consequence.** On a daily series the two agree except when the month's last day is not
traded; on an intraday series the calendar form marks every bar of the last day, which is
why `days_to_month_end` (the more useful causal form) is offered as a separate column.

**Where.** `backend/app/services/feature_catalog.py`, `features.py` (`month_end` computer
against `pd.Period` bounds), `tests/test_features_compute.py`
(`test_month_end_follows_the_calendar_not_the_last_observed_bar`).

---

### 49. Features the specification lists twice are emitted once

**Decision.** Aroon appears once, as `aroon_up_14` / `aroon_down_14` / `aroon_osc_14` —
not also as `AroonU_14` / `AroonD_14`. Vortex is one checkbox producing two columns
(`vortex_plus_n`, `vortex_minus_n`). EVWMA takes a window parameter rather than a baked-in
period.

**Why.** The specification lists Aroon twice under two spellings, and shipping both would
put two identical columns in one dataset under two names — which the collinearity scan
would then report as a genuine finding, and which a model would treat as two features. A
single column cannot represent the Vortex indicator, whose whole content is the
relationship between VI⁺ and VI⁻. EVWMA's period *is* its character; baking one would be
a bigger invention than asking.

**Consequence.** EVWMA carries a caution recording that it is a crypto adaptation of an
equity indicator — the original weighs by a share count that has no counterpart in a crypto
dataset — so the column is labelled for what it is.

**Where.** `backend/app/services/feature_catalog.py` (`aroon`, `vortex`, `evwma`),
`tests/test_features_catalog.py` (`test_no_two_features_claim_the_same_column`).

---

### 50. Exogenous features are carried, never computed — and the collision check covers generated columns only

**Decision.** `last_funding_rate`, `sum_open_interest`, `settlement`, `yield_curve_slope`
and the rest of the exogenous group are `FeatureKind.CARRIED`: selecting one keeps the
column, computing nothing. They are available only when a column of that name exists in the
dataset and is not listed in `unpopulated_fields`; otherwise the entry is disabled with
"merge one in first". The output-column collision check is scoped to **generated** columns.

**Why (carried).** These are outputs of A1 or of a B4 merge. Nothing is computed, so nothing
can be invented — which is exactly what the specification asks for, and the reason the
group is kept structurally separate from the rest. A Yahoo futures frame carries a
`settlement` column that is empty in every row; offering it because the column exists would
produce a feature that is entirely null, so `unpopulated_fields` is consulted too.

**Why (the collision fix, found by reading this module against its own test).** The check
originally ran over every resolved item. A carried column is *by definition* one that is
already in the frame, so this refused `volume`, `quote_volume`, `num_trades`,
`taker_buy_volume` and `taker_buy_quote_volume` — five catalogue entries that could never be
selected on the datasets that have them, for a reason the user would read as "this column
already exists", which is the whole point of carrying it. The check now runs over generated
names only: a *generated* name that collides is refused (auto-renaming would make the output
schema depend on the input schema in a way the report does not explain), and a carried name
is not a collision because it is the same column.

**Consequence.** A carried column also passes through byte-identical, which is asserted
rather than assumed. Carrying a merged column is only as-of-safe as the B4 merge that
produced it — an `exact` or `backward` match is safe, and B4 does not offer any other — so
the merge step's own decision (§32) is what protects this one.

**Where.** `backend/app/services/features.py` (`generated_names`, `carried`),
`tests/test_features_compute.py`, `tests/test_features_api.py`.

---

### 51. The catalogue is a module, and the frontend publishes no formula of its own

**Decision.** Every formula, parameter list, source requirement and availability rule lives
in `services/feature_catalog.py`. The engine reads it to compute, `GET /api/features/catalog`
serves it, and the ℹ️ icon renders the catalogue's `formula` and `detail` strings verbatim.
There is **no** `GET /api/datasets/{id}/features`: the catalogue is served for a dataset as
a query parameter of one endpoint.

**Why.** Three consumers of one description is what stops the description drifting from the
code. If the frontend restated "Bollinger = SMA(20) ± 2σ", that sentence and the
implementation would be two independent statements of the same mathematics, and the user
reading the info icon would have no way to tell which one the column came from. The page
also never re-derives availability: whether a 4h window is usable depends on the dataset's
timeframe, the divisibility rule (§44), and the feature's minimum, and a browser that
computed that itself would disagree with the server eventually — showing a tick that fails,
or hiding one that works.

**Consequence.** A catalogue integrity test asserts that no entry can exist without a
formula, a unique id, a unique column template, and a computer in the engine's table; and
that every name in `needs` is in the known column vocabulary. The engine's computer table is
asserted to have no orphans either, so a feature cannot be computable without being offered
or offered without being computable.

**Where.** `backend/app/services/feature_catalog.py`, `routers/features.py`,
`tests/test_features_catalog.py`, `frontend/src/components/InfoTip.tsx`.

---

### 52. A selection is recorded as a hash, and the price source as provenance

**Decision.** Every feature run reports a `selection_hash` — a SHA-256 over the sorted
selections, the resolved windows, the price source and `keep_only_selected` — and writes it,
with `feature_price_source`, `operation: "features"` and `derived_from`, into the child
dataset's provenance.

**Why.** A feature dataset is only reproducible if the exact request that produced it is
recoverable. The selections visible in the report are a rendering; the hash is the artefact
that can be compared. Sorted before hashing, because two runs that ask for the same features
in a different order produce identical frames and must therefore produce identical hashes —
otherwise the hash would be an ordering check wearing a reproducibility label.

**Consequence.** The report panel shows the hash, the resolved price source and its
definition, and the per-feature warm-up. `keep_only_selected` is part of the hash because it
changes the output schema: a run that dropped the source columns is a different dataset from
one that kept them, even though every generated column is identical.

**Where.** `backend/app/services/features.py` (`_selection_hash`, the report dict),
`services/datasets.py` (`features`, `extra_provenance`), `tests/test_features_api.py`.

---

### 53. Availability is answered from the data where the data can answer it

**Decision.** `DatasetContext` carries an `intraday: bool | None` flag, computed by
`quality.has_intraday_pattern` from the timestamps: `True` when the bars fall at more than
one time of day, `False` when they do not, `None` when there is no usable time column. The
`hour_of_day` guard reads that flag first and falls back to the declared timeframe only when
it is `None`.

**Why.** This started as a guard on the declared timeframe alone, which refused
`hour_of_day` on *every* uploaded CSV — including an hourly one whose timestamps plainly
show twenty-four distinct clock times. The column is perfectly well defined in such a file;
greying it out for a missing label is a menu option disabled for a reason the user cannot
see and cannot fix. Asking the timestamps instead makes the refusal be about the data: a
daily series is refused because every bar really is at the same time of day (`False`), not
because a field is empty. The same reasoning removed the timeframe guard from `month_end`,
`days_to_month_end` and `days_to_quarter_end`, which are pure functions of the date and need
no bar length at all — `_needs_calendar` was deleted rather than narrowed (§48).

**Consequence.** A declared timeframe that disagrees with the timestamps does not win: if
the bars all sit at one clock time, a declared `1h` does not make them intraday, because the
column would still be constant. Three tests cover the three branches, and the HTTP test
proves the whole path end to end on real uploads.

**Where.** `backend/app/services/quality.py` (`has_intraday_pattern`),
`services/feature_catalog.py` (`_needs_intraday`), `services/datasets.py`
(`_feature_context`), `tests/test_features_catalog.py`, `tests/test_features_api.py`.

---

### 54. Two known defects are surfaced in the report rather than hidden

**Decision.** A feature run on an uploaded dataset emits a note saying the asset class is a
placeholder, and a run including a recursive feature emits a note saying so.

**Why.** An upload is stored with `asset_class = crypto_spot` as a placeholder (§25). Every
asset-class gate in the catalogue — which crypto-only columns are offered, what the
annualisation factor is — is therefore unreliable on an uploaded file, and there is no way
for the page to detect that from the metadata. Silently gating on a value known to be wrong
is worse than gating on it and saying so. The recursive note covers EMA, RSI, ADX and the
Wilder family: they carry state from bar to bar, so an empty cell in the source resumes the
recursion from the last real observation, and the affected rows are left empty rather than
bridged. Both are properties of the output that no column name reveals.

**Consequence.** The notes appear in the report panel's "what this did to the data" block
alongside the price-source note. They are notes, not warnings: nothing is wrong, and both
are true of every run they appear on.

**Where.** `backend/app/services/features.py` (the `notes` list), `tests/test_features_api.py`.

---

### 55. The specification's column names are mapped to this project's canonical ones

**Decision.** Where the specification's exogenous names differ from the columns A1 actually
writes, the catalogue uses this project's spelling and the difference is recorded here:
`quote_asset_volume` → `quote_volume`, `taker_buy_quote_asset_volume` →
`taker_buy_quote_volume`, `number_of_trades` / `trade_count` → `num_trades`,
`open_interest` → `sum_open_interest`.

**Why.** These are not synonyms to be reconciled at read time; they are the same quantity
under two names, and the right-hand names are the ones on disk — they come from the
`data.binance.vision` dump headers (§4), which is where the canonical schema was fixed. A
catalogue that declared `quote_asset_volume` would find it missing on every dataset and
disable the feature with "this dataset has no quote_asset_volume", which is true and
useless.

**Consequence.** A test asserts every name in every entry's `needs` is in
`SOURCE_COLUMNS`, so a specification name that was not mapped fails a test rather than
silently disabling a feature. `open_interest` is mapped to `sum_open_interest` because that
is the summed open interest Binance's metrics endpoint reports; the name carries the
aggregation rather than implying a per-venue figure the dataset does not have.

**Where.** `backend/app/services/feature_catalog.py` (`SOURCE_COLUMNS`),
`tests/test_features_catalog.py` (`test_needs_names_real_columns`).

---

### 56. C6 and C7: interfaces fixed now, not built

**Decision.** C6 (chronological splits) will save a split as an **`outputs` plan table**
`{fold, role, start, end, bars}` with lineage to the source dataset, never as a mutation of
the price data. C7's transformations will be fitted **on the training span only** —
whole-series fitting is not offered at all — and will refuse to run until a split plan is
chosen, except for the causal rolling z-score, which needs no fit and no split. The fitted
parameters are saved as a second `outputs` table so any prediction can be inverse-transformed
from the lineage alone.

**Why.** A split is a description of a dataset, not a change to it: storing it as rows in an
`outputs` table leaves the price dataset's checksum untouched and makes the split itself
inspectable and reusable, which is what C7 then reads. Fitting a scaler over the whole series
and then splitting is the single most common way a correct-looking pipeline leaks: the
training rows carry the test span's mean and standard deviation, so the model has already
seen the future before it is trained. Fitting on train only makes the leak impossible rather
than merely discouraged, and saving the parameters is what lets the transform be undone —
without them, a scaled column cannot be turned back into a price.

**Consequence.** Two specification ambiguities are recorded for the C6 pass: the spec shows
both a fold count *and* per-fold train/test dates, which cannot both be independent inputs
(the count and the scheme generate the folds; the dates bound the span they are drawn from),
and it bundles purge and embargo under one heading when they are different operations.
Purging needs a label horizon, which the spec never asks for, so a `label_horizon_bars`
input will be added — without it the purge flag cannot mean anything.

**Where.** Recorded here only. No C6 or C7 code exists yet.

---

### 57. Five driver checks asserted a status code this application never returns

**Decision.** The `status == 400` expectations in `tools/e2e_part_b.py` (three) and
`tools/e2e_part_b_merge.py` (two) are corrected to 422.

**Why.** Every deliberate refusal in this application is a `ValidationFailure`, whose
`status_code` is 422; `app/core/errors.py` maps nothing to 400. The five checks would
therefore have reported a correct server as broken — an e2e driver that fails on correct
behaviour is worse than no driver, because it trains its reader to ignore it. Found by
reading the drivers against the error module rather than by running them, and fixed in the
same pass so the first run of the new `e2e_features.py` does not land next to five red
inherited checks.

**Consequence.** The comment above the column-op checks, which explained why a *400* could
only mean what the check said, now says 422. `tools/e2e_features.py` asserts the same codes
for the same reason.

**Where.** `backend/tools/e2e_part_b.py`, `backend/tools/e2e_part_b_merge.py`.

---

### 58. A multi-column feature's warm-up is the last of its columns' first-defined rows

**Decision.** For an entry that emits several columns, `FeatureSpec.warmup(n)` is the index
of the first row at which **every** one of those columns has a value — the latest of their
individual first-defined rows, not the earliest. `report["warmup_bars"]` applies the same
rule across a whole selection, and `usable_from` dates that row.

**Why.** The two are only the same number when every column of an entry starts on the same
row, and MACD is the entry where they are not: the line is `EMA(12) − EMA(26)`, defined from
row 25, while the signal is a nine-bar average *of the line*, defined from row 33. Declaring
25 would date a row as usable where the signal is still empty — a row the user cannot
actually use. Declaring something later would throw away rows that do carry values. 33 is
the only number that describes what the report means by "usable from", and it is also the
number the entry already declared, so no behaviour changed: what changed is that the rule is
now written down. The alternative — emitting MACD as three entries, or masking the line to
start where the signal does — was rejected because it would either split one checkbox in the
specification across three catalogue rows or delete eight real bars of the line.

**Consequence.** A test that asserts "every column is empty before the declared warm-up and
populated at it" fails this entry while the engine is right. The assertion in
`test_features_compute.py` was rewritten to check the rule that actually holds: the declared
warm-up equals the latest first-defined row, and every column is populated at it.

Five further defects in the never-run Part C test files were found and fixed — two by reading
them against the source, three only when the suite was first executed, and all five in the
tests rather than in the engine. Found by reading: `test_returns_and_momentum` asked for
`roc_2h` while passing the window `"2"`, which a duration feature refuses (§45's counterpart: a
bare number is ambiguous between bars and hours, so it is refused rather than guessed), and
`test_keep_only_selected_projects_the_stored_frame` expected five dropped columns where the
uploaded frame has six, having forgotten the `close_time` column that `hourly_frame` writes and
an upload stores unchanged. Found by executing: that same test's *hand-computed* expectation
for `roc_2h` on the last bar was wrong — it assumed a close the fixture does not carry at that
index, and `momentum_2h` beside it carried the same assumption, which no amount of reading the
file would have exposed; both were recomputed against the fixture's own closes, and the case
where the two-bar move is exactly zero was added, because that is the one an implementation
that guards a stale denominator gets wrong. Also found by executing:
`test_an_empty_selection_is_refused` read the refusal's hint from `error["hint"]` where the
envelope nests it under `error["details"]["hint"]`, and
`test_a_custom_bar_count_below_the_minimum_is_refused` passed a bare `"1"` to a *duration*
feature, so the server refused it with "'1' is not a duration" — a different refusal from the
one the test was named for. That test now passes `"1h"` and is named for what it checks.

**Where.** `backend/app/services/feature_catalog.py` (`FeatureSpec.warmup`),
`backend/app/services/features.py` (`report["warmup_bars"]`),
`backend/tests/test_features_compute.py`, `backend/tests/test_features_api.py`.

---

### 59. The collinearity scan has a named threshold, and the threshold travels in the report

**Decision.** After every run the engine scans the **generated** columns pairwise for
|Pearson r| ≥ `COLLINEARITY_R` (0.99) over the sample and reports up to
`COLLINEARITY_MAX_PAIRS` (20) pairs, strongest first, as `{a, b, r}`. The threshold is a named
constant in `features.py` and is also sent as `report["collinearity_threshold"]`, which is the
number the page prints in the sentence above the table.

**Why.** A pair of near-identical columns is the one modelling hazard a user cannot see in the
frame: five neighbouring averages of one price are five columns carrying about one signal, and
the only visible symptom is a coefficient that moves when the solver is re-run. The threshold is
deliberately strict, in the same spirit as the quality detectors' conservatism: the scan's value
is that a line in the report means *these two are the same column*, so it must stay silent about
pairs that are merely correlated — a scan that fired at 0.9 would flag an average against the
hour of the day on any trending series and be noise. The number was written out in three places
when the pass was first built (the constant, the frontend's sentence, and the e2e driver's
check); the driver's is a *check* and may keep its subject, but a constant restated in prose is a
constant that can disagree with itself, so the page now renders the value the engine sent.

**Consequence.** The scan is part of the *report*, so a preview and a write describe collinearity
identically, and nothing is ever dropped or renamed on the strength of it — a pair is named, with
its `r`, and left alone. Two tests pin the behaviour rather than the constant: a perfectly
collinear ramp pair must be reported, and an unrelated pair must not. The positive test's subject
is a straight ramp rather than the seeded random walk the rest of the file uses, because on a
linear ramp SMA(5) is exactly `close − 2` and SMA(10) exactly `close − 4.5`, so r = 1 by
arithmetic; whether a random walk trips a 0.99 threshold is a property of the seed, which is the
wrong thing for a threshold test to pin.

**Where.** `backend/app/services/features.py` (`COLLINEARITY_R`, `COLLINEARITY_MAX_PAIRS`,
`_collinearity`, the report dict), `frontend/src/api/types.ts`,
`frontend/src/pages/FeatureEngineering.tsx`, `backend/tests/test_features_compute.py`,
`backend/tests/test_features_api.py`, `backend/tools/e2e_features.py`.

---

### 60. The catalogue's query parameter is `dataset_id`, and the client was sending `datasetId`

**Decision.** The feature catalogue is read as `GET /api/features/catalog?dataset_id=<id>`. Every
name on the wire in this application is snake_case — route parameters (`/datasets/{dataset_id}`),
request bodies (`base_dataset_id`), response fields — and query parameters are no exception. The
only query parameter that is more than one word is this one, and `frontend/src/api/client.ts` was
sending it as `datasetId`, a camelCase TypeScript local that had leaked into the URL. Corrected at
the call site; the endpoint is unchanged.

**Why.** The server was right: asked for a parameter it does not declare, FastAPI answered
`422 {"detail":[{"type":"missing","loc":["query","dataset_id"],"msg":"Field required"}]}` — a
refusal that reads as nonsense in the access log, because the request did carry the id, under a
different name. Nothing in the build could see it. The TypeScript types describe the *response*, so
they constrain what a URL returns and not how it is spelled, and `test_features_api.py` passes
`params={"dataset_id": ...}`, building the query string itself — so the only catalogue URL any test
ever sent was one the test had written. The client's URL reached no server until a browser was
pointed at the page, and the failure was total rather than partial: every dataset selected on the
Features Engineering page rendered "Field required" where its catalogue belongs, so C5 was
unreachable in the application as shipped while 952 unit tests passed.

**Consequence.** A live driver was the instrument that found it, and only because it sent the URL
the browser sends: `tools/e2e_features.py` was spelling it `datasetId` too, so it reproduced the
page's request rather than the endpoint's contract. Its two call sites were corrected alongside the
client's, deliberately — a driver that used the spelling the server wants would have hidden the
defect it exists to catch. The rejected alternative was to accept both spellings server-side with a
query alias: two names for one parameter is a second thing to keep in step, and the API's
convention is unambiguous everywhere else.

**Where.** `frontend/src/api/client.ts` (`featureCatalog`), `backend/tools/e2e_features.py`.
Evidence: `backend/uvicorn.log` — `GET /api/features/catalog?datasetId=…` answered 422 on every
attempt, the same request spelled `dataset_id` answered 200.

---

### 61. The drivers pin their own output encoding

**Decision.** `tools/verify_all.py` starts every child with `PYTHONIOENCODING=utf-8` in its
environment, and each of the four drivers reconfigures `sys.stdout` and `sys.stderr` to UTF-8 as
its first act.

**Why.** The drivers print the arrows, mid-dots and em dashes their labels read better with
(`badge → path`, `ok  [check] — detail`). When `verify_all.py`'s own stdout is a pipe — which it is
whenever its output is captured rather than watched — a child that inherits that stdout is no
longer attached to a console, so Python encodes with the machine's locale code page instead of the
console's Unicode writer. This machine's is cp1256, an Arabic code page in which `→` (U+2192) has
no byte at all: the drivers died with `UnicodeEncodeError: 'charmap' codec can't encode character
'→'` *after* passing every check they had run — `e2e_ui.py` had already printed five `ok`
lines, and `e2e_part_b_merge.py` printed none, its first line containing an arrow. The two ends
were also disagreeing: the parent already decoded the child's output as UTF-8.

**Consequence.** The characters stay. Deleting them to suit one machine's code page would make the
output worse everywhere else, so the fix went into the plumbing rather than the labels. Setting the
variable in `verify_all.py` covers every child, including any added later; the per-driver guard
covers the documented way of running one alone (`python tools\e2e_features.py`), which is a case the
parent cannot reach. In the first full run of the suite six of the ten gates passed and all four
drivers failed — two of them on this single character, one on §60's URL, one on §62's fixture.

**Where.** `backend/tools/verify_all.py` (`run`), `backend/tools/e2e_part_b.py`,
`backend/tools/e2e_part_b_merge.py`, `backend/tools/e2e_features.py`, `backend/tools/e2e_ui.py`.

---

### 62. The Part B cleaning driver uploads its own fixture

**Decision.** `tools/e2e_part_b.py` uploads `DEFECTIVE_CSV` at the start of every run and binds
`DEFECTIVE` to the id it gets back; it no longer reuses a dataset id left in the store. Because the
frame is then known rather than found, the duplicate policies' row counts are asserted exactly —
keep 15, drop_exact 14, keep_first_by_time 13, keep_last_by_time 13 — instead of as an inequality.

**Why.** The check that failed on the first full run, "drop_exact removes rows that keep retains",
was asserting against a frame the driver had not built. It accepted any stored dataset whose columns
looked like OHLCV and whose row count was non-zero, and the one it found held 13 rows in which
timestamps repeat but no row does. `drop_exact` therefore removed exactly the nothing it is defined
to remove — 13 → 13, step reported, 0 cells touched — the two by-time policies removed 2 (13 → 11),
and the check reported a correct server as broken: it required `drop_exact < keep`, and the two are
equal whenever the frame's only repeated timestamps carry different values. The premise in its
comment, "the dataset is known to contain duplicates", was true in the timestamp sense and false in
the exact sense, and nothing in the driver distinguished the two. A fixture that is *found* measures
whatever the store happens to contain, which is not a property of the application.

**Consequence.** Every run adds one 15-row uploaded dataset — which is what the driver already did
for the other frames it uploads and for the cleaned child it writes. The inequality became exact
arithmetic, which is both the stronger check and the one a known fixture supports: the seeded frame
carries one row repeated in every column (the 04:00 bar) and one timestamp repeated with different
values (the 06:00 bar, a provider correction), so `drop_exact` must remove one row and each by-time
policy two. The old form would have passed a policy that removed the wrong row.

**Where.** `backend/tools/e2e_part_b.py` (`DEFECTIVE_CSV`, the seeding block, `DUPLICATE_OUTCOMES`).

---

### 63. ag-Grid v33 registers no modules of its own, and says so only in the console

**Decision.** `frontend/src/components/DataGrid.tsx` calls
`ModuleRegistry.registerModules([AllCommunityModule])` at module scope, beside its stylesheet
imports. Registration is global and idempotent, so it belongs there and not in a component body.

**Why.** Driving the page in a browser for the first time showed that the data grid had never
worked at all. `GET /data/{id}` rendered the 5+5 preview, the toolbar, and an empty container:
zero `.ag-row` and zero `.ag-header-cell`. In ag-Grid v33 an explicit registration is required —
`createBeansList` returns before any bean is created when `_hasUserRegistered()` is false, after
`_logPreInitErr(272, void 0, NoModulesRegisteredError())` — and `ag-grid-react` does not call
`registerModules` on the application's behalf. Nothing in `frontend/src` mentioned `ModuleRegistry`,
so nothing ever did.

The failure is quiet in the way that matters most: the toolbar above the grid is ordinary React and
kept reporting "0–33 of 33", so the page looked like a working grid that had been handed no data
rather than like a grid that never started. It was found by reading
`node_modules/ag-grid-community/dist/package/main.esm.mjs` — the error is only ever logged, never
thrown, and it is logged at a level this project's build does not surface.

**Consequence.** Four of this driver's grid checks now pass against real rows and real columns
(they had been failing, and each was first misread as a driver defect). The `grid` chunk grew
because `AllCommunityModule` pulls in every community feature rather than the ones this page uses;
the figure is in "Still open". Registering the specific feature modules instead would be smaller
and is the obvious next move, but it is a build-size decision and not a correctness one.

**Where.** `frontend/src/components/DataGrid.tsx`; exercised by `backend/tools/e2e_ui.py`, which
waits for `.ag-row` to appear.

---

### 64. The UI driver compares rendered text without regard to case

**Decision.** `backend/tools/e2e_ui.py` compares page prose through `shows(page, phrase)`, which
lower-cases both sides, and reads report labels through `REPORT_KEYS`, which lower-cases each
`.report__key` as it collects it.

**Why.** The stylesheet upper-cases most of the interface's chrome — `.section__title`,
`.meta__key`, `.badge`, `.policy__legend`, `.report__key`, `.dtable th`, every `label` — and
Chrome's `innerText` returns text as *rendered*, not as authored. Twelve prose checks and every
report-label check were therefore asserting on the font's letter case: "a quality report is present"
read the page back as "DATA QUALITY" and failed; the column-editor heading was looked up with a
case-sensitive `includes('first 5 and last 5')`, which returned `null` and then raised `TypeError`
on the subtraction that followed — a traceback from inside the reporter instead of a verdict. The
checks are about which words the page shows, not about how the stylesheet renders them, so the
comparison drops case rather than the stylesheet dropping its transformation.

**Where.** `backend/tools/e2e_ui.py` (`shows`, `REPORT_KEYS`, and the `.section__title` finder).

---

### 65. The UI driver's base dataset is one the suite has not written

**Decision.** `pick_datasets()` prefers an OHLC dataset with `parent_dataset_id is None`, and falls
back to the full candidate list — printing that it did — only when every candidate is derived.

**Why.** The driver picks a base and then runs A2, DM, B3, B4 and C5 against it, and it also writes
datasets: B3's cleaned child, B4's merge, C5's feature set. Choosing from the listing in order
therefore let each run's *output* become the next run's *input*. A feature set is the one base C5
cannot be run against: it already carries the columns the catalogue generates, so ticking the first
feature and pressing select-all asks for `price_lag_1` on a frame that already has it, and
`features.py` refuses by design — "These columns already exist in the dataset", with a hint to
rename them in Resample & merge. Eight checks reported a working page as broken, and the console's
only complaint was a status code with no request attached to it.

`parent_dataset_id` is the store's own record of the distinction the driver needs: every write
these drivers make forks a child and stamps the parent on it, so a row without one is an input — a
fetch or an upload — and not an output. Preferring a `raw_data` category instead would have worked
against today's catalog and encoded a weaker claim.

**Consequence.** The base is stable from run to run, and it is printed with its row count, timeframe
and category, because several checks below read the numbers on the page rather than a fixed
expectation and a failure is only readable against the dataset it was measured on.

**Where.** `backend/tools/e2e_ui.py` (`pick_datasets`).

---

### 66. A column count is read from `aria-colcount`, not from the header cells in the DOM

**Decision.** The Data Management section measures a column count as
`document.querySelector('.ag-root').getAttribute('aria-colcount')`.

**Why.** ag-Grid virtualises columns horizontally. At the width the driver runs the grid, five of
eight header cells are in the DOM, and a column appended past the right edge is in the column model
and not in the viewport — so "Add column adds a column" compared `0 -> 0` rendered headers against
a grid that had correctly added the ninth. `aria-colcount` is set from `colModel.getCols().length`
on every `onGridColumnsChanged`, which is the count the check claims to be about, and it is
independent of the viewport and of virtualisation.

**Where.** `backend/tools/e2e_ui.py` (`COLCOUNT_GRID`, the Add/Rename/Hide checks).

---

### 67. The browser is driven at an emulated desktop viewport, not at the one it launches with

**Decision.** `Page.__init__` sends `Emulation.setDeviceMetricsOverride` with a 1440×1000 viewport,
one device pixel per CSS pixel, not mobile.

**Why.** Headless Edge opens a 500×450 viewport and ignores `--window-size`, which this driver
first tried and which had no effect. The shell is `grid-template-columns: auto auto 1fr`, so at that
width the main pane — the `1fr` track — collapsed to nothing while the fixed sidebar and explorer
held their own width: `.page` measured 52px and `.grid__body` 2px. Every grid check would then have
been measuring the window rather than the application, and failing against a page that renders
correctly at any size a person would use. Setting the viewport through the protocol does take
effect.

**Where.** `backend/tools/e2e_ui.py` (`Page.__init__`).

---

### 68. Three UI-driver expectations described the fixture rather than the page

**Decision.** Three checks were rewritten to assert what the page owes rather than what the run's
particular data happened to produce.

**Why.** Each failed against a page that was behaving correctly, and each failed for a different
reason.

- **Funding under spot.** The A1 section set `#dtype` to `funding` while `#market` was
  `crypto_spot` and expected the timeframe control to disappear. Funding is a futures-only
  product, and the data-type select offers exactly the products the chosen market serves — so the
  assignment did not take, the market stayed on klines, and the timeframe control correctly stayed
  with it. The market is now switched to `crypto_futures` first and the option's presence asserted
  before it is used.
- **`Next` on a one-page dataset.** The pager check asserted that clicking `Next` changes the
  visible row range. The driver's base held 33 rows and the smallest page size is 50, so `Next` was
  correctly disabled and the click correctly did nothing. The check now reads the pager's own
  "page 1 / 1" and asserts one of the two correct behaviours accordingly: either the arrow moves
  the window, or it is disabled because there is no second page.
- **The alignment radios.** `document.querySelectorAll('.join .policy input[type=radio]')`
  collected both policy groups, because Alignment and Match are both `.join .policy`, and compared
  the union `['backward', 'exact', 'left', 'inner']` against the two look-ahead-safe methods the
  check is about. The group is now selected by its legend, so the check tests the alignment list
  and not the match list.

**Where.** `backend/tools/e2e_ui.py` (the A1 data-type block, the DM pager block, the B4 join
block).

---

### 69. A refused request is named in the driver's log

**Decision.** The driver records every response with a 4xx or 5xx status as `"<status> <path>"` and
prints the list under the console errors. It is reported, not failed, like `CONSOLE_ERRORS`.

**Why.** Some of these are refusals the driver provokes on purpose, and the console message for a
refused `fetch` is "Failed to load resource: the server responded with a status of 422", which
names a status and not a request. While §§63–68 were being fixed, a real 422 was visible in the
log with no way to tell which call produced it; identifying it took a second run with the URL
captured. Naming the request in the log is what made the C5 collision (§65) legible the first time
it appeared instead of the second.

**Where.** `backend/tools/e2e_ui.py` (`_observe`, `REFUSALS`, the summary block).

---

### 70. The sidebar check waits for the heading instead of sampling it

**Decision.** The check that presses each sidebar entry now waits for `.page__title` to carry the
expected text, rather than reading it once as soon as `location.pathname` has settled.

**Why.** The path changes with the router's navigation and the heading is rendered a frame or two
later, so the check sampled the *previous* page's title — it reported `A1 → /fetch` with
`title='Data Management'` about one run in ten, against a sidebar that had navigated correctly.
Nothing about the application was wrong: `Data Management` was the entry pressed immediately
before, and its title was still in the DOM when the read happened. The difficulty is that a check
like this fails rarely and non-reproducibly, which is the worst way for a driver to be wrong; the
two waits are not belt-and-braces, they are the two halves of what the check claims — that pressing
the entry routes there *and* that a titled page renders behind it.

**Where.** `backend/tools/e2e_ui.py` (the `sidebar · every entry` loop).

---

### 71. A header sort is clicked on the label, and counted only when the server served it

**Decision.** The Data Management section sorts by clicking `.ag-header-cell-text`, and asserts two
separate things: that the toolbar reports `sorted by close:asc`, and that no `/rows` request was
refused while it did.

**Why.** ag-Grid binds the sort handler to the header's *label*, not to the header cell —
`sortSvc.setupHeader(this, column, this.eLabel)` in ag-grid-community 33.3.2, where `eLabel` is the
`.ag-header-cell-label` element. A click dispatched on the surrounding `.ag-header-cell` bubbles
*uphill*, away from a listener on a descendant, so the grid would not have sorted at all and the
check would have waited out its timeout against a working application. This is the same class of
defect as §66 and §68 — an expectation that described the DOM rather than the page — except that
this one was found by reading the library instead of by running the driver.

The second assertion exists because the toolbar's text is local state: it reads `sorted by` from
the moment the header is clicked, whether or not the server accepted `GET /rows?sort=close:asc`.
A refused sort would leave the grid showing an unordered frame under a label claiming otherwise,
and `_order_clause` refuses an unknown column rather than silently falling back to unordered. The
driver already records refused responses for exactly this kind of question (§69), so the check
reads what it records rather than adding a new instrument.

**Where.** `backend/tools/e2e_ui.py` (`DM · sorting`).

---

### 72. The grid-edit checks target a float column, and read the page after it settles

**Decision.** The cell-menu checks open the menu on the `close` cell — found by its `col-id`
attribute, which ag-Grid stamps from the column's id — and the rows-per-page check reads the
toolbar's window only once it has stopped reading `loading…`.

**Why.** The first cell with text on screen is `open_time`, so "Clear cell" on it stages an edit
that blanks the dataset's own timeline. The save that follows could then be refused for a reason
that has nothing to do with the menu under test, and the check would be measuring the wrong thing.
A float column can be blanked and still leave a frame every later step can read.

The range span carries `loading…` while a window is in flight, so sampling it immediately after
the page-size change compares that placeholder against the previous text: it reports a widening
that never happened, or a failure that did not. The check waits for a settled range and branches
on the total the toolbar reports, because whether a larger page widens anything depends on how
many rows the driver's chosen dataset holds — a property of the store, not of the page.

**Where.** `backend/tools/e2e_ui.py` (`DM · editing`, `DM · page size`).

---

## Part D — Regime detection

### 73. `n_states` defaults to 4, not 3, and the fourth state is not invented

**Decision.** The request model defaults `n_states` to **4** where the specification says 3. The
output schema is fixed at four probability columns — `prob_trending_up`, `prob_trending_down`,
`prob_ranging`, `prob_crisis` — plus the derived `Transition`. After a fit, each latent state is
mapped to one of the four names by a disclosed rule (ordered by mean return and mean volatility:
the highest-return low-volatility state is Trending Up, the lowest-return high-volatility state is
Trending Down, the remaining low-volatility state is Ranging). A state that ends up carrying no
name is reported in `states_without_a_regime` and its mass is redistributed rather than assigned
to a name it did not earn.

**Why.** The specification's own output schema names four states while its own `n_states` default
is 3. Those cannot both hold: a three-state fit produces three probability vectors and four columns
have to be filled, so something must be invented — either a mapping from three states to four names
or a fourth column of zeros. Both are worse than fitting four. The schema is the contract the rest
of the pipeline reads, so the schema wins and the default moves. The specification's 3 is not
discarded — it is still reachable by moving the slider, and the naming rule above is what makes
that a legitimate choice rather than a degraded one: two latent states genuinely sharing one name
is a finding about the market, and it is reported rather than hidden.

**Where.** `backend/app/models/regime.py` (`RegimeDetectRequest.n_states`),
`backend/app/services/regime_engine.py` (`_name_states`, `_collapse_to_names`).

---

### 74. `refit_every` is measured in bars from the start of the series, never as a fraction

**Decision.** The walk-forward schedule is anchored to absolute bar indices. The first fit trains on
`[valid_from, valid_from + min_train_bars)` and each subsequent refit advances by exactly
`refit_every` bars. Truncating the series removes later refits and moves none of the earlier ones.

**Why.** The natural implementation — "refit every 10% of the series" — is silently non-causal, and
the failure is invisible in any single run. With a 10 000-bar series the refits land at 1 000,
2 000, …; truncate to 8 000 and they land at 800, 1 600, … — every training window in the first 80%
has changed, so the labels in the first 80% change too, purely because of data that had not printed
when those bars were labelled. The specification's causality test (truncate to 80%, rerun, compare
the first 80%) fails, but only for a reason that is very hard to see: the model is fine, the numbers
are plausible, and the dependence is on the *length* of a series rather than on its values. There is
a regression test that pins the refit points at `[223, 423, 623]` and asserts the first training
window starts at the same bar in both runs, so the property cannot be lost by a later refactor.

**Where.** `backend/app/services/regime_engine.py` (`_refit_schedule`, `_Segment`),
`backend/tests/test_regime_engine.py`
(`test_refit_schedule_is_anchored_to_bar_index_not_series_length`).

---

### 75. XGBoost replaces LightGBM in the hybrid model

**Decision.** The hybrid detector is an `xgboost.XGBClassifier`, not the specification's LightGBM.
The substitution is stated in the model's own docstring, in the report, and here.

**Why.** LightGBM is not installed in this environment and cannot be assumed to be. XGBoost is
installed and is the same family of model — gradient-boosted decision trees, the same inductive
bias, the same handling of the feature set here (a handful of continuous columns, a few hundred to
a few thousand rows). The two are not identical and the difference is not claimed to be zero; what
is claimed is that the choice between them is not load-bearing for this task at this scale. What
would be worse is the alternative the specification implies — a module that raises `ImportError` on
a machine without LightGBM, for a dependency the user did not know they needed.

**Where.** `backend/app/services/regime_engine.py` (`_fit_and_label_hybrid`).

---

### 76. An additive table does not bump `SCHEMA_VERSION`

**Decision.** The `jobs` table was added to the catalog with `CREATE TABLE IF NOT EXISTS` and
`SCHEMA_VERSION` stayed at 1.

**Why.** `initialise` refuses to start on a version mismatch, by design — a changed column meaning
or a dropped table must not be opened by code that does not expect it. But an *additive* table is
not that: `CREATE TABLE IF NOT EXISTS` creates it on the next start whether or not the version
moved, so bumping would tell every existing installation to migrate or delete its catalog in order
to gain a table it was about to be given anyway. The rule this leaves behind: bump for a change the
user must act on, not for a change the schema can absorb.

**Where.** `backend/app/storage/catalog.py` (`SCHEMA_VERSION`, `_SCHEMA`).

---

### 77. The HMM is hand-rolled, because the library returns smoothed posteriors

**Decision.** `regime_models.fit_gaussian_hmm` and its forward recursion are implemented in this
repository rather than delegated to `hmmlearn.predict_proba`.

**Why.** The two are answering different questions, and only one of them is causal.
`predict_proba` returns the **smoothed** posterior `P(s_t | x_1..x_T)` — the state at bar *t* given
the entire series, including bars that had not printed when *t* was live. The quantity a strategy
can act on is the **filtered** posterior `P(s_t | x_1..x_t)`, from the forward recursion alone.
They look similar, they often agree, and where they disagree is exactly at the turns — the smoothed
path knows a crash is coming several bars early, so a backtest on it is a backtest on the future.

This is the one place in the module where the specification's own verification test (Part 5.1:
truncate to 80%, rerun, assert the first 80% is unchanged) cannot pass on the obvious
implementation. Smoothing conditions every label on the tail, so truncating changes all of them.
The test was written first and the implementation was changed to satisfy it, not the reverse.

Baum-Welch's E-step legitimately uses smoothing over the *training* window — that is what fitting a
model means, and the training window ends before the bars being labelled. The distinction that
matters is between smoothing inside a fit and smoothing across the boundary between a fit and the
labels it produces.

**Where.** `backend/app/services/regime_models.py` (forward recursion, Baum-Welch),
`backend/tests/test_regime_engine.py`
(`test_truncating_the_series_does_not_change_the_earlier_labels`).

---

### 78. `confidence` is the committed state's probability, and the crisis override revises the posterior

**Decision.** Two linked decisions about what the output frame's columns mean.

* **`confidence` is `P(committed state)`**, not the largest probability in the row. On a bar
  labelled Transition with no incumbent committed, it is the weak leader's probability.
* **The crisis override is written into the probabilities**, not only into the label. A bar whose
  trailing realized volatility breaches the threshold has `prob_crisis` raised to at least
  `override_probability` (default 0.90) and the remaining mass redistributed across the other three
  *in proportion to what the model already believed*. The revision is applied inside
  `commit_labels`, not by its caller.

**Why.** Anti-thrashing makes the committed label lag the argmax: the model can be most confident
of Ranging while the label still reads Trending Up, because the challenger has not held
`commit_confidence` for `min_state_duration` bars yet. Reporting the row maximum on those bars
would tell the user "70% confident it is Trending Up" when the 0.70 belonged to Ranging — and it
would make the interface's own "below 55%, treat this as a gray zone" warning unreachable, because
the maximum is high by construction whenever any state is doing well. A warning that cannot fire is
not a safeguard.

The crisis half was a real defect found by running the code, not by reading it. The first version
relabelled the bar Crisis while leaving `prob_crisis` at ~1e-183 — the model's own likelihood, which
had not anticipated a spike. The frame then contradicted itself twice over: Crisis with a crisis
probability of zero, and an `argmax` over the probability columns that disagreed with
`dominant_state`. A consumer reading the columns instead of the label got the opposite answer, and
nothing raised. The fix treats the breach as what it is — evidence the likelihood did not model —
and revises the posterior with it.

The revision lives inside `commit_labels` rather than beside it because the call-order version of
the fix is the version that broke: a caller that passed a `crisis_mask` without having applied the
override first got a self-contradicting frame, and nothing in the type system prevented it. Making
`commit_labels` return the revised posterior means the frame written to disk and the labels chosen
from it are the same array.

**Where.** `backend/app/services/regime_engine.py` (`apply_crisis_override`, `commit_labels`,
`Labelling.named`), `backend/tests/test_regime_engine.py`
(`test_confidence_falls_when_the_committed_state_loses_support`,
`test_the_label_and_its_probabilities_never_contradict_each_other`).

---

### 79. Net Value-Add is a disclosed rule, scored in-sample, and never reported without the caveat

**Decision.** Net Value-Add is computed as `return_improvement − switching_cost`, where the strategy
is **long in Trending Up and flat otherwise**, a position set at bar *t* earns the return of *t+1*,
the benchmark is buy-and-hold over the same bars, and each **change of position** — each entry into
the long and each exit from it — is charged `switching_cost_penalty`. The definition string travels
with the number in every payload, and every payload also carries `in_sample: true`.

**Why.** The specification asks for Net Value-Add and does not define it. A number called
"Value-Add" with no stated rule is worse than no number, because it invites the reader to supply
their own rule and reach a different conclusion from the same figure. The rule chosen is the
simplest one that makes "avoiding a drawdown" show up as a gain, and it is deliberately *not*
tuned — a rule fitted to flatter the detector would defeat the point of reporting it.

The causality detail is not incidental: the position is set from the label at *t* and earns the
return of *t+1*, so the metric cannot be earned by a label that was partly computed from the
return it is being paid for. Getting this backwards is the classic way a regime detector backtests
beautifully.

`in_sample: true` is not a hedge. The metric is computed on the same bars the model was fitted on,
so it is a diagnostic — "did the labels separate the returns at all" — and not evidence that the
detector works. Reporting it without that flag would let a reader reasonably take it for
out-of-sample performance, and the field exists so that a future walk-forward figure can be told
apart from it at a glance.

The cost is charged per *position* change, not per state change, and the two are reported under
different names for that reason — see §83.

**Where.** `backend/app/services/regime_engine.py` (`net_value_add`),
`backend/app/services/regime.py` (`_lineage_detail`).

---

### 80. ADX and RSI are carried, never computed; availability is decided server-side

**Decision.** `adx` and `rsi` are marked `carried` in `GET /api/regime/options` and are available
only when a column of that name already exists in the dataset and is not in `unpopulated_fields`.
They are never computed here. The same route answers, for every model and every observation,
whether it is available and **why not** — and the frontend renders that verdict rather than
re-deriving it.

**Why.** ADX and RSI belong to C5, which owns their formulas, their parameters, and their info
icons. Computing them a second time in D would create two implementations of the same indicator,
and the two would drift — at which point the user's C5 `rsi_14` and D's `rsi` would disagree with
no way to tell which the model saw.

The availability rule is computed once, on the server, for the same reason the feature catalogue is:
if the browser decided "hybrid is greyed out because there is no ADX" and Python decided "hybrid is
refused because there is no ADX", the two can only ever agree by coincidence. Deriving it twice is
how a UI ends up offering a control the backend then rejects. `kind: "computed" | "carried"` is
what lets the message be right in both cases — "produce it in C5 and merge it in" is a different
instruction from "this dataset has no volume column".

**Where.** `backend/app/services/regime.py` (`_FEATURE_SOURCES`, `_CARRIED`,
`RegimeService.options`), `frontend/src/pages/RegimePage.tsx`.

---

### 81. Plotly is split into its own chunk and reached only through a lazy route

**Decision.** `plotly.js-dist-min` is named as a manual chunk in `vite.config.ts` **and** the D1
page is a `React.lazy` route. The combination is the decision; neither half is sufficient.

**Why.** Plotly is 4.7 MB minified — several times the size of the entire rest of the application —
and exactly one page uses it. A manual chunk on its own changes how the bytes are *laid out*, not
when they are fetched: the route was still a static import, so the entry chunk pulled Plotly in on
first paint, which took the entry from 365 kB to 5 098 kB. Deferring the import is what makes the
split mean anything; measured, the entry drops back to 365 kB (111 kB gzipped) and Plotly becomes a
separate 4.7 MB chunk fetched on the first visit to `/regime`.

D1 is the only lazily-loaded route, and that is deliberate rather than an oversight. The other pages
share nearly all of their dependencies, so splitting them buys a round trip and nothing else —
there is no general code-splitting policy here to be consistent with, only one very large
dependency that one page needs.

**Where.** `frontend/vite.config.ts`, `frontend/src/App.tsx`.

---

### 82. `warmup_bars` is the observation warm-up; `unlabelled_bars` is the whole null prefix

**Decision.** The report carries both. `warmup_bars` is the number of bars the observation window
consumes before a feature vector exists at all (24 at the default volatility window).
`unlabelled_bars` is the entire run of rows that carry no committed state, which also includes the
bars the first model trained on, and `unlabelled_bars == warmup_bars + min_train_bars`. The
`state_counts` bucket is named `warmup` for historical reasons but counts `unlabelled_bars`, and the
two now agree; the frontend labels it "Unlabelled (warm-up + training)" rather than "Warm-up".

**Why.** Found by running the driver: on a 200-bar frame with a 60-bar training window, the first
label appears at bar 84, and the report was calling the gap 24. Three and a half times too small,
under a name that reads as "the rows before the first label" — the reading a user will bring to it.
Worse, the same payload already contained the correct number under `state_counts["warmup"]`, so the
two fields disagreed about one concept. That is the same class of defect as §78: a payload that
contradicts itself is not merely imprecise, it is unusable, because no field in it can be trusted
over any other.

The two quantities are genuinely different and both are worth reporting. The observation warm-up is
a property of the feature set; the training window is a property of the schedule the user chose.
Collapsing them would hide which knob to turn when the labelled region is too short.

**Where.** `backend/app/services/regime_engine.py` (`detect`'s report),
`frontend/src/api/types.ts` (`RegimeReport`), `frontend/src/pages/RegimePage.tsx`.

---

### 83. Net Value-Add reports `position_changes` and `bars_long`, not a second `switch_count`

**Decision.** The evaluation block keys its cost count as `position_changes` and adds `bars_long`,
the number of scored bars the strategy actually held a position on. The report's top-level
`switch_count` continues to mean committed state changes.

**Why.** The evaluation counts changes of *position*; the report counts changes of *state*. Moving
from Ranging to Crisis is a state change that trades nothing and costs nothing under the rule in
§79, so the two numbers differ — and the evaluation was calling its count `switch_count` inside the
same payload as the report's. A reader checking the arithmetic by hand, which is exactly what one
does with a figure like this, would multiply the wrong count by the penalty and find the total
wrong. Two fields, one name, is a defect whether or not either number is individually correct.

`bars_long` exists because a run whose rule never went long has `return_improvement` of exactly
`−benchmark_log_return`, and a Net Value-Add that measures the market rather than the model is not
recognisable from the number itself. This is not hypothetical: the heuristic path multiplies every
state's probability by `quiet = 1 − volatility / crisis_threshold`, so on the driver's frame
(volatility ≈ 0.14 against a 0.35 threshold) `quiet` is 0.6 and no calm bar can reach the 0.60
commit threshold — the heuristic labels 150 of 175 scored bars Transition and commits only through
the crisis override. That behaviour is now asserted in the driver rather than rediscovered, and the
D1 panel says so in words when `bars_long` is zero.

**Where.** `backend/app/services/regime_engine.py` (`net_value_add`),
`backend/tests/test_regime_engine.py`, `frontend/src/pages/RegimePage.tsx`.

---

## Part C, second pass — C6 Chronological Splits and C7 Scaling

*Placed here rather than inside the Part C block above because the entries are numbered in one
continuous series across the document: the Part C block ends at §72, and §73–§83 stand under
Part D. Putting §84 inside that block would place it ahead of §73–§83. The heading is what
groups these; the numbers stay in the order they were written.*

Both modules exist to hold one invariant, and it is worth stating before the entries that
follow, because almost every decision below is a way of making it impossible to violate rather
than merely discouraged:

> **Fit on the training span. Transform everything. Never fit on anything the model will be
> scored on.**

### 84. C6 and C7 are a service and a router, not an `app/modules/` package

**Decision.** The specification's `backend/app/modules/splits/` and `backend/app/modules/scaling/`
are built as `app/services/splits.py` + `app/routers/splits.py`, and the same pair for scaling.

**Why.** There is no `app/modules/` anywhere in this codebase. Every module is a service holding
the arithmetic and a router holding the HTTP shape, and C5 and D1 already follow it; a parallel
layout for two of the seven modules would be a second convention to learn for no gain, and the one
question a reader arrives with — where does the arithmetic live — would have two answers. The
split is not cosmetic: the services import with no FastAPI in the path, which is how
`tests/test_splits.py` drives the span geometry and `tests/test_scaling.py` drives the fit
directly rather than through HTTP.

**Where.** `backend/app/services/{splits,scaling}.py`, `backend/app/routers/{splits,scaling}.py`.

---

### 85. C6 and C7 refuse with 422, not 400 — and a pydantic violation has a different shape

**Decision.** Every refusal in the two modules is a `ValidationFailure` (422) or a failed job,
never a 400.

**Why.** Same reasoning as §57, restated because the specification asks for 400 again here:
`app/core/errors.py` maps nothing to 400, so a driver written against 400 would report a correct
server as broken. One wrinkle shapes both new drivers and is worth recording: the handler is
registered for `QuantmetricsError` only, so a *pydantic* constraint violation — `n_folds: 11`,
`gap_size: -1`, a missing `split_type` — is refused by FastAPI before any handler runs and arrives
as `{"detail": [{"loc": [...], "msg": ...}]}` with no `error` key at all. A driver reading only
`error.message` gets an empty string and reports "refused, for no stated reason". Both drivers
therefore read `error.message` **or** a JSON dump of the body, which is what `message_of` is.

**Where.** `backend/tools/e2e_splits.py`, `backend/tools/e2e_scaling.py` (`message_of`).

---

### 86. A span is dated by the observation column, never by `close_time`

**Decision.** The specification's "validate `close_time` strictly sorted" is implemented as a
strictly-increasing check on the dataset's *observation* column, resolved by
`quality.find_time_column` (`open_time`, `create_time`, `calc_time`, `Date`, `date`, `timestamp`).
`close_time` is not a candidate, and `tests/test_splits.py` asserts it appears nowhere in the
module.

**Why.** A span that starts at a bar's close instant dates every span one bar late and puts the
last training bar inside the test span — a one-bar leak with nothing in the resulting table to
show it. §14 fixed timestamp semantics as the *open* instant for exactly this reason, and
`tests/test_leakage_guards.py` already asserts `close_time`'s absence from the candidate list.
The e2e driver checks the consequence rather than the rule: the plan's `start_time` must equal the
source's `open_time` at `start_index`, and must *not* equal its `close_time`.

**Where.** `backend/app/services/splits.py` (`_time_column_of`), `backend/tools/e2e_splits.py`.

---

### 87. The embargo and the purge overlap, so the plan reports what each rule actually removed

**Decision.** `gap_size` and `purge_flag` are both offered, and the report carries a boundary row
per cut stating `cut_index`, `gap_rows_removed`, `purge_rows_removed` and `effective_end_index`.

**Why.** The specification presents them as two independent knobs, and they are not additive. The
embargo drops `[c - gap, c)`; the purge drops every training row whose label horizon `h` reaches
the cut, `[c - h, c)`. Together the training span ends at `c - max(gap, h)`, so with `gap >= h`
the purge removes **nothing** — and a plan reading `gap_size: 5, purge_flag: true` would otherwise
look like twice the safety while being exactly as safe as either one alone. The report states the
counts, and `build_spans` emits the literal warning "purge_flag removed no rows" in that case
rather than leaving the user to derive it.

**Where.** `backend/app/services/splits.py` (`_apply_boundary`, `_warnings`),
`frontend/src/pages/SplitsPage.tsx` (the boundaries table).

---

### 88. Gap and purge are offered for percentage and walk-forward splits, not for date splits

**Decision.** `SplitParameters` refuses a non-zero `gap_size` or a set `purge_flag` on
`date_3way` and `date_2way`, with a message saying to move the date instead.

**Why.** The boundary leak is identical for a 70/15/15 split; the reason for the asymmetry is not
that a date split is safe. It is that on a date split the *user* names the boundary, and silently
shrinking a span they typed would be worse than having no knob at all — they would read a
train span in the table that ends where they did not put it. Where the application chooses the
cut, a knob is the only way to move it; where the user chooses it, the date is the knob.

**Where.** `backend/app/models/splits.py` (`SplitCreateRequest._check`),
`frontend/src/pages/SplitsPage.tsx` (the note under the date inputs).

---

### 89. `label_horizon_bars` is a required input whenever `purge_flag` is on

**Decision.** `purge_flag: true` with `label_horizon_bars` unset is refused by the request model,
not defaulted.

**Why.** §56 recorded that the purge cannot mean anything without a horizon: the number of rows it
removes *is* the horizon (minus the gap). A silent default — 1, or 5, or "one bar" — would be a
methodology choice made by the application on the user's behalf and invisible in the resulting
plan. The UI shows the box only when the purge is ticked, and a client-side check refuses to send
the request until it holds a positive number, so the refusal is explained before the round trip as
well as after it.

**Where.** `backend/app/models/splits.py` (`_check`), `frontend/src/pages/SplitsPage.tsx`.

---

### 90. A rolling walk-forward window is one block, derived from the fold count and printed

**Decision.** `tscv_rolling` takes no window length. `block = floor(N / (n_folds + 1))`, fold *k*
tests `[k·block, (k+1)·block)` (the last fold extended to `N`) and trains on the single block
immediately before it. `block >= 2` or the request is refused, with both numbers in the message.

**Why.** A rolling scheme needs a window length and the specification supplies none; adding a
second slider would also make the fold count and the window length independently settable while
the geometry ties them together, so most combinations would be refused. Deriving it keeps the fold
slider the only control, and the derivation is *stated in the report* rather than left implicit —
including the fact that the first block is never tested, because it is the training data every
fold starts from.

**Where.** `backend/app/services/splits.py` (`_tscv_spans`),
`frontend/src/pages/SplitsPage.tsx` (the fold slider's note).

---

### 91. The plan schema is a superset: row indices, a checksum, and the plan's own id in the table

**Decision.** The stored plan carries `split_id, fold, role, start_index, end_index, start_time,
end_time, row_count, dataset_checksum` — the specification's `{fold, role, start_time, end_time,
row_count}` plus three columns.

**Why.** The specification itself asks for `{fold, role, start_index, end_index}` in the TSCV case,
so both conventions are already in play; storing both means a plan is never reconstructed by
*searching* a timestamp column, which would silently pick the wrong row on a frame with a repeated
or missing bar. The checksum makes a plan refuse to apply to a frame it was not computed on, which
the rest of this application cannot otherwise guarantee. And `split_id` is written as a column
because the plan table is exportable: a copy of it that could not be traced back to the row it
came from would be a plan with no provenance. C6 supplies the id to `persist_dataset` through its
`dataset_id` kwarg so the plan's id and its dataset id are one identifier, not two.

**Where.** `backend/app/models/splits.py` (`SplitPlanRow`),
`backend/app/services/datasets.py` (`persist_dataset`, `dataset_id`),
`backend/app/services/splits.py` (`_plan_frame`).

---

### 92. The fitted parameters are an `outputs` dataset, not a file beside the frame

**Decision.** C7 writes two datasets: the scaled frame to `processed_data`, and the fitted
parameters to `outputs` as a one-row-per-feature table (`feature, scaler_type, center_, scale_,
constant, fitted_rows, formula`). Confirmed with the user before it was built.

**Why.** The specification asks for a `scaler_params.json`. A dataset is checksummed, lineage-linked,
visible in the Explorer, and exportable as JSON through the existing
`GET /api/datasets/{id}/export?format=json`, so the format the specification asked for is still
obtainable — from one source of truth rather than from a file beside the data that nothing verifies
and no page can show. What makes it worthwhile rather than merely tidy is `center_`/`scale_`: every
fitted scaler is reduced to that single affine pair, so `x = scaled · scale_ + center_` inverts
min-max, standard and robust alike, and a prediction can be returned to price units from the
lineage alone without knowing which scaler produced it.

**Where.** `backend/app/services/scaling.py` (`_parameters`, `_params_frame`),
`frontend/src/pages/ScalingPage.tsx` (the parameters card).

---

### 93. A walk-forward plan requires a fold in C7

**Decision.** `ScalingRequest` refuses a fitted scaler against a `tscv_*` plan with no `fold`, and
refuses a fold outside the plan. The UI shows the fold selector exactly when the selected plan is
a walk-forward one.

**Why.** The specification does not address TSCV and scaling. A walk-forward plan holds one
train/test pair per fold, so fitting on the union of every fold's training rows would compute fold
1's statistics partly from rows lying *after* fold 1's test span — look-ahead relative to that
fold, and invisible in the resulting frame, which looks exactly like a correctly scaled one.
Per-fold fitting is the only sound reading, and it is enforced rather than offered.

**Where.** `backend/app/models/scaling.py`, `backend/app/services/scaling.py` (`_load_plan`),
`frontend/src/pages/ScalingPage.tsx`.

---

### 94. Splits create is 201; scaling is 202 with a job

**Decision.** `POST /api/splits/create` runs its work through `run_blocking` and answers **201**
with the finished plan. `POST /api/scaling/fit_transform` answers **202** with a job handle,
polled through `GET /api/jobs/{id}` exactly as D1's regime detection is.

**Why.** The mandate behind both is that heavy work must not block the event loop, and
`run_blocking` — the sole sanctioned chokepoint, enforced by `tests/test_async_rule.py` —
satisfies it without a job. Creating a plan reads one column of the frame and writes a handful of
rows; a 202 would put a polling loop on the page for work that finishes in milliseconds, and a
caller that treated the handle as a result would show an empty table under a request that had
already succeeded. Scaling is the opposite case: it reads a whole Parquet file and writes a whole
one back, which on a large frame takes long enough that a request-scoped response would time out.

**Where.** `backend/app/routers/splits.py`, `backend/app/routers/scaling.py`.

---

### 95. The plan's timeline is positioned by row index, not by wall-clock time

**Decision.** `components/Timeline.tsx` is a stack of absolutely positioned `div`s whose offsets
and widths are `start_index / N` and `row_count / N`. Not a Plotly figure.

**Why.** A Gantt drawn on wall-clock time renders an hourly series and a daily series with the same
picture, and it cannot show a gap at all: an embargo that removes five bars is five hours wide on
an hourly frame and invisible at a glance, while on a daily frame it is five days. Positioned by
index, every bar is proportional to the rows it actually holds and any rows no span covers — the
embargo, the purge, the first walk-forward block — appear as empty space exactly where they are. A
gap the user asked for and a gap they did not ask for look the same, which is the honest
rendering: both are rows no model will see. Plotly is several megabytes and is currently reached
only through the lazily-loaded D1 route (§81); using it here would pull it into this page's chunk
to draw rectangles.

**Where.** `frontend/src/components/Timeline.tsx`, `frontend/src/styles/data.css` (`.timeline*`).

---

### 96. A scaled frame lands in `processed_data`, and the plan is a second parent

**Decision.** The scaled frame is a `processed_data` dataset whose lineage records two parents:
the source frame and the split plan. The parameters table hangs off the *scaled frame*, not off
the source.

**Why.** `processed_data` rather than `outputs`, because a scaled frame is still a frame of bars
feeding the next module — exactly like a C5 feature dataset — where `outputs` is for the products
of an analysis (§36). The plan is a genuine second input and is recorded as one: it is not merged
into the numbers, it decides which rows produced them, and `catalog.ancestors` follows the single
parent column rather than the lineage table, so a reader who only follows that chain would never
learn the parameters had been fitted on a subset. The parameters hang off the frame they produced
rather than off the source, because they describe that frame's columns.
`tests/test_lineage.py` asserts all of it from the catalog rather than from the service's return
value.

**Where.** `backend/app/services/scaling.py` (`_compute`), `backend/tests/test_lineage.py`.

---

### 97. A `operation` field on the dataset summary is what keeps plans out of the pickers

**Decision.** `DatasetSummary` and `DatasetMeta` carry `operation: str | None`, read from the
provenance the `datasets` row already holds, and `catalog.derived_datasets(parent_id, operation)`
returns every dataset produced from one parent by one named operation, newest first.

**Why.** A split plan is a table of row indices and a parameter table's columns *are* the numbers.
Both would otherwise appear in C6's and C7's dataset pickers, offering a plan as something to
split and inviting the user to scale `center_` and `scale_`. The backend refuses either on its own
terms — the filter is a convenience, not the guard — but offering them at all is an invitation to
a refusal. The `derived_datasets` lookup goes through the lineage table rather than through
`datasets.parent_dataset_id` because the question is "which *split plans* were made from this
dataset", and the parent column cannot answer it: a regime run and a plan can be outputs of the
same dataset, and the verb lives on the lineage row.

**Where.** `backend/app/storage/catalog.py` (`derived_datasets`),
`backend/app/models/dataset.py`, `frontend/src/pages/{SplitsPage,ScalingPage}.tsx`.

---

### 98. The UI driver's "not implemented" checks became a negative over every entry

**Decision.** `tools/e2e_ui.py` no longer navigates to `/splits` and `/scaling` to assert they say
"not implemented". The sidebar loop now asserts, on the document it has already proved rendered,
that *no* entry says it.

**Why.** Those two checks were correct until C6 and C7 were built, and would then have failed a
correct application — the same class of defect as §68. Deleting them outright was the other option
and was rejected: §13's promise is that an unbuilt route renders an explicit bar rather than a
plausible-looking mock, and that promise has no live instance left, which is exactly when a
regression is easiest to miss. Kept as a negative, a future registry entry that is registered but
not built still fails this driver.

**Where.** `backend/tools/e2e_ui.py` (`sidebar · every entry`).

---

### 99. The C7 refusal is checked against a dataset the driver knows has no plan

**Decision.** The check that the three fitted scalers are unavailable without a plan runs against
a dataset found by asking the server which datasets have no plan
(`dataset_without_a_plan`, via `GET /splits/plans?datasetId=`), not against the driver's base
dataset.

**Why.** Plans outlive the run. C6 runs before C7 in the driver and leaves two plans on the base
dataset, so a check written against the base dataset would find a plan there on the *second* run
of the driver and pass for the wrong reason on the first — that is, the check would have been
measuring leftover state rather than the page. The refusal itself is the load-bearing one: the
reason text has to be on the radio, naming C6, because a control that is merely greyed out teaches
the user nothing about why, and this one is the difference between a backtest that is honest and
one that is not. The assertion is `>= 3` rather than `== 3` because a dataset with no numeric
column correctly has the rolling z-score unavailable too.

**Where.** `backend/tools/e2e_ui.py` (`dataset_without_a_plan`, `C7 · scaling`).

---

### 100. What the first execution of the C6/C7 verification found

**Decision.** The pass is recorded here in full, including the four checks that were wrong, because
each was written to look like a test of the thing it claimed to test and none of them was.

1. **`parquet_store.read_dataset` takes a `Path`, and the two new leakage tests passed it a `str`.**
   `DatasetMeta.storage_path` is a string column and `WrittenDataset.storage_path` is a `Path`; the
   store takes the latter, and `datasets.py` already converts at its own call site. The tests now do
   the same. This is the ordinary cost of the two types coexisting, and it surfaced as an
   `AttributeError` on `str.exists` — loud, immediate, and only reachable by running the tests.

2. **`e2e_scaling.py` aborted itself before reaching the leakage checks.** `page_rows` refuses any
   page where `total > len(rows)` — a guard against silently comparing the first thousand rows of a
   longer frame — and the fit-span check called it with `limit=1` and `limit=TEST_FROM` on a 400-row
   dataset. Both calls hit the guard and `sys.exit(1)`. A single-row helper was added, and the two
   timestamps are now compared as instants rather than as strings, since one comes from pandas and
   the other from the Parquet reader.

3. **The "documented default" check supplied the value it was checking.** The run that was supposed
   to show the server picking a window passed `window=WINDOW` explicitly, making it a repeat of the
   run above rather than a test of the default. It now omits `window` and reads the value back out
   of the returned report, which is the only way the check can fail when the default is wrong.

4. **The "report names the fold" check could not fail for the right reason.** It searched the whole
   page for `fold 1`, and C7's own fold `<select>` contains an option reading exactly that — so it
   would have passed on a page whose report never mentioned the fold. It now reads
   `.report__value` only.

   A fifth, smaller one: `test_scaling.py` carried its own copy of the `center=True` source guard
   that `test_leakage_guards.py` already held. Two copies of a mechanical guard are two places to
   update and a pair that can drift apart with neither test noticing, so the copy was reduced to a
   pointer at the original.

**Why.** Three of the four are the same mistake in different clothes: an assertion that reads a
value it did not obtain from the thing under test. They are recorded because the interesting part is
not that they were wrong — it is that they were *plausible*, and that a green run would have been
taken as evidence for properties nothing was checking.

**Where.** `backend/tests/test_leakage_guards.py`, `backend/tests/test_scaling.py`,
`backend/tools/e2e_scaling.py`, `backend/tools/e2e_ui.py`.

### 101. What the second execution of the C6/C7 verification found

**Decision.** The C6 and C7 drivers were run against a live server for the first time, and the
run found three defects in the *application* — one of them a 500 on the most ordinary input
the tool accepts — plus five in the drivers themselves. All eight are recorded, in that order,
because the split between them is the useful part: §100's lesson was about assertions that
read nothing, and this run's was about a module tested only against the frame shape it
happened to be developed on.

**In the application.**

1. **Every uploaded CSV failed to split, with a 500.** `pandas.read_csv` types nothing, so a
   file whose timestamps look like timestamps is stored — and read back — as a `string`
   column. Only A1's fetched datasets are born with a `datetime64` one, and every unit test
   builds its frame with `conftest.hourly_frame`, so nothing in the suite had ever handed C6
   the shape the upload path actually produces. `_to_row` called `.to_pydatetime()` on a
   `str` and the exception reached the client as a bare `Internal Server Error`.
   `splits._observation_times` now coerces the column once, with `utc=True, errors="coerce"`,
   and refuses an unparseable value — the same coercion, and the same refusal, that `cleaning`
   and `merge` already apply.

   The correctness half matters more than the crash. The monotonicity and uniqueness checks
   ran on the *text*: `is_monotonic_increasing` over strings compares them lexicographically,
   which agrees with the clock only while every row is spelled identically, and the spans
   would then have been contiguous in row terms, ordered, non-overlapping, and covering the
   wrong bars. `_date_span`'s `searchsorted` would have compared a string column to a
   `datetime` and either raised or matched nothing. The coercion is what makes the timeline
   an ordered set of instants before any arithmetic touches it.

2. **A walk-forward plan recorded one more fold than it held.** `splits.create` wrote
   `split_folds` as `max(fold) + 1`. A percentage plan labels its three spans fold 0, so that
   reads 1 — right by accident; a walk-forward plan labels its folds 1..n, so it reads n+1.
   C7 builds its fold selector from that number (`Array.from({length: plan.folds})`), so a
   5-fold plan offered a sixth fold in the dropdown and the run that picked it was refused by
   the server as `Fold 6 is not in '<plan>' (folds 1–5)` — a control the page should never
   have rendered. Now `len({row.fold for row in rows})`, the same expression the report
   already used, so the two representations of "how many folds" cannot disagree again.

   `tests/test_splits.py` asserted the wrong one: `assert summary.folds == 4` for a plan asked
   for with `n_folds=3`. The test was not merely silent about the bug, it held it in place —
   which is why it now reads `assert summary.folds == 3 == summary.parameters["n_folds"]`,
   tying the summary to the request and to the report.

3. **A plan's `rows` was labelled as bars in one place and spans in another.** The field is
   the plan table's own row count — one row per span — which is what C6's *Existing plans*
   table means by it and what C7's dropdown was labelling "rows". C6's *Spans* table on the
   same page uses "Rows" for the bars inside a span, so one word meant two quantities, and
   the dropdown read `· 1 fold(s) · 3 rows` on a plan covering 400 bars. The existing-plans
   column is now **Spans** and the dropdown says **span(s)**; the API field keeps its name.

**In the drivers.** `e2e_ui.py` asked `/splits/plans?datasetId=` where the endpoint takes
`dataset_id`, so the check for a dataset with no plan 422'd and killed the run; its fold check
asserted `== 5` against whichever walk-forward plan the dropdown offered first, which is a
leftover from an earlier run as often as it is the one C6 just made; and `e2e_splits.py`'s
frame carried no `close_time`, so the check in its own module docstring — the one that proves
spans are dated by the bar's open and not its close — raised `KeyError` instead of running.
That check is now executable, and it passes: `start_time` equals the source's `open_time` at
`start_index` and differs from its `close_time` by one bar.

Two more assertions were wrong in the same direction as §100's. The listing check read a
plan's `rows` as the source frame's row count; it now compares against `len(plan)`, the spans
the driver read out of its own 201. And the purge check required the purged rows to appear in
no span *anywhere in the plan*, which is false for an expanding scheme by construction: a row
purged at fold 1's boundary is legitimately inside fold 2's training span, 85 bars upstream of
fold 2's test start. It is now scoped to the fold the rows were cut from, where the property
it was written to test actually holds.

**Why.** A green suite is evidence about the shapes it builds. The unit tests built frames
with a `datetime64` column because that is what the fetch path produces; the split module was
the first thing in the pipeline that had to accept a frame from the *upload* path, and nothing
in 1179 passing tests was pointed at one. That is the same failure as §100 in a different
place: not a wrong assertion, but a missing one — no test asked what C6 does with the data a
user actually gives it.

**Where.** `backend/app/services/splits.py`, `backend/tests/test_splits.py`,
`backend/tools/{e2e_splits,e2e_ui}.py`, `frontend/src/pages/{SplitsPage,ScalingPage}.tsx`.

---

### 102. What the third execution found: the driver's own exhaust, not the application's

The third run passed, and left one line that did not belong in a clean run:

```
4 console error(s):
  - console: Failed to load resource: the server responded with a status of 500
all UI checks passed
```

Nothing in the application returned a 500. `backend/uvicorn.log` is opened `"w"` by
`tools/serve.py`, so it holds only the current run — and it holds no 500 at all. What
`frontend/vite.log` holds is eight `http proxy error: /api/datasets … connect ECONNREFUSED
127.0.0.1:8000`, all stamped `12:32:35 AM`, which is the instant `serve.py` restarted the API
underneath a page that was still open. Those eight became four console errors, and the page
that was still open was not this run's.

**The cause is the driver's own port choice.** `CDP_PORT` is fixed at 9222 and `launch()`
starts the browser with `--remote-debugging-port=9222`; the exit path called
`browser.terminate()`. On Windows that kills the process `Popen` started, which for Edge is a
launcher — the browser proper is a process of its own that survives it, keeps the port, and
stays on whatever page it was last driven to. So a run ended by leaving a browser behind, and
the *next* run's `wait_for_debugger` asked the **port** rather than its own process, adopted
that browser, drove its tab and inherited its console log. A probe found 48 such processes
from earlier runs and a leftover browser sitting on `http://127.0.0.1:5173/regime/...`, which
is exactly the page that polls `/api/health` and `/api/datasets`.

This is worth stating plainly because of what it does to the evidence. Every "clean" run
before this one was clean *including* the previous run's console errors, and nothing said so.
The four errors were not a wrong assertion — the check that prints them is a printout, not a
failure — they were a measurement of the wrong browser.

**The fix, and its bounds.** `close_stale_browser()` runs before `launch()` and closes a
browser on the port only when one of its targets is showing this application; a browser
belonging to something else is left alone and `launch` fails the way it always did, because a
driver that silently drove a stranger's browser would be worse than one that refuses. The
exit path now closes through the protocol (`Browser.close`) rather than the process, falling
back to `terminate()` and then `kill()`, so the run does not become the next run's stale
browser. Measured afterwards: port 9222 free, no `qml-cdp-` browser left, no profile left.

**A second defect it exposed, on the first run of the fixed driver under the full gate.**
`e2e_ui.py` died at the B4 section with `Cannot read properties of undefined (reading
'disabled')`: `page.goto` is fire-and-forget, and that section waited only for `.field` — the
dataset picker, which the page renders as soon as it mounts — while the *Preview* button
every check below it indexes is rendered once the dataset list has arrived. So the wait
returned on a page whose actions did not exist yet and the next line threw.

This had been latent for as long as the driver drove a reused browser, and it is the same
mechanism as the first defect rather than a coincidence: a browser left over from the previous
run holds the application's modules in its HTTP cache, so navigating within it is served
locally and the page mounts before the wait gives up. The race only needs a browser that has
never seen the app — which is what the fix produced — immediately after `serve.py` has
restarted both processes, which is what the gate does. The wait now covers both the field and
the button, which is what every other section already did: `.policy`, `.feature__row`,
`#split-dataset`, `#scaling-plan` and `#regime-dataset` are all waited on by name. B4 was the
one section waiting on a shell element, so this is a fix to the wait rather than a retry
around the throw.

**Two smaller leaks, found by checking the same claim.** `tempfile.mkdtemp` never removes
its directory, so each run also left a browser profile behind — 23 of them, 5.0 GB, by the
time this was measured. They are now deleted on exit, retried rather than attempted once: a
single `shutil.rmtree(..., ignore_errors=True)` after `browser.wait()` removed 129 files and
reported success, because `wait` returns when the launcher exits and Edge's renderer and
utility children release their handles a moment later. And the last two console errors, the
ones that survived the browser fix, were the page's own default `/favicon.ico` request
404ing — `index.html` declared no icon and there is no `public/` directory. It now declares an
inline SVG one, so no request is made. The gate is now silent on both counts: no console
errors, no refusals, no leftover browser, no leftover profile.

**Why.** §100 and §101 were about checks that passed on wrong behaviour. This is the step
before that: a check that reports on the wrong subject, and — in the B4 race — a check whose
subject did not exist yet. Both were invisible for the same reason: the harness had no test
of its own hygiene, and the one line that would have shown the first was read as application
noise for two runs. The rule the three entries now share is that a driver's *environment* is
part of its evidence: if it does not control which browser it drives, or wait for the control
it is about to press, nothing it prints about either means anything.

**Where.** `backend/tools/e2e_ui.py`, `frontend/index.html`.

---

## Part E — Volatility forecasting

### 103. Module E is a service and a router, and it closes a dependency gap

**Decision.** The brief's `backend/app/modules/volatility/{router,service,schemas,models,evaluation}.py`
is built as `app/routers/volatility.py`, `app/services/volatility.py` (the engine and the job),
`volatility_models.py` (the five models behind one protocol), `volatility_eval.py` (the metrics)
and `app/models/volatility.py` (the schemas). The evaluation functions are separated from the
engine because they are pure functions over arrays and their tests are hand-calculated. `xgboost`
and `scikit-learn` are added to `pyproject.toml`, where neither was declared before.

**Why.** §84 established the shape: there is no `app/modules/` in this codebase, and D is a
service plus a router. The dependency gap is the part that would have failed on another machine.
Module D's hybrid already imported `xgboost` and Module E's TDNN leg imports `scikit-learn`;
both were installed here, neither was declared, so a fresh install would have raised
`ImportError` from inside a background job — which the user sees as a failed run, not as a
missing dependency.

**Where.** `backend/app/{routers,services,models}/volatility*.py`, `backend/pyproject.toml`.

---

### 104. Two forecast objects, and only one of them is an observation

**Decision.** The report carries **two** distinct forecasts, each labelled. The *scored* forecast
is one-step-ahead at each walk-forward position, and it is what MQLIKE, the Mincer–Zarnowitz
regression and Diebold–Mariano are defined over. The *forward path* is `forecast_horizon` steps
past the last observation, and it is what the fan chart draws. Only the scored span is
persisted: `VOL_FORECAST_COLUMNS` has no `is_forward_path` column, because a constant-false one
would carry no information.

**Why.** The brief's evaluation section assumes a one-step forecast and its UI section assumes a
multi-step one; they are different objects. Scoring a multi-step projection with MQLIKE against
the one-step actual is a category error — there is no realised value at that horizon to score it
against. And a dataset row in this application is an observation of something that happened:
writing a projection into the store would make a forecast indistinguishable from a fact to every
consumer downstream. The projection therefore lives in the job's report, which is where a claim
about the future belongs and where it cannot be mistaken for a record of the past.

**Where.** `backend/app/services/volatility.py` (`_frame`, `VOL_FORECAST_COLUMNS`),
`frontend/src/pages/VolatilityPage.tsx`.

---

### 105. The observation column is `open_time`, and Phase 0 repaired a mislabelled one

**Decision.** The conditioning join and the output schema use `open_time`. Before Module E
existed, `regime_engine.py` built its frame with a column **named** `close_time` that **held**
the observation instant; it is now named after what it holds. Its three consumers
(`RegimePage.tsx`, `tools/e2e_regime.py`'s expected-column list and row lookups) follow the
rename, and `tests/test_leakage_guards.py` asserts the string appears nowhere in the module —
the same mechanical technique that file already uses for `datasets.py`, `splits.py` and
`scaling.py`.

**Why.** §14 and §86 fix the rule: `close_time` is the instant a bar *finished*. A forecast made
at bar `t` is made from information available at the open of `t`, and joining on `close_time`
aligns it with the bar it predicts — off by one bar, with no error and a chart that looks
plausible. The repair had to land before the first conditioning join existed, because that join
is precisely the operation the wrong name silently corrupts. This is the third time a brief has
inverted the timestamp rule (§§14, 86) and the first time the error had already reached the
code.

**Where.** `backend/app/services/regime_engine.py`, `frontend/src/pages/RegimePage.tsx`,
`backend/tools/e2e_regime.py`, `backend/tests/test_leakage_guards.py`.

---

### 106. A NaN is counted, never filled

**Decision.** Every written row carries a `finite` flag. Every model's score carries `n_finite`,
`n_steps`, `n_nan` and `n_inf`, and the metrics table prints them. Nothing is interpolated,
forward-filled or dropped, and a model with too few finite steps gets no metric at all rather
than a metric computed over whatever survived.

**Why.** Priority order: transparent calculations. A model that produced 400 NaNs and was
forward-filled produces a chart indistinguishable from one that did not, and the difference is
exactly what a reader needs in order to decide whether to believe the number beside it. The
`finite` column also makes the output frame self-describing: a consumer can tell which rows were
scored without reading the report.

**Where.** `backend/app/services/volatility.py` (`_frame`, `_score`),
`frontend/src/pages/VolatilityPage.tsx` (`MetricsTable`).

---

### 107. What is refused at the door, and what is refused from inside the job

**Decision.** A dataset that does not exist is a **404** on the `POST`, and the pydantic request
validator's rules — no models, a duplicate model, `o=0` for an asymmetric model, a blank price
source, `train_window < 50`, `test_window < 30`, `step_size` out of range, `forecast_horizon >
500`, `lags > 100` — are **422** at the door. Everything else surfaces through the job's `error`
field as a `ValidationFailure`, with the computed numbers in the message: an impossible
walk-forward schedule, a non-stationary return series, an unknown price column. Nothing returns
**400** (§85: `app/core/errors.py` maps nothing to it).

The schedule is validated against `series.returns.size`, never against `row_count`.

**Why.** The row count is not the series the walk-forward runs over. Returns derived from close
are one bar shorter than the frame — the first difference has no predecessor — and the usable
span can start later still when the series opens with a hole. Validating against `row_count`
would accept schedules that cannot run and refuse schedules that can, and it would do both on
the `POST`, before the file has been read. So the schedule is checked where the array is: the
refusal is a job failure whose message names the two numbers that disagree.

**Where.** `backend/app/models/volatility.py` (`WalkForwardSpec`, `VolForecastRequest`),
`backend/app/services/volatility.py` (`_validate_schedule`, `_execute`).

---

### 108. arch is fitted on percentage returns, and the rescaling is stated

**Decision.** The return series is multiplied by 100 before it reaches arch and the resulting
variance is divided by 10 000 on the way out. The factor is `scale_factor` in the report, and the
page prints it with the reason.

**Why.** arch's optimiser is poorly conditioned at a return scale of 1e-4 — its own documentation
and every worked example scale the series — and the transform is exact in both directions, so it
changes the conditioning and not the model. What matters is that it is *stated*: a scale factor
that is applied silently is a modelling choice the user cannot see, and the round trip is
asserted by a test rather than assumed, because a factor applied on the way in and not on the way
out is a variance that is wrong by four orders of magnitude and looks like a working model.

**Where.** `backend/app/services/volatility_models.py` (`SCALE`),
`backend/tests/test_volatility_models.py` (the scale round trip).

---

### 109. The hybrids' features are lagged, and their correction is clamped and reported

**Decision.** Both hybrid legs consume only *lagged* returns, lagged absolute returns, and lagged
conditioning columns. The delay line is built by `delayed_design`, whose lag enumeration starts at
1 and whose rows are described as reading `j-lags .. j-1`; there is no offset parameter and no
default that could shift it, so an un-lagged feature is not something a caller can express. The
combination is `σ̂²_{t+1} = σ²_{arch,t+1} · ẑ̂²_{t+1}`, with the learned correction clamped to
`CORRECTION_CLAMP = (0.1, 10.0)`. The number of steps that hit the clamp is counted and, when it
is non-zero, reported as a warning in the report.

**Why.** A same-bar return in an XGBoost feature matrix is the target, not a predictor: putting
one there produces a model with a very high in-sample R² that has learned the answer it is being
asked for, and nothing in the output frame shows it. Making the shift structural is a stronger
guard than refusing it at runtime, because there is no code path that could express the mistake —
and the guard is completed from both ends, by a source assertion on the lag enumeration (a loop
written `range(0, lags)` would put the current return beside its own target) and by a behavioural
test that changes one bar and requires every row at or before it to come back bit-identical,
which is run on the exogenous block as well, since a regime posterior concatenated unshifted
would read today's regime to predict today's volatility.
The clamp exists because `E[z²] = 1` by construction, so an unclamped product can wander by an
order of magnitude on a step where the correction is extrapolating — and a bound that is counted
and printed is a disclosed limitation, where one applied silently is a hidden model.

**Where.** `backend/app/services/volatility_models.py` (`delayed_design`, `CORRECTION_CLAMP`,
`correction_clamped`), `backend/tests/test_leakage_guards.py` (the lag enumeration, the change-one-
bar experiment, and the same experiment on the exogenous block),
`backend/app/services/volatility.py` (the clamp warning).

---

### 110. Diebold–Mariano is hand-rolled, with the small-sample correction

**Decision.** `volatility_eval.diebold_mariano` computes the loss differential, its long-run
variance by a Newey–West Bartlett kernel truncated at `horizon − 1` lags, the Harvey–Laybourne–
Newbold small-sample factor, and `p = erfc(|t| / √2)`. It is compared on the bars where **both**
models produced a finite forecast.

**Why.** statsmodels has no DM test, and the two-line "t-test on the loss differential" that
fills the gap is the usual way this gets done wrong: without the HAC variance the statistic
ignores the autocorrelation the loss differentials always have, so it over-rejects and reports
differences that are noise — which is the failure mode that matters here, because two volatility
models on a few hundred bars usually *are* indistinguishable. Comparing on the intersection is
the second half of the same point: a "difference" computed on two different samples is partly a
difference between the samples.

**Where.** `backend/app/services/volatility_eval.py` (`diebold_mariano`),
`backend/tests/test_volatility_eval.py` (the hand-calculated case).

---

### 111. The realised proxy is the squared log return, and the report says what that costs

**Decision.** `ACTUAL_PROXY = "squared log return"`, stated in the report and on the page, and
printed together with the consequence: a Mincer–Zarnowitz R² near zero is the expected result for
a correctly specified model against this proxy, not evidence of a broken one.

**Why.** There is no realised variance in the data this application holds — no intraday
high-frequency series to integrate, no option-implied surface. The squared return is the standard
noisy proxy for it, and its noise is *unpredictable by construction*, so an R² near zero is what
a good model looks like here. A reader who has not been told that will read the number as a
verdict on the model, and will be wrong in a direction that discards a working estimator. The
page therefore ranks on MQLIKE, which is robust to the proxy, and labels the R² as a calibration
diagnostic.

**Where.** `backend/app/services/volatility.py` (`ACTUAL_PROXY`),
`frontend/src/pages/VolatilityPage.tsx` (the caveat under the metrics table).

---

### 112. The fan is the model's own error, not an assumed distribution

**Decision.** The bands are the empirical quantiles of the walk-forward **forecast-error ratio**
for that model, at 5/25/50/75/95, drawn as two nested bands and labelled `5–95%` and `25–75%`.
The legend says whose error they are.

**Why.** The alternative — deriving a band from the fitted conditional distribution — is
inventing a distributional assumption the user did not ask for, and it would be a *narrower*
band than the model has earned, because it ignores parameter uncertainty and the model's own
demonstrated bias. The empirical quantiles answer the question a reader actually has: how wrong
has this model been, at this horizon, on this series. That the bands are wide is the honest
result, not a defect.

**Where.** `backend/app/services/volatility_eval.py` (`forecast_error_quantiles`, `FAN_BANDS`),
`frontend/src/pages/VolatilityPage.tsx` (`buildFan`).

---

### 113. The unscored message names the sample, not the optimiser

**Decision.** When a model has fewer than `MIN_SCORED_STEPS` finite forecasts, the reason has two
branches. If every step produced a finite forecast and there were no failures, it says so — "All
10 steps produced a finite forecast, but a metric needs at least 30 to be an estimate of
anything. Widen the test window, or shorten the step size" — and otherwise it reports how many
of how many succeeded plus the first failure.

**Why.** The single message it replaced read "Only 10 of 10 steps produced a finite forecast",
which is not a sentence, and it put the blame on the optimiser for what is a sample-size
problem. The reader it misled is the one who would then distrust a working estimator and go
looking for a defect in the fit. The distinction the two branches draw is between a model that
failed and a test window too narrow to score — a user-facing honesty defect of exactly the kind
priority order puts first, and one that only exists because the two cases were originally the
same code path.

**Where.** `backend/app/services/volatility.py` (`_score`),
`backend/tools/e2e_volatility.py` (asserted on a short protocol with a robust estimator, so the
message cannot be satisfied by the failure branch).

---

### 114. The simulated forward path is seeded, because arch's simulation is not

**Decision.** A multi-step path for EGARCH or GJR is drawn by simulation — analytic multi-step
forecasting is not available for either — with `simulations=1000`, and the fitted model's
distribution is replaced by a freshly seeded instance immediately before the draw. `SEED` is
fixed at 20240101.

**Why.** A path drawn from a stream that started wherever the process happened to be gives a
different answer on every run, which makes the reproducibility gate unpassable and makes two
runs of the same request look like two different results. arch's own `random_state` argument is
read by the bootstrap method and not by simulation, so fixing the seed at fit time is not
enough: the generator is a stream, and a second call would continue it. The instance is
therefore rebuilt at the moment of the draw. A one-step-ahead forecast is analytic and does not
depend on this at all, which is why the determinism test asserts on the *forward path* — it is
the only part of the output where a shock is drawn.

**Where.** `backend/app/services/volatility_models.py` (`ARCH_DISTRIBUTION_CLASS`,
`_seeded_distribution`, `forecast_variance`),
`backend/tools/e2e_volatility.py` (two runs compared on `json.dumps(forward_paths)`).

---

### 115. The default schedule is sized against the returns, not the rows

**Decision.** `/volatility/options` reports `usable_returns = row_count − 1` and derives
`train_window` and `test_window` from *that* number, so the form opens on a schedule the dataset
can run. The `feasible` flag and the `too_short` warning carry both counts.

**Why.** The off-by-one is not cosmetic. Log returns derived from close are one bar shorter than
the frame, so a default sized to `row_count` exceeds the series by exactly that bar on every
dataset short enough to reach the `usable // 2` branch — the form would open on its own refusal
message, and the user would read that as the module being broken rather than as the default being
wrong. The numbers are in the warning for the same reason they are in the refusal: a message
that says "only 60 bars" beside a schedule needing 80 invites the reader to count the dataset's
rows and find 60, while the walk-forward is over 59.

**Where.** `backend/app/services/volatility.py` (`options`, `_option_warnings`),
`frontend/src/pages/VolatilityPage.tsx` (the live arithmetic, which uses the server's `limits`).

---

### 116. Which metric is a ranking, and which direction is better, come from the server

**Decision.** `VOL_METRIC_RANKING` and `VOL_METRIC_LOWER_IS_BETTER` live in
`app/models/enums.py`, travel in the options payload as `is_ranking` and `lower_is_better`, and
are the only source the page uses to mark a best cell and to print the ↑/↓ on a column header.
The Mincer–Zarnowitz R² is `is_ranking: false`, and no cell in its column is ever crowned.

**Why.** Marking the highest MQLIKE as best is a one-character mistake — the sign of one
comparison — that inverts the conclusion of the entire report, and it is a mistake that would
otherwise be available to be made twice, independently, in the engine and in the page. The R² is
excluded because it is a calibration diagnostic with no ranking direction, and "best R²" beside
four models is a comparison that means nothing.

**Where.** `backend/app/models/enums.py` (`VOL_METRIC_RANKING`, `VOL_METRIC_LOWER_IS_BETTER`),
`backend/app/services/volatility.py` (`_metric_options`),
`frontend/src/pages/VolatilityPage.tsx` (`MetricsTable`).

---

### 117. A refused request is answered in the application's envelope, whoever refused it

**Decision.** `app/core/errors.py` registers a `RequestValidationError` handler beside the
`QuantmetricsError` one. Every 422 now leaves as `{"error": {"code": "validation_failure",
"message": …, "details": {"errors": [...], "count": n}}}`, whether the refusal came from a
service or from a pydantic validator.

**Why.** FastAPI answers a validator failure with `{"detail": [...]}` unless something says
otherwise, and four of this project's request models refuse in a `model_validator`:
`regime.py`, `scaling.py`, `splits.py` and `volatility.py`. So the envelope was missing on the
path most malformed requests actually take — and `src/api/client.ts` builds its `ApiError` from
`error.code` and renders `error.message`, which means the page degraded to a generic failure for
precisely the refusals written to be specific. The five Module E tests that assert
`response.json()["error"]["code"]` were all failing on `KeyError: 'error'` before this landed.

Two details are decisions rather than plumbing. The first error's location is folded into the
message (`"test_window: Input should be greater than or equal to 50"`), because the envelope has
one message field and naming the field is the actionable half of a refusal; the structured list
survives in `details.errors` for a caller that wants to mark several fields at once. And that
list is rebuilt field by field rather than passed through: pydantic puts the original exception
object under `ctx`, and the payload is rendered with `allow_nan=False`, which raises on anything
the encoder cannot represent — the error envelope failing to serialise would turn a 422 into a
500.

**Where.** `backend/app/core/errors.py` (`_handle_request_validation`),
`backend/tests/test_volatility_api.py::test_an_invalid_request_is_refused_with_422`.

---

### 118. A diverged fit is a step, not a value

**Decision.** The walk-forward refuses a one-step variance forecast that is non-finite, not
strictly positive, more than `MAX_FORECAST_VARIANCE_MULTIPLE` (**100**) times the largest
squared return in its own training window, or less than `MIN_FORECAST_VARIANCE_RATIO`
(**1e-30**) times that window's **mean** squared return. A refused step is recorded as a
failure, counted as non-finite, never scored, never written to the output frame as a number,
and left as a gap in the forward path. The count and the first reason reach the report.

**Why, the ceiling.** The bound is relative because a fixed ceiling cannot be right: `1e-6` is enormous for
daily decimal returns and negligible for annualised basis points, and the same dataset can
arrive either way — the window's own largest squared return is the only yardstick that travels
with the data. A hundred is a hundred because of how these recursions work: GARCH's forecast is
a weighted average of past squared returns and lies *inside* the window's range, GJR's adds an
asymmetric term to the same recursion, and EGARCH's one-step factor `exp(α(|z| − E|z|) + γz)` is
single-digit for parameters near the optimum. Measured over thirty windows of this project's own
fixture, the ratio of forecast to window-maximum squared return is: GARCH ≤ 0.36, GJR ≤ 0.38,
EGARCH median 0.2 with its largest legitimate value at 15, and then a gap to **3 495** for a
window whose EGARCH fit pinned β = 0.999. The bound sits above the former and below the latter.

**Why, the floor, and why it is the *mean*.** The ceiling alone left the other tail open, and
the other tail is not the mirror image of it: a recursion does not merely overshoot, it
*underflows*. The same measurement that fixed the ceiling gives the legitimate range on the
low side — across GARCH, GJR and (on a window long enough to identify it) EGARCH, a one-step
forecast lands between **0.35 and 7.9** times its own window's mean squared return, a spread of
well under two orders of magnitude centred on one, which is what a recursion anchored to its
sample should produce. Thirty orders is therefore not a threshold separating large from small;
nothing legitimate comes within twenty-five orders of it. It is the line past which the value is
a floating-point boundary rather than an estimate, and the measurement put the breakdown far
beyond it: unidentified EGARCH fits returned `1e-190` and `3e-39` on 100- and 140-bar windows of
a series whose identified range is two orders wide, and the XGBoost hybrid on the driver's
fixture returned `1e-127` while its two companions returned MQLIKEs of 0.88 and 0.91 — a model
that looks scored, and is, on a forecast that says the market's variance is about to vanish. The
reference is the window's **mean** squared return rather than its minimum, because a minimum is
one flat bar away from being meaningless: a single zero return makes any comparison against it
vacuous, while the mean is exactly the quantity a correctly-anchored recursion stays near. A
tighter floor would risk refusing a genuinely quiet forecast after a regime shift; this one only
ever fires on a recursion that has collapsed.

**Why, the cost.** What the runaway fit costs downstream is the argument. One such step — arch produced `0.0` from
one window and `8.28e213` from another, both with `convergence_flag = 0`, because that flag is
scipy's optimiser status and says only that the solver stopped — turned a run's headline MQLIKE
into `4.7e293`, made its RMSE `inf` (which cannot even leave the process: `allow_nan=False` on
the response, and a browser's `JSON.parse` throws on `Infinity`), wrote a variance of `8.3e213`
into the saved dataset as though it were an observation, and emptied that model's comparison row
without a word. The collapsed tail costs the same thing from the other end: the metrics are
computed on it without complaint, and the number a reader takes away is one the data never
supported.

The forward path is refused by the same rule and for a sharper reason: it is what the fan chart
draws as its headline. A diverged final fit would set the chart's axis at `8e213` and flatten
every band and the realised line onto zero — a picture that reads as a rendering bug and is
really a fit that left the parameter space. A collapsed one does the opposite and is harder to
notice: the whole fan sits on the axis and reads as a very confident forecast of calm.

When a model is unscored *because* a fit diverged, the reason ends with the likeliest fix — a
longer `train_window` — which is the difference between a report that says "broken" and one that
says what to change. An exception from arch does not get that sentence, because a wider window is
not obviously the answer to it.

**Where.** `backend/app/services/volatility.py` (`_forecast_refusal`,
`MAX_FORECAST_VARIANCE_MULTIPLE`, `MIN_FORECAST_VARIANCE_RATIO`, `FORECAST_REFUSAL_TAG`,
`_walk_forward`, `_forward_path`, `_score`), `backend/tests/test_volatility_models.py`
(`test_a_diverged_forecast_is_refused_with_the_numbers`,
`test_a_collapsed_forecast_is_refused_too`,
`test_a_diverged_step_is_counted_and_never_scored`),
`backend/tests/test_volatility_api.py::test_a_diverged_fit_is_refused_end_to_end_and_never_reaches_the_frame`.

---

### 119. The metrics drop a zero forecast rather than flooring it

**Decision.** `volatility_eval._aligned` keeps only the pairs whose forecast is finite **and
strictly positive**, for every metric alike. `rmse` and `mae` compute the error scaled by the
largest magnitude present, so a forecast too large to square returns a finite number instead of
`inf`.

**Why.** The previous `np.maximum(forecast, 1e-300)` floor did not remove the infinity QLIKE
reaches at a zero forecast — it moved it. The ratio becomes `1e300` and that step's loss
`1e295`, which is larger than every other step's loss put together by hundreds of orders of
magnitude; the mean then describes the floor rather than the run. That is not hypothetical: it
is where the `4.7e293` MQLIKE above came from. A variance forecast of exactly zero is an
optimiser at a boundary or a recursion that underflowed, and no loss value expresses "predicted
zero" against a realised variance that is positive by definition, so the pair is dropped — and
the *service* counts it as a step that produced no forecast, which is where the user is told.
Applying it to all four metrics rather than only the ratio-based ones keeps a model's MQLIKE and
its RMSE on the same pairs; scoring them on different samples puts two sample sizes in one row
with nothing on the page saying so.

The scaling is exact rather than approximate wherever the direct formula is representable, which
is why the ordinary cases are still asserted against hand-computed values.

**Where.** `backend/app/services/volatility_eval.py` (`_aligned`, `_scaled_errors`, `rmse`,
`mae`), `backend/tests/test_volatility_eval.py`
(`test_a_zero_forecast_is_dropped_rather_than_floored`, `test_every_metric_drops_the_same_pairs`,
`test_the_error_metrics_stay_finite_for_a_forecast_too_large_to_square`).

---

### 120. A comparison that could not be run is reported with both names in it

**Decision.** `_compare` returns the pairs it tested *and* a sentence per pair it could not,
naming both models and the shared step count against the `MIN_SCORED_STEPS` floor. The gaps go
into the report's warnings.

**Why.** It used to `continue`, so a two-model run could come back with `comparisons: []` and
nothing anywhere saying why — indistinguishable, to a reader, from a comparison that found the
models equal, which is the opposite of what happened. That is exactly the case a short window
with one diverged model lands in, and it is the case where the reader most needs the verdict
explained rather than omitted.

**Where.** `backend/app/services/volatility.py` (`_Comparisons`, `_compare`, `_report`),
`backend/tests/test_volatility_models.py::test_a_pair_with_too_little_overlap_is_reported_rather_than_dropped`.

---

### 121. The fixture clusters, and its train window is longer than the module's floor

**Decision.** `tests/test_volatility_api.py::price_frame` generates returns from a GARCH(1,1)
recursion (α = 0.12, β = 0.80) rather than i.i.d. normals, and `FAST` runs a 180-bar train
window instead of the module's `MIN_TRAIN_WINDOW` of 50.

**Why.** Two different mistakes, one fixture. On an i.i.d. series a GARCH has *nothing* to
estimate — the ARCH and GARCH terms trade off along a ridge of equally-likely values — so the
first version of this fixture put the optimiser on the ridge and made divergence the ordinary
outcome rather than the exception; clustering is also what real return series do, and a
walk-forward over one asks the models the question they exist to answer. The train window is the
second half: at 50 bars EGARCH's β pins at ≈ 0.999 on this fixture, a random walk in
log-variance the likelihood cannot identify from 50 points, which the new guard correctly
refuses — leaving a fixture in which the *ordinary* path diverges and therefore cannot also be
the fixture that checks the ordinary path's arithmetic. 180 bars is the shortest window on which
all five models fit cleanly on all thirty steps, and it is in the same range as the page's own
derived default, which is half the series.

This is a statement about the module's limits and not just about its tests: `MIN_TRAIN_WINDOW`
is a floor on what may be *attempted*, not a guarantee that a four-parameter asymmetric
recursion is identified on it. The refusal and the widening hint are what the user gets.

**Where.** `backend/tests/test_volatility_api.py` (`price_frame`, `FAST`).

---

### 122. What the Module E verification found

**Decision.** Recorded, because the pattern recurs: the module's own tests were where the last
several defects were, not the module.

- **`WalkForwardSpec.n_scored` never reached the page.** It was a plain `@property`, and
  pydantic does not serialise a property — the report embedded the spec, so the field the page
  reads arrived as `undefined`. It is a `computed_field` now. This is the failure mode the
  frontend's hand-written types cannot catch: a renamed or unserialised JSON field surfaces in
  the browser, not at compile time, which is why `test_volatility_api.py` asserts every field by
  name.
- **The DataGrid's `n_scored` beside the schedule** is a live hint in `VolatilityPage.tsx`; a
  serialisation change is a page change whether or not the page changed.
- **`np.allclose(before[11:], after[11:], "the change never propagated forwards")`** passed the
  sentence as `rtol`. numpy raised `TypeError: ufunc 'isfinite' not supported for the input
  types` from inside its own comparison, so the leakage test failed for a reason that had nothing
  to do with leakage — and would have *kept* passing had the array been smaller.
- **The causality tests compared `nan == nan`.** Once a diverged step is refused, both runs can
  legitimately record no forecast at the same bar, and `pytest.approx` fails on two NaNs.
  Agreement on "unusable" is a causality property like any other — the refusal depends only on
  the training window's content and the fitted value, both identical for a shared origin — so the
  comparison handles it explicitly, and one run refusing where the other did not still fails.
- **`linspace` is not binary-exact**, which cost a run: `a − (a + 0.25)` on `linspace` endpoints
  carries a jitter of order `1e-17`, whose variance is tiny but *positive*, so a test written for
  the zero-variance branch took the ordinary one and reported a statistic near `1e17`.
- **The report dated the scored span one bar early.** `_report` sliced `series.times` — which is
  already indexed in return space, so position `j` of it is position `j` of the schedule — and
  then subtracted `series.offset` as well, moving every reported timestamp one bar back whenever
  the returns had to be derived from close, which is always unless the frame already carries a
  return column. `_output_frame` adds the offset to the run's indices and subtracts it again,
  which is why the saved frame was right and only the report was wrong: a disagreement between
  two renderings of one walk-forward, invisible to every check that reads one of the two alone.
  The driver now ties them — the frame's per-model non-finite rows and the report's
  `n_nan + n_inf` must be the same number — because that is the gap the defect lived in.
- **EGARCH is not identified on a short window, and its failure has both tails.** Measured while
  choosing the driver's fixtures: on the four-hour frame at a 100-bar train window, the ratio of
  the EGARCH leg's one-step forecast to its own window's mean squared return spanned **`1.6e-130`
  to `3.4e+9`** across the forty scored steps — the same model, in the same run, eleven orders
  above the ceiling and a hundred and twenty-five orders below the floor. On the hourly upload the
  collapse persists through a 220-bar window and clears at 400. Both driver runs had been sitting
  inside that band, so their strongest assertions — "no scored step came back non-finite", "every
  model is scored over every step" — were failing for a reason about the fixture rather than the
  engine, which is exactly the failure a driver must not be able to have. The fixtures moved to
  the shortest measured windows on which all five models estimate: 180 bars on the four-hour
  frame, 400 on the hourly upload. It is the finding §121 records for the unit fixture, at a
  different length, and the reason the lower bound exists at all.

**Where.** `backend/app/models/volatility.py` (`n_scored`), `backend/tests/test_leakage_guards.py`,
`backend/tests/test_volatility_models.py` (`assert_prefixes_agree`,
`test_a_forecast_does_not_change_when_the_future_is_appended`),
`backend/tests/test_volatility_eval.py`,
`backend/tools/e2e_volatility.py` (`TRAIN`, the determinism schedule, the identity and
forward-path checks).

---

### 123. `annualized_vol` takes a duration window; the annualisation factor stays √(bars per year)

**Decision.** `annualized_vol` is now a duration-windowed feature (15m/1h/4h/1d/7d/21d/30d,
default 1d) rather than a bar-count one (20/30/60/90). This makes it match its formula
predecessor `realized_vol`, which already offered the same durations. The annualisation factor
is unchanged: `realized_vol(w) · √(bars per year)`, derived from the dataset's own declared
timeframe and asset class and written into the report (§45). The interval names the *window*;
it does not name the *factor*.

**Why not the literal request.** The request specified `AnnVol = σ · K` where `K` is the bars
per year for that interval, applied **linearly**, and with `K` varying by interval (crypto
hourly `K = 365·24`; "multi-day 7d: `K = 365/7`"). Both points are wrong, and both would have
produced a number that overstates volatility and could not be compared across datasets:

- **The multiplier is the square root, not the factor itself.** Volatility is a standard
  deviation, and standard deviations scale with the square root of the span, not with the span.
  Multiplying the per-bar σ by `bars/year` instead of `√(bars/year)` overstates the annualised
  figure by a factor of `√K` — about 19× for daily crypto, 94× for hourly. `features.py:486`
  already multiplies by `per_year ** 0.5` for exactly this reason.
- **The factor is a property of the bar length, not of the window.** `K = 365/7` for a 7d
  window treats the window as a period and says there are 52 of them a year. But the sample
  standard deviation of the returns inside a 7-day window of hourly bars is the *within-window*
  dispersion of 168 returns, not a "per-7-day volatility." There is no standard that
  annualises that way, and no reason the 7d value should differ from the 30d value. The
  correct factor is one number per dataset — the bars a year holds at its own bar length — and
  is the same for every window.

The request's annualisation test was self-contradictory in the same way: a *constant* 1% daily
return has sample standard deviation 0, so its annualised volatility is `0 · K = 0`, not
`0.01 · 365`. To reach `0.01 · K` the returns would have to have std `0.01`, which a constant
series does not. The added test pins the absolute factor the honest way — the engine's own
`realized_vol` column times `√365` (daily crypto) or `√252` (daily equity) — and the
point-in-time guarantee the request asks for is already carried, more broadly, by
`test_every_feature_is_point_in_time`.

**Consequence.** The old bar-count windows are no longer offered as ticks, but are still
expressible: the custom box accepts a duration, and "60h" on an hourly dataset is 60 bars. A
sub-day interval on a longer bar is refused rather than rounded — 15m on a daily dataset is
"not a whole number of 1d bars" — for the same reason §44 refuses it for `realized_vol`.

**Where.** `backend/app/services/feature_catalog.py` (`annualized_vol` spec, now `_duration`),
`backend/tests/test_features_compute.py`
(`test_annualized_volatility_pins_the_square_root_of_the_annualisation_factor`, and the two
existing factor tests re-windowed), `backend/tests/test_features_catalog.py`
(`test_a_sub_day_interval_is_refused_on_a_daily_dataset`).

---

## Resolved during implementation

Each of these was an assumption that turned out to need checking. Recorded so the
correction is visible rather than quietly absorbed.

1. **Yahoo futures fields** — *resolved.* Neither settlement price nor historical open
   interest is served for futures contracts. Chosen response: §2 (null columns with
   provenance).
2. **Binance endpoint shape** — *resolved, and the original assumption was wrong.* The
   specification's funding and metrics column names are not REST field names at all;
   they are the header rows of the `data.binance.vision` public dumps. See §4. This
   changed the transport for two of the three crypto datasets.
3. **`duckdb` / `yfinance` on Python 3.14** — *resolved.* Both install cleanly from
   wheels on CPython 3.14.3 (`duckdb` 1.5.6, `yfinance` 1.7.0). No build toolchain
   needed; the PyArrow-only fallback in the plan is not required.

## Still open

- Whether Yahoo's intraday equity endpoints honour requests for 1h and 4h bars over
  long spans, or silently return only the most recent ~730 days. Yahoo has historically
  restricted intraday history without documenting a limit. The fetcher compares the
  returned range against the requested range and raises `SpanTooWide` on a short result
  rather than storing less data than was asked for (§7).

- **Uploaded CSVs have no declared time range** (§24). `start_timestamp` and
  `end_timestamp` are null for every upload, so the Data Management panel reads "Start —"
  and the explorer cannot show a span. Deliberate: a CSV carries no timezone, and both
  available assumptions are ones this project refuses to make silently. The missing piece
  is a way for the user to *declare* the zone on an upload; until that exists, the honest
  output is a null.

- **`frontend/src/pages/Pending.tsx` is now unreferenced.** C6 and C7 were the last two routes
  rendering it, and `App.tsx` routes to the real pages instead (§13's list is now empty). It is
  kept rather than deleted: it *is* the mechanism §13's promise is made of, and the next route
  that is registered before it is built needs exactly this component. Deleting it would leave the
  promise with no implementation, which is the state in which someone substitutes a mock. The
  negative check in §98 still fails the driver if any route renders it.

- **A failed fetch leaves no run record.** `record_run` is called only after the dataset
  is committed (§ A1), so `GET /fetch/runs` shows successes and nothing else — a fetch
  that raised appears to have never been attempted. Left as is because no page surfaces
  the run log yet and a runs view is outside Part A; the endpoint's shape is settled and
  the fix is a `try/except` around the fetch, which should land with the page that reads
  it rather than before it.

- **The UI has been clicked through, in one browser, headless.** `tools/e2e_ui.py` drives Edge
  over the DevTools Protocol and presses every sidebar entry; the fetch form's market, data-type
  and timeframe controls; the chart's overlay toggle; on the grid, both pager arrows, the
  rows-per-page select, a header click to sort and the *Clear sort* that unlocks editing, the
  header menu's four items, the cell menu's *Clear cell*, *Show all columns*, *Add column*,
  *Discard changes* and *Save as new dataset*; both B pages and their run buttons; the C5
  page's grouping, info icon, ticks, select-all, typed custom window and both Run buttons; and
  the C6 and C7 pages, where it generates a percentage plan and a walk-forward one, reads the
  spans and the timeline back, drives a date split into its client-side refusal, then scales
  against the plan it just made — through the 202/poll handshake — and checks the refusal on a
  dataset with no plan. It is what found both of the defects recorded in §63 and §65 — a data
  grid that had never initialised in a browser at all, and a refusal the driver was reporting as
  a broken page. The driver itself is held to that standard too: eleven of its own expectations
  were wrong on a correct application and were corrected (§§64, 66–68, 70–72, 98, 100), because a
  driver that fails on correct behaviour trains its reader to ignore it.

  **The C6 and C7 sections have now been run, end to end.** They were written in the same pass
  as the modules they drive and could not be executed in that session — the machine's
  shell-command classifier was unavailable, so `tools/verify_all.py`, `tools/serve.py` and
  every driver invocation were refused before they reached a shell while `ruff`, `mypy` and
  `pytest` went through. The first execution that got through found three defects in the
  application and five in the drivers (§101); the second was clean, and the third found that
  the harness itself was not — it had been adopting a browser left over from the previous run
  and reporting that browser's console errors as this one's, and once it launched its own
  browser instead, one section's wait turned out to have been too weak all along (§102). It
  now closes what it opens and waits for the control it is about to press, so a run leaves no
  browser, no temp profile and no console error behind. Every check this file describes is a
  check that has actually been evaluated against a running server and a running browser, in a
  browser the run itself chose. §100–§102 record what the three runs cost.

  What it does not cover: any engine but Edge; any viewport but the emulated 1440×1000 of §67;
  keyboard and focus order, since every interaction is a synthetic `.click()` with no `Tab` in it;
  ag-Grid's inline cell editor, which the cell menu's *Clear cell* stands in for — both reach the
  same `onEdit`, but driving the editor itself means synthesising a double-click and then typing
  through ag-Grid's own value API, and a check that cannot tell that apart from an editor which
  never opened is not worth having; the C7 *stop waiting* button, which aborts the poll without
  stopping the server-side job and so would need the driver to acquire a job it then abandons;
  and downloads, which the HTTP drivers exercise as bytes rather than through the browser's
  download path. It is evidence about the controls this application renders; it is not a
  substitute for a person using the tool.

- **Yahoo is unrun.** Binance klines, funding, and metrics have each been fetched live
  (see §4); no Yahoo fetch has been made from this repository, so the equity and equity-futures
  paths — including the `SpanTooWide` guard above — have never executed.

- **`npm run build` warns about a large chunk.** The ag-Grid chunk is 964 kB (269 kB
  gzipped), above Rollup's 500 kB advisory. It was 541 kB (152 kB) before §63, and the
  growth is the cost of that fix: `AllCommunityModule` registers every community feature,
  where the grid on these pages uses a handful. Accepted for now rather than tuned — it is
  a lazy-loaded vendor chunk that changes only when the dependency does, and the page is
  correct without it being smaller. The smaller fix is to register the specific feature
  modules the grid actually mounts instead of the whole bundle, which is a build change
  with no behaviour attached and belongs with whatever next touches `DataGrid.tsx`.
