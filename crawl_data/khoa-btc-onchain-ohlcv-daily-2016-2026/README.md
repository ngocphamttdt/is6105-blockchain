# Bitcoin Daily OHLCV + On-Chain Metrics, 2016–2026 (Binance, Coin Metrics, Blockchain.com)

One table, one row per UTC day from **2016-01-01 to 2026-08-31** (3,896 days). Each row holds the Binance BTCUSDT daily price candle, on-chain metrics from Coin Metrics Community and Blockchain.com, and two derived valuation ratios (Puell Multiple, NVT). Built for research on whether on-chain data adds value to next-day BTC direction forecasting (course project IS6105, group BLOCK&ROLL).

Values are stored as published. **Missing days are left empty: nothing is filled, interpolated, scaled or smoothed.**

## Files

| File | Rows | What it holds |
|---|---|---|
| `master_daily.csv` | 3,896 | **Main table** (23 columns): `date`, the 6 OHLCV columns, and the 16 on-chain/derived columns kept after quality checks. |
| `ohlcv_daily.csv` | 3,302 | Binance BTCUSDT daily candles (2017-08-17 → 2026-08-31), built from the hourly bars, with extra columns: quote volume, trade count, taker-buy volumes. |
| `ohlcv_1h.csv` | 79,117 | The Binance BTCUSDT hourly bars behind the daily candles (useful for realized volatility). |
| `coinmetrics_daily.csv` | 3,896 | Every Coin Metrics metric downloaded (15), using the original API codes, plus revision-status columns for the exchange-flow metrics and `AssetEODCompletionTime`. |
| `blockchain_com_daily.csv` | 3,896 | Blockchain.com `difficulty` and `estimated-transaction-volume-usd`. |
| `derived_metrics.csv` | 3,896 | `puell_multiple` and `nvt_ratio_bc`. |
| `data_dictionary.csv` | 28 | One row per column (including dropped metrics): source, API code, group, unit, transform, role, keep/drop reason, missing share, licence and attribution. |
| `metric_availability.csv` | 16 | Per-metric availability check on the Coin Metrics Community API (OK/FAIL, HTTP status, first/last date, sha256 of the raw response). |
| `ohlcv_checks.json` | — | Quality checks on the OHLCV data (incomplete days, anomalies, comparison with Binance's own daily candles). |

## Columns in `master_daily.csv`

| Column | Group | Unit | Meaning |
|---|---|---|---|
| `date` | key | UTC day | Observation day (YYYY-MM-DD) |
| `ohlcv_open` | Price | USDT | First hourly open of the UTC day |
| `ohlcv_high` | Price | USDT | Highest hourly high |
| `ohlcv_low` | Price | USDT | Lowest hourly low |
| `ohlcv_close` | Price | USDT | Last hourly close |
| `ohlcv_volume` | Price | BTC | Sum of hourly base volume |
| `ohlcv_n_bars_1h` | Price | bars | Hourly bars behind the day (24 when complete) |
| `cm_AdrActCnt` | Activity | addresses | Active addresses |
| `cm_TxCnt` | Activity | transactions | Transaction count |
| `bc_est_tx_vol_usd` | Activity | USD | Estimated on-chain transaction volume (Blockchain.com) |
| `cm_FlowInExNtv` | Flows | BTC | Exchange inflow |
| `cm_FlowOutExNtv` | Flows | BTC | Exchange outflow |
| `cm_SplyExNtv` | Flows | BTC | Supply held on exchanges |
| `cm_HashRate` | Mining | TH/s (daily mean) | Network hash rate |
| `cm_FeeTotNtv` | Mining | BTC | Total fees |
| `bc_difficulty` | Mining | difficulty | Mining difficulty (Blockchain.com) |
| `cm_CapMVRVCur` | Valuation | ratio | MVRV (market value / realized value) |
| `puell_multiple` | Valuation | ratio | IssTotUSD(t) / mean(IssTotUSD over days t−364..t) |
| `nvt_ratio_bc` | Valuation | ratio | CapMrktCurUSD(t) / bc_est_tx_vol_usd(t) |
| `cm_IssTotUSD` | Input | USD | Daily issuance value |
| `cm_CapMrktCurUSD` | Input | USD | Market capitalisation |
| `cm_IssTotNtv` | Input | BTC | Daily issuance |
| `cm_PriceUSD` | Input | USD | Coin Metrics reference price |

**Dropped by quality checks** (still in `coinmetrics_daily.csv`): `TxTfrCnt` (log-diff correlation 0.918 with `TxCnt`) and `BlkCnt` (0.993 with `HashRate`). **Not available for free:** `TxTfrValAdjUSD` (HTTP 403).

Quality rules for the on-chain metrics, fixed before they were run on the data and using no price or label data: drop a metric missing more than 5% of days in 2018-01-01..2026-08-31; difference monotonically increasing series (none qualified); within a group, if two log-differenced series correlate above 0.9, keep one by a fixed priority order.

## Missing values

| Column(s) | Empty days | Why |
|---|---|---|
| all `ohlcv_*` | 594 | 2016-01-01 → 2017-08-16, before Binance BTCUSDT started trading |
| `puell_multiple` | 364 | Needs 365 days of history; first value is 2016-12-30 |
| `bc_est_tx_vol_usd`, `nvt_ratio_bc` | 8 | Gaps in the Blockchain.com series |
| `bc_difficulty` | 3 | Gap in the Blockchain.com series (2025-11-13 to 2025-11-15) |
| all other columns | 0 | |

## OHLCV notes

- Daily candles are built from Binance hourly bars by UTC day: open = first bar's open, high = max, low = min, close = last bar's close, volume = sum. Binance spot data from 2025 uses microsecond timestamps; both units were detected and handled.
- **30 days have fewer than 24 hourly bars** (exchange outages), shown by `ohlcv_n_bars_1h < 24`. The list is in `ohlcv_checks.json`. 2018-02-08 has a single bar. On those days the candle covers only the hours that traded.
- Compared with Binance's own daily candles, the values are identical except around the 2018-02-08/11 outage (43 off-hour bars, 2018-02-09 → 11), and for small volume differences (≤ 2.4%) on five other days. Details are in `ohlcv_checks.json`.
- Large real moves are kept as they are, for example 2020-03-12 (daily |log return| 0.50).

## Important caveats

- **Publication delay / look-ahead.** Coin Metrics publishes day *d* about 2–9.5 hours after (d+1) 00:00 UTC (median ≈ 3 h, measured on recent days). A model predicting at (t+1) 00:00 UTC can use OHLCV through day t, but should only use on-chain rows dated ≤ t − 1 (lag ≥ 1 day).
- **Restated values, not point-in-time.** Coin Metrics values are the latest revision. Exchange-flow metrics carry status `flash` and depend on exchange-address labels that are updated over time, so historical flow values can contain information that was not known at the time.
- **NVT definition.** `nvt_ratio_bc` divides Coin Metrics market cap by Blockchain.com *estimated* USD volume. It is not Coin Metrics' adjusted NVT, because adjusted transfer value is not in the free tier.
- **Two on-chain sources.** The Blockchain.com and Coin Metrics day conventions were cross-checked: the best day lag is 0. Implied difficulty from Coin Metrics hash rate matches Blockchain.com difficulty with correlation 0.998.

## Sources, licence and attribution

- **Binance Data Vision**, https://data.binance.vision — BTCUSDT spot klines (1h). Public market data; redistribution terms were not verified.
- **Coin Metrics Community Network Data**, https://coinmetrics.io — licensed **CC BY-NC 4.0** (non-commercial use only, attribution required). Attribution: "Coin Metrics Community Network Data (coinmetrics.io), CC BY-NC 4.0".
- **Blockchain.com Charts**, https://www.blockchain.com/explorer/charts (public Charts API, api.blockchain.info). No data licence was found, and redistribution terms were not verified.

Because it contains CC BY-NC 4.0 data, this dataset as a whole is for **non-commercial use** and must keep the attributions above.

## Integrity

`content_hash` of the master table (sha256 of a canonical text serialisation): `373b003629284486ad16fda17a8ad2bcce98b11b3becadca1a81defc09ae6c20`.
