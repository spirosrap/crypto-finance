# Short-Term Crypto Opportunity Finder

A swing-trading companion to `long_term_crypto_finder.py` that zooms in on the
next days-to-weeks horizon. It scans Coinbase markets, emphasises fast-moving
technical clues, and produces both LONG and SHORT trade plans with tighter
stops and closer profit targets.

## Highlights

### ⚡ Faster Technical Pulse
- **Condensed Lookback**: Defaults to ~120 daily bars and recent hourly data
  to spotlight momentum shifts.
- **High-Frequency Indicators**: Shorter RSI (7), MACD (8/21/5), and ATR (7)
  to respond quickly to volatility regime changes.
- **Volume Confirmation**: Volume spike heuristics and rolling 3-vs-15 day
  volume thrust sit inside the technical composite score.
- **Impulse & Breakout Context**: Fresh 3/10/21-day return differentials,
  breakout/breakdown distance, and ADX strength reward genuine momentum
  accelerations instead of stagnant mean reversion.
- **Momentum Score**: Uses a 20–45 bar log-price regression to capture swing
  acceleration.

### 🧮 Scoring & Filters
- **Overall Score (0–100)** per side, combining technical, momentum, and risk
  signals after risk-haircutting.
- **Side-Specific Technical Scores**: Separate logic for LONG vs. SHORT to
  reward the right combination of RSI, MACD bias, Bollinger posture, local
  trend slope, ADX strength, volume ratio/thrust, and impulse follow-through.
- **Risk Controls**: Risk bands (`LOW` → `VERY_HIGH`), volatility awareness,
  and min-score thresholds ensure crowded or weak setups can be excluded.
- **Liquidity Guard Rails**: Optional 24h USD volume and volume/market-cap
  filters prevent thin markets from surfacing, keeping fills realistic.
- **Intraday Awareness**: Configurable hourly (or faster) Coinbase candles add
  intraday momentum, volatility, and volume surges to the daily backbone.
- **Optional LLM Refinement**: When you set `SHORT_USE_OPENAI_SCORING=1`, the
  top swing setups are re-scored by an OpenAI model (default `gpt-5-mini`),
  blending a qualitative `llm_score` and rationale into the final rank.

### 🎯 Trade Planning
- **Entry**: Current price snapshot.
- **Stops**: 1.3× ATR baseline with swing-high/low and volatility clamps.
- **Targets**: Default 2.2× risk multiple, blended with recent swing extremes.
- **Sizing**: Reuses the shared ATR-based position sizing helper (respects
  `CRYPTO_RISK_PER_TRADE`, `CRYPTO_POS_CAP_PCT`, etc.).
- **Short-Line Summaries**: Each candidate ends with a concise one-liner for
  quick triage.

## Installation

Install repo dependencies once:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Run

```bash
python short_term_crypto_finder.py
```

Displays the top short-term opportunities (LONG and/or SHORT depending on
filters) with trade levels and summary blurbs.

### Focused Scans

```bash
# Evaluate 40 symbols, show best 12 setups overall
python short_term_crypto_finder.py --limit 40 --max-results 12

# Scan only SOL and AVAX pairs, long bias only
python short_term_crypto_finder.py \
  --symbols SOL,AVAX --side long --max-results 5

# Cap risk tier at MEDIUM and dump JSON for automation
python short_term_crypto_finder.py \
  --max-risk-level MEDIUM --output json --save short_setups.json

# Run the "wide" preset and write a clean text report for external sharing
python short_term_crypto_finder.py --profile wide --plain-output finder_short.txt --suppress-console-logs

# Preview available profiles (and the default marker) without running a scan
python short_term_crypto_finder.py --list-profiles

# Run the focused LLM preset for tighter swing candidates and saved report
python short_term_crypto_finder.py --profile focused_llm --plain-output finder_short.txt

`--plain-output` mirrors the console view, and `--suppress-console-logs`
removes the extra logging noise so you no longer need a `tee | grep` filter.
```

### Watchdog Reporting Companion

- After executing the finder or automated bots, summarize how finished trades performed with
  ```bash
  python watchdog_reporting.py --start-date 2025-10-01 --top-n 10
  # Review only the most recent 25 trades after date filtering
  python watchdog_reporting.py --start-date 2025-10-01 --last 25
  # Review trades 101-200 on or after the cutoff
  python watchdog_reporting.py --start-date 2025-10-01 --start-count 101 --end-count 200
  ```
- Add `--incremental-cache` (or set `SHORT_INCREMENTAL_CACHE=1`) when rerunning the finder frequently; it reuses cached metrics for symbols whose candles haven't changed, cutting rerun time dramatically on large universes.
- The helper reads `trade_logs/watchdog_closed_positions.csv`, filters by date (defaults to 2025-10-01),
  and prints headline PnL, closure-reason splits, duration buckets (0–12h / 12–24h / ≥24h), and top/bottom trades.
- The reporting helper now prints `Std Dev ($/trade)` and `Std Dev Drawdown ($)` so you can gauge both per-trade variance and equity swings.
- Add `--json` to feed the stats into dashboards or notebooks, and adjust `--duration-bounds` (for example `--duration-bounds 6 18 30`).
- Pair it with `short_term_crypto_finder.py --plain-output finder_short.txt` to loop: scan → trade execution → `watchdog_reporting.py` review.
- For expectancy/drawdown snapshots, run the legacy metrics helper with a matching window:
  ```bash
  python watchdog_stats.py --start-date 2025-10-01 --r-basis avg_loss
  # Restrict to the latest 50 trades
  python watchdog_stats.py --last 50
  # Inspect trades 101-200 after the cutoff
  python watchdog_stats.py --start-date 2025-10-01 --start-count 101 --end-count 200
  ```
  Combine `--start-date` with `--last` or `--start-count/--end-count` for rolling slices (for example `--start-date 2025-10-01 --last 40`).

### Daily Equity Drilldown

Use `watchdog_daily_equity.py` when you want the full day-by-day equity curve (with cumulative PnL, drawdown, and daily returns) over any slice of the watchdog log.

```
# Full window from 1 Oct 2025
python watchdog_daily_equity.py --start-date 2025-10-01

# Focus on trades 101-200 and export the table
python watchdog_daily_equity.py --start-date 2025-10-01 --start-count 101 --end-count 200 --output daily_equity.csv

# Emit machine-readable output
python watchdog_daily_equity.py --start-date 2025-10-01 --json
```

The tool mirrors the same filters as `watchdog_reporting.py` / `watchdog_stats.py` (date window, count window, `--last` tail selection) and reports additional metrics:

- Daily cumulative equity and drawdown columns
- Per-day return percentages and trade counts
- Aggregate variance stats (`Std Dev ($/trade)` and `Std Dev Drawdown ($)`)
- Ending equity, max drawdown, and Sharpe based on daily returns

### Preset Profiles

- `default`: Mirrors environment defaults (or `SHORT_*` overrides) without extra changes.
- `wide`: Evaluates a broader universe (`limit` 400) with more workers for faster bulk scans.
- `focused_llm`: Concentrates on liquidity-plus momentum swings (`limit` 200, `top_per_side` 5, OpenAI scoring enabled, ≥$5M 24h volume, ≥3% volume/market-cap ratio, 20-day intraday lookback, `unique_by_symbol`, `max_risk_level` MEDIUM); pair with `--plain-output` to persist the run.
- `focused_llm_400`: Same scaffolding as `focused_llm` but widens the scan (`limit` 400) so the LLM layer can sift a much larger universe while keeping the identical liquidity and risk guardrails.
- `focused_llm_100`: Mirrors `focused_llm` but trims the universe (`limit` 100) for faster refreshes while keeping the same OpenAI-assisted filters and risk guardrails.

Even when other constraints shrink the candidate list—say only `3S + 1L` or
`2S + 1L` qualify—you should still consider executing the surfaced trades
instead of waiting for a full five-per-side roster.

### CLI Reference

| Option | Default | Description |
|--------|---------|-------------|
| `--profile` | env (`SHORT_FINDER_PROFILE` or `default`) | Apply preset bundle (`default`, `wide`, …) |
| `--list-profiles` | - | Print available profiles (marks the default) and exit |
| `--plain-output` | - | Write the formatted console report (no log header) to disk |
| `--suppress-console-logs` | false | Disable console logging for clean stdout piping |
| `--limit` | 30 (`SHORT_DEFAULT_LIMIT` or profile) | Universe size to analyse before ranking (must be >0) |
| `--min-market-cap` | env / ≥$50M | Minimum market cap filter (must be >0) |
| `--min-volume` | env | Minimum 24h USD volume (must be >0 when provided) |
| `--max-results` | env/profile | Number of setups to display (must be >0 when provided) |
| `--output` | `console` | `console` or `json` |
| `--side` | env (`both`) | Restrict to `long`, `short`, or `both` |
| `--unique-by-symbol` / `--no-unique-by-symbol` | env | Keep only top side per symbol |
| `--min-score` | env (≥20) | Drop results below this overall score |
| `--symbols` | - | Comma-separated tickers to force-include |
| `--top-per-side` | env (10) | Cap longs/shorts before merge; pairing `--top-per-side 5` with other constraints may surface fewer than 5 long and 5 short candidates when conditions are tight |
| `--save` | - | Persist output (`.json` or `.csv`) |
| `--max-workers` | env/profile | Override concurrency for data fetch (must be >0 when provided) |
| `--offline` / `--no-offline` | env | Use cached data only when possible |
| `--force-refresh` / `--no-force-refresh` | env (`SHORT_FORCE_REFRESH_CANDLES`) | Force fresh candle downloads instead of cache |
| `--incremental-cache` / `--no-incremental-cache` | env (`SHORT_INCREMENTAL_CACHE`) | Reuse cached metrics when candles unchanged |
| `--quotes` | env | Preferred quote currencies (e.g., `USDC,USD,USDT`) |
| `--risk-free-rate` | env (~1%) | Annualised rate for Sharpe/Sortino |
| `--analysis-days` | env/profile (120) | Daily bars for swing analytics (must be >0 when provided) |
| `--intraday-lookback-days` | env (14) | History window (days) for intraday candles feeding hourly metrics |
| `--intraday-granularity` | env (`ONE_HOUR`) | Coinbase candle granularity for intraday fetches |
| `--intraday-resample` | env (`4H`) | Pandas resample alias used when aggregating intraday stats |
| `--min-vmc-ratio` | env | Minimum volume-to-market-cap ratio (e.g., `0.03` for 3%) |
| `--max-risk-level` | env | Highest allowed risk tier |
| `--use-openai-scoring/--no-use-openai-scoring` | env (`SHORT_USE_OPENAI_SCORING`) | Toggle LLM-assisted scoring from the CLI |
| `--openai-weight` | env (`SHORT_OPENAI_WEIGHT`) | Blend ratio between baseline and LLM score |
| `--openai-model` | env (`SHORT_OPENAI_MODEL`) | Override OpenAI model identifier |
| `--openai-max-candidates` | env (`SHORT_OPENAI_MAX_CANDIDATES`) | Cap number of candidates sent to the model |
| `--openai-temperature` | env (`SHORT_OPENAI_TEMPERATURE`) | Set temperature for the OpenAI call (defaults to model standard) |
| `--openai-sleep-seconds` | env (`SHORT_OPENAI_SLEEP_SECONDS`) | Pause between OpenAI calls |

> Numeric count arguments (`--limit`, `--max-results`, `--top-per-side`, `--analysis-days`, etc.) now fail fast if a non-positive value is supplied, keeping runs from silently accepting invalid thresholds.

## Environment Overrides

Short-term settings read both the generic `CRYPTO_*` variables and the
`SHORT_*` variants. Key overrides:

| Variable | Purpose | Default |
|----------|---------|---------|
| `SHORT_DEFAULT_LIMIT` | Default for `--limit` | 30 |
| `SHORT_FINDER_PROFILE` | Default profile when `--profile` is omitted | `default` |
| `SHORT_ANALYSIS_DAYS` | Daily lookback window | 120 |
| `SHORT_MIN_MARKET_CAP` | Market-cap floor (USD) | max(`CRYPTO_MIN_MARKET_CAP`, 50M) |
| `SHORT_MAX_RESULTS` | Default for `--max-results` | `CRYPTO_MAX_RESULTS` |
| `SHORT_INTRADAY_LOOKBACK_DAYS` | Intraday lookback (days) | inherits `CRYPTO_INTRADAY_LOOKBACK_DAYS` / 14 |
| `SHORT_INTRADAY_GRANULARITY` | Intraday Coinbase granularity | inherits `CRYPTO_INTRADAY_GRANULARITY` / `ONE_HOUR` |
| `SHORT_INTRADAY_RESAMPLE` | Resample alias for intraday features | inherits `CRYPTO_INTRADAY_RESAMPLE` / `4H` |
| `SHORT_MIN_VOLUME_24H` | Minimum 24h USD volume | inherits `CRYPTO_MIN_VOLUME_24H` / 0 |
| `SHORT_MIN_VMC_RATIO` | Minimum volume-to-market-cap ratio | inherits `CRYPTO_MIN_VMC_RATIO` / 0 |
| `SHORT_TOP_PER_SIDE` | Pre-cap per direction | 10 |
| `SHORT_SIDE` | Default side selection | `both` |
| `SHORT_MIN_SCORE` | Minimum overall score | 20.0 |
| `SHORT_RISK_FREE_RATE` | Annual risk-free rate | 0.01 |
| `SHORT_RSI_PERIOD` | RSI length | 7 |
| `SHORT_ATR_PERIOD` | ATR length | 7 |
| `SHORT_MACD_FAST` | MACD fast EMA | 8 |
| `SHORT_MACD_SLOW` | MACD slow EMA | 21 |
| `SHORT_MACD_SIGNAL` | MACD signal EMA | 5 |
| `SHORT_BB_PERIOD` | Bollinger SMA length | 14 |
| `SHORT_STOCH_PERIOD` | Stochastic length | 10 |
| `SHORT_WILLIAMS_PERIOD` | Williams %R length | 10 |
| `SHORT_CCI_PERIOD` | CCI length | 14 |
| `SHORT_MAX_RISK_LEVEL` | Highest risk tier allowed | inherits / optional |
| `SHORT_MAX_WORKERS` | Thread pool size | `CRYPTO_MAX_WORKERS` |
| `SHORT_REQUEST_DELAY` | Global throttle seconds | `CRYPTO_REQUEST_DELAY` |
| `SHORT_USE_OPENAI_SCORING` | Enable LLM refinement | `CRYPTO_USE_OPENAI_SCORING` |
| `SHORT_OPENAI_MODEL` | Override OpenAI model | `CRYPTO_OPENAI_MODEL` |
| `SHORT_OPENAI_WEIGHT` | Blend factor for LLM score | `CRYPTO_OPENAI_WEIGHT` |
| `SHORT_OPENAI_MAX_CANDIDATES` | Cap candidates sent to LLM | `CRYPTO_OPENAI_MAX_CANDIDATES` |
| `SHORT_OPENAI_TEMPERATURE` | Temperature override for OpenAI queries | `CRYPTO_OPENAI_TEMPERATURE` |
| `SHORT_OPENAI_SLEEP_SECONDS` | Pause between OpenAI calls | `CRYPTO_OPENAI_SLEEP_SECONDS` |

All other shared risk controls (`CRYPTO_RISK_PER_TRADE`, `CRYPTO_POS_CAP_PCT`,
etc.) apply identically to both finders.

## Output Structure

The console report mirrors the long-term finder with a short-term banner and
per-asset cards containing:

1. **Snapshot Metrics**: price, market cap/rank, 24h/7d/30d change, ATH/ATL,
   volatility, Sharpe/Sortino, drawdown, RSI, MACD bias, Bollinger stance,
   trend strength, ADX, impulse/continuation, recent breakout distance,
   scores, and risk classification.
2. **Trading Levels**: entry, stop, target, risk:reward, and suggested
   position size.
3. **Short-Line Summary**: one-line punchlist (`score`, `RR`, `RSI`, MACD
   nuance, risk tier, trend delta) for quick scanning or downstream parsing.

Example excerpt:

```
================================================================================
SHORT-TERM CRYPTO OPPORTUNITIES ANALYSIS
================================================================================
Generated on (UTC): 2025-09-17 14:05:11Z
Total opportunities listed: 8
================================================================================

1. SOL (Wrapped SOL) — SHORT
--------------------------------------------------
... [full metrics truncated] ...

Short-Line Summaries
--------------------------------------------------
1. Summary: SOL short – score 68.02, 2.3× RR; RSI 66; bullish MACD fade; risk medium_low; trend -0.18%/d.
2. Summary: AVAX short – score 66.41, 2.4× RR; RSI 64; bullish MACD fade; risk medium_low.
```

## Workflow Tips

- Warm the HTTP and candle caches with the long-term finder, then run the
  short-term finder in `--offline` mode for rapid iteration.
- Use `--profile wide` (or set `SHORT_FINDER_PROFILE=wide`) to jump straight to
  the 400-symbol scan with 12 workers and a 90-day window.
- Tighten `SHORT_REQUEST_DELAY` cautiously; Coinbase 429s may require backing
  off.
- Combine with `add_position_from_finder.py` to create ready-to-send perp
  orders (the parser now ignores the trailing summary block automatically).

### Top-5 Selection Helper

If you want to automatically pick only the strongest candidates for execution, use `add_top5_from_finder.py` on the plain text output:

Selection logic:
- best 2 LONGs by score
- best 2 SHORTs by score
- next best remaining by score (any side)

Examples:

```bash
# Generate plain text and select top-5 (2L/2S + next best), dry run
python short_term_crypto_finder.py --profile wide --plain-output finder_short.txt --suppress-console-logs
python add_top5_from_finder.py --file finder_short.txt --portfolio-usd 25000 --leverage 5 --order limit --expiry 12h

# Execute immediately with market orders and 24h GTD expiry for brackets
python add_top5_from_finder.py --file finder_short.txt --portfolio-usd 25000 --leverage 5 --order market --expiry 24h --execute
```

Notes:
- `--expiry` threads through to GTD bracket orders (choices: `12h`, `24h`, `30d`; default `30d`).
- If fewer than 2 per side are available, the script selects whatever exists and still caps total to 5.

### Position Age Watchdog (auto-close stale perp positions)

After you place positions from the short-term finder (for example via `add_top5_from_finder.py`), you can enforce a maximum holding window for INTX perpetuals using `watchdog_close_old_positions.py`. It scans your INTX portfolio and market-closes any open perp position older than a configured age.

Purpose:
- Keep the book fresh by auto-closing positions that have lingered past your short-term horizon
- Prevent forgotten small residuals from accumulating overnight/weekend

Usage:
```bash
# Close positions older than 24h (one-shot)
python watchdog_close_old_positions.py --max-age-hours 24

# Run continuously every 5 minutes (still logs every closure to CSV)
python watchdog_close_old_positions.py --max-age-hours 24 --interval-seconds 300

# Only act on a specific product
python watchdog_close_old_positions.py --max-age-hours 12 --product BTC-PERP-INTX

# Enable verbose logs
python watchdog_close_old_positions.py --max-age-hours 24 --verbose

# Backfill the most recent 10 logged closures using exchange fills
python watchdog_close_old_positions.py --backfill-last 10

# Log take-profit/stop-loss closures detected in recent fills (one shot)
python watchdog_close_old_positions.py --log-fills

# Continuously close stale positions (skip CSV logging; capture fills separately)
python watchdog_close_old_positions.py --interval-seconds 300 --no-log-closures

# Run fill logging on its own cadence (separate process or cron)
python watchdog_close_old_positions.py --log-fills --fills-interval 300 --skip-close
```

Options:
- `--max-age-hours` (int, default 24): Age threshold; positions opened before now−N hours are closed
- `--product` (str, optional): Only check/close this product id (e.g., `BTC-PERP-INTX`)
- `--interval-seconds` (int, default 0): If >0, run continuously with this interval between scans
- `--backfill-last` (int, default 0): Recompute exit price/PnL for the most recent N log entries and exit (no new closes)
- `--skip-close`: Skip age-based closing and only run ancillary actions (for example, fill logging)
- `--log-fills`: Append TP/SL closures from recent fills to the log using the watchdog checkpoint
- `--start-count` / `--end-count`: When you use `watchdog_reporting.py`, slice the post-filter trades by ordinal position (1-based). For example `--start-count 101 --end-count 200` reviews the 101st–200th trades in the window.
- `--no-log-closures`: Skip writing age-triggered closes to `watchdog_closed_positions.csv` when you prefer to log fills manually (for example, via `--log-fills --skip-close`)
- `--fills-limit` (int, default 500): Number of recent fills to inspect when `--log-fills` is enabled
- `--fills-interval` (int, default 0): If >0, poll fills continuously every N seconds (requires `--skip-close` when used in the same process)
- `--fills-bootstrap-existing`: On the first run with `--log-fills`, treat existing cycles as new so historical TP/SL closures are captured
- `--verbose`: Enable debug logging

How it works:
- Looks up your INTX portfolio and fetches current perp positions
- Determines each position's open time from common fields; if missing, infers it by replaying filled orders to find when net size last moved from 0 to non‑zero
- Cancels any open orders for a product before attempting to close its position
- Sends a market IOC order on the opposite side for the net size (uses CROSS margin; preserves reported `leverage`)
- Writes every successful closure to `trade_logs/watchdog_closed_positions.csv`, capturing entry/exit prices, realized PnL, hold duration, and excursion stats (MAE/MFE) so you can audit risk handling later
- Treats expired positions with |PnL| ≤ `$WATCHDOG_BREAKEVEN_ABS` (default `1.0`) as breakeven, setting PnL to zero and tagging the row `expired_breakeven` to keep expectancy calculations realistic for time-stop exits

Safety notes:
- Uses market IOC; expect small slippage versus limit exits
- Requires valid INTX API credentials (`API_KEY_PERPS`, `API_SECRET_PERPS` in `config.py`)
- Only acts on non‑zero net perp positions; if none found, it exits quietly
- Set `--interval-seconds` to keep enforcing the policy throughout the day; add `--no-log-closures` if you plan to capture trade history via manual `--log-fills` runs instead of the auto-closer.

File: `watchdog_close_old_positions.py`

### TP/SL Fill Logging (capture bracket completions)

With the same script, supply `--log-fills` to poll recent INTX fills, detect
round-trip closures (net position back to zero), and append them to the shared
CSV. A checkpoint in `trade_logs/watchdog_tp_sl_checkpoint.json` prevents
duplicates. Use `--fills-bootstrap-existing` on your first run if you need to
backfill historical TP/SL completions.

**Accuracy notes**
- Fill-derived rows always reflect actual execution prices and occur only when
  the fills stream shows the position returning to zero (TP, SL, or manual
  closure). They supersede the time-stop estimates logged in older workflows.
- If you notice a historical row missing after a bootstrap run, it usually
  means the required fills are outside the requested window or the exchange
  never reported the matching exit leg. You can copy that single row from a
  trusted backup if you are certain the trade completed.

**Full backfill workflow**
1. Delete any existing log/checkpoint files if you want to rebuild from scratch:
   ```bash
   rm trade_logs/watchdog_closed_positions.csv trade_logs/watchdog_tp_sl_checkpoint.json
   ```
2. Run a bootstrap pass that fetches enough recent fills to cover the history
   you want to reconstruct (adjust `--fills-limit` as needed):
   ```bash
   python watchdog_close_old_positions.py --log-fills --fills-limit 600 --fills-bootstrap-existing --skip-close --verbose
   ```
   The verbose flag helps confirm each detected cycle.
3. Keep the newly created checkpoint. Future runs should omit
   `--fills-bootstrap-existing` so only brand-new closures are appended:
   ```bash
   python watchdog_close_old_positions.py --log-fills --skip-close
   ```

When you need both behaviors running continuously, launch them in separate
processes (for example, two terminals or background jobs). The script enforces
this separation to avoid interleaving loops that manage orders and fills in the
same thread. Use `--skip-close` whenever you are only refreshing the log.

### Performance Snapshot (expectancy & drawdowns)

After your watchdogs populate `trade_logs/watchdog_closed_positions.csv`, run
`watchdog_stats.py` to compute win rate, expectancy, standard deviation, max drawdown %, and average R multiples across the logged trades.

```
# Full history with default average-loss risk basis
python watchdog_stats.py

# Focus on the most recent 50 trades
python watchdog_stats.py --last 50

# Inspect trades 101-200 after the cutoff
python watchdog_stats.py --start-date 2025-10-01 --start-count 101 --end-count 200

# Infer starting equity from a known ending balance (for example $200 now)
python watchdog_stats.py --last 50 --ending-equity 200

# Use a fixed $75 risk per trade for R multiples
python watchdog_stats.py --r-basis fixed --risk-dollar 75

# Emit structured output for dashboards / scripts
python watchdog_stats.py --json
```

Metrics are derived from the `profit_loss` column (already breakeven-adjusted by
the watchdogs). The additional standard-deviation line shows how widely per-trade
PnL has been swinging around the mean, so you can gauge variance alongside expectancy. `closure_reason=expired_breakeven` rows are treated as zero PnL
to keep expectancy realistic.

If your account balance changed during the window, supply `--ending-equity` so
the script infers the starting balance as `ending_equity - cumulative_profit_loss`
for the selected trades. Otherwise, pass `--starting-equity` with the known
opening figure.

File: `watchdog_stats.py`

## See Also

- [`long_term_crypto_finder.py`](long_term_crypto_finder.py)
- [`README_LONG_TERM_CRYPTO_FINDER.md`](README_LONG_TERM_CRYPTO_FINDER.md)
- [`add_position_from_finder.py`](add_position_from_finder.py)
