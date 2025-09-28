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

`--plain-output` mirrors the console view, and `--suppress-console-logs`
removes the extra logging noise so you no longer need a `tee | grep` filter.
```

### CLI Reference

| Option | Default | Description |
|--------|---------|-------------|
| `--profile` | env (`SHORT_FINDER_PROFILE` or `default`) | Apply preset bundle (`default`, `wide`, …) |
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
| `--top-per-side` | env (10) | Cap longs/shorts before merge |
| `--save` | - | Persist output (`.json` or `.csv`) |
| `--max-workers` | env/profile | Override concurrency for data fetch (must be >0 when provided) |
| `--offline` / `--no-offline` | env | Use cached data only when possible |
| `--force-refresh` / `--no-force-refresh` | env (`SHORT_FORCE_REFRESH_CANDLES`) | Force fresh candle downloads instead of cache |
| `--quotes` | env | Preferred quote currencies (e.g., `USDC,USD,USDT`) |
| `--risk-free-rate` | env (~1%) | Annualised rate for Sharpe/Sortino |
| `--analysis-days` | env/profile (120) | Daily bars for swing analytics (must be >0 when provided) |
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

## See Also

- [`long_term_crypto_finder.py`](long_term_crypto_finder.py)
- [`README_LONG_TERM_CRYPTO_FINDER.md`](README_LONG_TERM_CRYPTO_FINDER.md)
- [`add_position_from_finder.py`](add_position_from_finder.py)
