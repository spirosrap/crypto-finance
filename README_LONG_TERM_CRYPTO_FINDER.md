# Long-Term Crypto Opportunity Finder

A comprehensive Python program that analyzes cryptocurrencies to identify the best long-term opportunities on both the LONG and SHORT side using multiple analytical approaches.

## Features

### 🔍 Comprehensive Analysis
- **Dual-Side Evaluation**: Computes and ranks both LONG and SHORT candidates for each asset.
- **Technical Analysis**: RSI, MACD, Bollinger Bands, trend strength, volatility metrics
- **Fundamental Analysis**: Market cap, volume analysis, ATH/ATL positioning (ATH/ATL sourced from CoinGecko)
- **Risk Assessment**: Sharpe ratio, Sortino ratio, maximum drawdown, risk-adjusted returns
- **Risk Filters**: Optional hard cap on acceptable risk level (e.g., include only `MEDIUM` or lower)
- **Momentum Analysis**: Recent price performance and trend acceleration

### 📊 Scoring System
- **Overall Score (per side)**: Risk-adjusted composite score (0-100) for LONG and SHORT separately; top results are selected across both sides.
- **Technical Score (LONG)**: Based on RSI (neutral/oversold favored), MACD bullishness, Bollinger position, trend strength, and momentum.
- **Technical Score (SHORT)**: Inverted bias — RSI overbought favored, MACD bearish, BB overbought, negative trend strength, and short momentum derived from `100 - long_momentum`.
- **Fundamental Score**: Based on market metrics and positioning (shared across sides).
- **Risk Score**: Comprehensive risk assessment with risk level classification (shared across sides).
- **Setup Quality Boost**: The final rank now rewards favourable risk:reward ratios and side-aware trend alignment before clamping to 0–100. Adjust the weighting via `CRYPTO_RR_WEIGHT` and `CRYPTO_TREND_WEIGHT` if you need a more or less aggressive bias.

### 🎯 Key Metrics Evaluated
- **Volatility (30-day)**: Price stability assessment
- **Sharpe/Sortino Ratios**: Risk-adjusted return metrics
- **Maximum Drawdown**: Worst-case scenario analysis
- **Trend Strength**: Long-term directional momentum
- **ATH/ATL Distance**: Current market positioning
- **Volume Analysis**: Liquidity and market interest

### 🧭 Trading Levels (Long & Short)
- **Entry Price**: Current market price as baseline
- **Stop Loss / Take Profit**:
  - **LONG**: SL below entry (ATR/support/volatility), TP above entry (R:R, recent high, ATH-scaled)
  - **SHORT**: SL above entry (ATR/resistance/volatility), TP below entry (R:R, recent low, ATL-scaled)
- **Risk:Reward Ratio**: Calculated from selected SL/TP
- **Position Sizing**: Suggested % of portfolio based on R:R (1.0%–2.0%)

## Installation

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python long_term_crypto_finder.py
```

By default, the tool evaluates both LONG and SHORT sides and ranks the top opportunities across both.

### Advanced Usage
```bash
# Analyze top 30 cryptocurrencies, show top 10 best opportunities across BOTH sides
python long_term_crypto_finder.py --limit 30 --max-results 10

# Only consider cryptocurrencies with market cap > $500M
python long_term_crypto_finder.py --min-market-cap 500000000

# Keep results at or below MEDIUM risk level
python long_term_crypto_finder.py --max-risk-level MEDIUM

# Output results in JSON format (includes position side and trading levels)
python long_term_crypto_finder.py --output json

# Run the "wide" preset and capture a clean text report without log chatter
python long_term_crypto_finder.py --profile wide --plain-output finder_long.txt --suppress-console-logs

`--plain-output` writes the same formatted report you see in the console, while
`--suppress-console-logs` removes the streaming log handler so you no longer
need to pipe through `tee`/`grep` to obtain a clean summary file.
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--profile` | env (`CRYPTO_FINDER_PROFILE` or `default`) | Apply preset bundle of parameters (e.g., `default`, `wide`) |
| `--plain-output` | - | Write the formatted console report (without log headers) to a file |
| `--suppress-console-logs` | false | Disable console log handler for clean stdout piping |
| `--limit` | 50 (`CRYPTO_DEFAULT_LIMIT` or profile) | Number of cryptocurrencies to analyze before ranking |
| `--min-market-cap` | 100000000 | Minimum market cap in USD ($100M) |
| `--max-results` | 20 (profile/env) | Maximum number of results to display |
| `--output` | console | Output format: `console` or `json` |
| `--side` | both | Evaluate `long`, `short`, or `both` |
| `--unique-by-symbol` | false | Keep only the best side per symbol |
| `--min-score` | 0.0 | Minimum overall score to include |
| `--symbols` | - | Comma-separated symbols (e.g., `BTC,ETH,SOL`) |
| `--top-per-side` | - | Cap results per side before final sort |
| `--save` | - | Save results to path (`.json` or `.csv`) |
| `--offline` | false | Avoid external HTTP where possible (use cache) |
| `--max-workers` | env/profile | Override parallel workers; defaults from env/profile or CPU count |
| `--quotes` | env | Preferred quote currencies, e.g., `USDC,USD,USDT` |
| `--risk-free-rate` | env | Annual risk-free rate (e.g., `0.03` for 3%) |
| `--analysis-days` | env/profile | Lookback window for technical/risk metrics (e.g., `365`) |
| `--max-risk-level` | env | Highest risk level to include (`LOW`, `MEDIUM_LOW`, `MEDIUM`, `MEDIUM_HIGH`, `HIGH`, `VERY_HIGH`) |

### Environment Variables
- `CRYPTO_DEFAULT_LIMIT`: Default for `--limit` (default `50`)
- `CRYPTO_FINDER_PROFILE`: Default profile applied when `--profile` is omitted (default `default`)
- `CRYPTO_MAX_RESULTS`: Default for `--max-results` (default `20`)
- `CRYPTO_MAX_WORKERS`: Default worker threads for parallelism (default `4`)
- `CRYPTO_REQUEST_DELAY`: Global throttle between outbound requests in seconds (default `0.5`)
- `CRYPTO_CACHE_TTL`: Seconds to retain HTTP cache (default `300`)
- `CRYPTO_RISK_FREE_RATE`: Annual risk-free rate used in Sharpe/Sortino (default `0.03`)
- `CRYPTO_ANALYSIS_DAYS`: Lookback window for historical analysis (default `365`)
- `CRYPTO_RSI_PERIOD`: RSI length (default `14`)
- `CRYPTO_ATR_PERIOD`: ATR length (default `14`)
- `CRYPTO_CB_CONCURRENCY`: Max in-flight Coinbase requests (default `3`)
- `CRYPTO_MIN_MARKET_CAP`: Default filter for market cap (default `$100,000,000`)
- `CRYPTO_MAX_RISK_LEVEL`: Maximum risk level to include (e.g., `MEDIUM`)
- `CRYPTO_RR_WEIGHT`: Weight applied to risk/reward bias during final scoring (default `0.15`)
- `CRYPTO_TREND_WEIGHT`: Weight applied to trend-alignment bias during final scoring (default `0.10`)
- `CRYPTO_RR_SIGMOID_CENTER`: Risk/reward value treated as neutral (default `2.0`)
- `CRYPTO_RR_SIGMOID_K`: Steepness of the risk/reward sigmoid curve (default `1.1`)
- `CRYPTO_TREND_ALIGN_SCALE`: Percent-per-day range where trend alignment saturates (default `1.5`)

See `CryptoFinderConfig.from_env()` for the authoritative list and defaults.

## Quick Start

```bash
# Install deps
pip install -r requirements.txt

# Run with sensible defaults
python long_term_crypto_finder.py --limit 50 --max-results 20

# Save JSON to file
python long_term_crypto_finder.py --limit 100 --max-results 20 --output json --save finder.json
```

## Performance & Speed Tips

For larger scans (e.g., `--limit 400`), tune concurrency and caching:

```bash
# Faster defaults for a single run (tune based on your network and API limits)
export CRYPTO_MAX_WORKERS=12
export CRYPTO_REQUEST_DELAY=0.05
export CRYPTO_CB_CONCURRENCY=8
export CRYPTO_CACHE_TTL=3600

python long_term_crypto_finder.py --limit 400 --max-results 20 --max-workers 12 --analysis-days 90
# or simply: python long_term_crypto_finder.py --profile wide
```

- Prefer `--max-workers` between 8–16 on modern CPUs; lower if you hit 429s.
- Set `CRYPTO_FINDER_PROFILE=wide` (and optionally `CRYPTO_DEFAULT_LIMIT`) when
  you want every run to use the larger scan without repeating flags.
- Reduce `CRYPTO_REQUEST_DELAY` gradually; increase it if you see rate limits.
- Reduce `--analysis-days` (e.g., 365 → 90) to cut compute for technical metrics.

### Caching and Offline Mode

The tool caches both HTTP responses and candles.

- HTTP cache: `cache/*.pkl`, TTL controlled by `CRYPTO_CACHE_TTL`.
- Candles cache: `candle_data/*.json`, managed by `historicaldata.py`.

Warm the cache, then run in offline mode for maximum speed:

```bash
# First run to populate cache
python long_term_crypto_finder.py --limit 120 --max-results 20

# Subsequent runs use cached data where possible
python long_term_crypto_finder.py --limit 400 --max-results 20 --offline
```

## Output Explanation

### Risk Levels
- **LOW**: Very low risk, stable assets
- **MEDIUM_LOW**: Moderate risk with good fundamentals
- **MEDIUM**: Balanced risk-reward profile
- **MEDIUM_HIGH**: Higher risk but potentially higher returns
- **HIGH**: Significant risk factors present
- **VERY_HIGH**: Extreme caution recommended

### Score Interpretation
- **90-100**: Excellent long-term potential
- **80-89**: Very good opportunity
- **70-79**: Good opportunity with monitoring
- **60-69**: Moderate potential, consider carefully
- **50-59**: Below average, may need improvement
- **0-49**: Poor long-term prospects

## Sample Output

```
================================================================================
LONG-TERM CRYPTO OPPORTUNITIES ANALYSIS
================================================================================
Generated on: 2025-01-15 14:30:22
Total cryptocurrencies analyzed: 15
================================================================================

1. BTC (Bitcoin)
--------------------------------------------------
Price: $45123.456789
Market Cap: $890,123,456,789 (Rank #1)
24h Volume: $23,456,789,012
24h Change: 2.34%
7d Change: -1.23%
30d Change: 15.67%
ATH: $69000.00 (Date: 2021-11-10)   # Sourced via CoinGecko
ATL: $67.81 (Date: 2013-07-05)
Volatility (30d): 0.234
Sharpe Ratio: 1.45
Sortino Ratio: 2.12
Max Drawdown: -0.156
RSI (14): 67.8
MACD Signal: BULLISH
BB Position: NEUTRAL
Trend Strength: 0.23% per day
Momentum Score: 78.5/100
Fundamental Score: 85.2/100
Technical Score: 72.1/100
Risk Score: 32.0/100
Risk Level: MEDIUM_LOW
Overall Score: 78.3/100

💼 TRADING LEVELS (LONG):
Entry Price: $45,123.456789
Stop Loss: $43,867.0   # blended (ATR / support / % stop)
Take Profit: $50,415.0 # blended (R:R / resistance / ATH-scaled)
Risk:Reward Ratio: 3.0:1
Recommended Position Size: 2.0% of portfolio
```

### SHORT Example (excerpt)
```
2. BTC (Bitcoin) — SHORT
--------------------------------------------------
... (metrics) ...
💼 TRADING LEVELS (SHORT):
Entry Price: $45,123.456789
Stop Loss: $46,830.0   # blended (ATR / resistance / % stop)
Take Profit: $41,000.0 # blended (R:R / support / ATL-scaled)
Risk:Reward Ratio: 2.0:1
Recommended Position Size: 1.0% of portfolio
```

## Methodology

### 1. Data Collection
- **Primary Market Data**: Coinbase REST API (candles, prices, volumes)
- **Historical Candles**: Coinbase (up to 365+ days via chunked retrieval and caching)
- **ATH/ATL**: CoinGecko API (accurate all-time extremes and dates)

### 2. Technical Analysis
- **RSI (14)**: Momentum oscillator for overbought/oversold conditions
- **MACD**: Trend-following momentum indicator
- **Bollinger Bands**: Volatility-based price channels
- **Volatility**: 30-day annualized volatility calculation
- **Trend Strength**: Linear regression slope analysis

### 3. Fundamental Analysis
- **Market Cap**: Size and market positioning
- **Volume**: Liquidity and market interest assessment
- **ATH/ATL Distance**: Current market cycle positioning
- **Price Stability**: Recent volatility assessment

### 4. Risk Assessment
- **Sharpe Ratio**: Risk-adjusted returns vs. risk-free rate
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Worst historical loss
- **Volatility Risk**: Price stability analysis

### 5. Scoring Algorithm
```
Overall Score = (Technical Score × 0.4 + Fundamental Score × 0.4 + Momentum Score × 0.2) × Risk Adjustment

Risk Adjustment = 1 - (Risk Score / 200)
```

For SHORT candidates, the Technical Score and Momentum components are computed with inverted bias (e.g., RSI overbought favored, MACD bearish), and the same risk adjustment is applied.

## Rate Limiting

The program includes rate limiting and graceful fallbacks:
- **Coinbase**: Requests are paced and historical candles are chunked and cached (JSON under `candle_data/`) to minimize calls
- **CoinGecko (ATH/ATL only)**: Responses are cached as `.pkl` under `cache/` with a short TTL (default 5 minutes). On 429, the program logs a warning and continues without ATH/ATL for that asset.

## Error Handling

- Graceful handling of API failures
- Continues analysis even if some cryptocurrencies fail
- Comprehensive logging for troubleshooting
- Fallback mechanisms for missing data

## Disclaimer

This tool is for educational and informational purposes only. It should not be considered as financial advice. Always conduct your own research and consider your risk tolerance before making investment decisions.

Cryptocurrency investments carry significant risk of loss. Past performance does not guarantee future results.

## License

This project is part of the Crypto Finance Toolkit.

## Notes on Caching

- `cache/*.pkl`: Pickle cache for HTTP responses (e.g., CoinGecko ATH/ATL). Auto-expires based on `CRYPTO_CACHE_TTL` (default 300s). Safe to delete anytime.
- `candle_data/*.json`: Candle cache managed by `historicaldata.py` (TTL 1h). Safe to delete; data will be refetched.

## Logging

- Location: logs are written under `logs/long_term_crypto_finder/` with daily rotation and ~14 backups, plus a size-rotated safety log.
- Format: each line includes a short `run_id` to correlate a single program run.
- Console: logs also stream to stdout by default.
- Tuning via env vars:
  - `CRYPTO_FINDER_LOG_LEVEL` (default `INFO`): e.g., `DEBUG`, `WARNING`.
  - `CRYPTO_FINDER_LOG_TO_CONSOLE` (default `1`): set to `0` to disable console logs.
  - `CRYPTO_FINDER_LOG_RETENTION` (default `14`): number of daily backups to keep.
  - `CRYPTO_FINDER_HISTDATA_VERBOSE` (default `0`): set to `1` for verbose candle cache/fetch logs.

## JSON Output Schema

When `--output json` is used (and optionally `--save <path>.json`), each result includes:

```json
{
  "symbol": "BTC",
  "name": "Bitcoin",
  "position_side": "LONG",
  "current_price": 45123.45,
  "market_cap": 890123456789,
  "market_cap_rank": 1,
  "volume_24h": 23456789012.0,
  "price_change_24h": 2.34,
  "price_change_7d": -1.23,
  "price_change_30d": 15.67,
  "ath_price": 69000.0,
  "ath_date": "2021-11-10",
  "atl_price": 67.81,
  "atl_date": "2013-07-05",
  "volatility_30d": 0.234,
  "sharpe_ratio": 1.45,
  "sortino_ratio": 2.12,
  "max_drawdown": -0.156,
  "rsi_14": 67.8,
  "macd_signal": "BULLISH",
  "bb_position": "NEUTRAL",
  "trend_strength": 0.23,
  "momentum_score": 78.5,
  "fundamental_score": 85.2,
  "technical_score": 72.1,
  "risk_score": 32.0,
  "overall_score": 78.3,
  "risk_level": "MEDIUM_LOW",
  "entry_price": 45123.45,
  "stop_loss_price": 43867.0,
  "take_profit_price": 50415.0,
  "risk_reward_ratio": 3.0,
  "position_size_percentage": 2.0,
  "data_timestamp_utc": "2025-01-15 14:30:22"
}
```

For CSV saves (`--save results.csv`), the columns mirror the JSON keys.

## Examples

```bash
# Top 10 across both sides from top 30 by market cap
python long_term_crypto_finder.py --limit 30 --max-results 10

# Filter by symbols and save to CSV
python long_term_crypto_finder.py --symbols BTC,ETH,SOL,ADA,DOT --max-results 20 --save results.csv

# Only LONG side, require minimum score, restrict per side
python long_term_crypto_finder.py --side long --min-score 70 --top-per-side 10

# Prefer specific quote currencies
python long_term_crypto_finder.py --quotes USDC,USD,USDT --limit 100 --max-results 20
```

## Troubleshooting

- Too slow on large scans:
  - Increase `--max-workers` and reduce `--analysis-days`.
  - Raise `CRYPTO_CACHE_TTL` and run once to warm cache, then use `--offline`.
- HTTP 429 (rate limited):
  - Lower `CRYPTO_CB_CONCURRENCY` and/or increase `CRYPTO_REQUEST_DELAY`.
  - Reduce `--limit` or run during off-peak hours.
- Missing ATH/ATL fields:
  - CoinGecko may throttle; values will be filled on subsequent runs or omitted gracefully.
