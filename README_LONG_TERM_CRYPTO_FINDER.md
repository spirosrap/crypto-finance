# Long-Term Crypto Opportunity Finder

A comprehensive Python program that analyzes cryptocurrencies to identify the best long-term opportunities on both the LONG and SHORT side using multiple analytical approaches.

## Features

### 🔍 Comprehensive Analysis
- **Dual-Side Evaluation**: Computes and ranks both LONG and SHORT candidates for each asset.
- **Technical Analysis**: RSI, MACD, Bollinger Bands, trend strength, volatility metrics
- **Fundamental Analysis**: Market cap, volume analysis, ATH/ATL positioning (ATH/ATL sourced from CoinGecko)
- **Risk Assessment**: Sharpe ratio, Sortino ratio, maximum drawdown, risk-adjusted returns
- **Momentum Analysis**: Recent price performance and trend acceleration

### 📊 Scoring System
- **Overall Score (per side)**: Risk-adjusted composite score (0-100) for LONG and SHORT separately; top results are selected across both sides.
- **Technical Score (LONG)**: Based on RSI (neutral/oversold favored), MACD bullishness, Bollinger position, trend strength, and momentum.
- **Technical Score (SHORT)**: Inverted bias — RSI overbought favored, MACD bearish, BB overbought, negative trend strength, and short momentum derived from `100 - long_momentum`.
- **Fundamental Score**: Based on market metrics and positioning (shared across sides).
- **Risk Score**: Comprehensive risk assessment with risk level classification (shared across sides).

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

# Output results in JSON format (includes position side and trading levels)
python long_term_crypto_finder.py --output json
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--limit` | 50 | Number of top cryptocurrencies to analyze |
| `--min-market-cap` | 100000000 | Minimum market cap in USD ($100M) |
| `--max-results` | 20 | Maximum number of results to display |
| `--output` | console | Output format: 'console' or 'json' |

### Environment Variables
- `CRYPTO_ANALYSIS_DAYS`: Lookback window for historical analysis (default 365)
- `CRYPTO_MAX_RESULTS`, `CRYPTO_MAX_WORKERS`, `CRYPTO_CACHE_TTL`, `CRYPTO_REQUEST_DELAY`, etc. — see `CryptoFinderConfig.from_env()` for the full set.

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
