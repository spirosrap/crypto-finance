# Long-Term Crypto Opportunity Finder

A comprehensive Python program that analyzes cryptocurrencies to identify the best long-term investment opportunities using multiple analytical approaches.

## Features

### 🔍 Comprehensive Analysis
- **Technical Analysis**: RSI, MACD, Bollinger Bands, trend strength, volatility metrics
- **Fundamental Analysis**: Market cap, volume analysis, ATH/ATL positioning
- **Risk Assessment**: Sharpe ratio, Sortino ratio, maximum drawdown, risk-adjusted returns
- **Momentum Analysis**: Recent price performance and trend acceleration

### 📊 Scoring System
- **Overall Score**: Risk-adjusted composite score (0-100)
- **Technical Score**: Based on technical indicators and momentum
- **Fundamental Score**: Based on market metrics and positioning
- **Risk Score**: Comprehensive risk assessment with risk level classification

### 🎯 Key Metrics Evaluated
- **Volatility (30-day)**: Price stability assessment
- **Sharpe/Sortino Ratios**: Risk-adjusted return metrics
- **Maximum Drawdown**: Worst-case scenario analysis
- **Trend Strength**: Long-term directional momentum
- **ATH/ATL Distance**: Current market positioning
- **Volume Analysis**: Liquidity and market interest

## Installation

Ensure you have the required dependencies installed:

```bash
pip install requests pandas numpy
```

## Usage

### Basic Usage
```bash
python long_term_crypto_finder.py
```

### Advanced Usage
```bash
# Analyze top 30 cryptocurrencies, show top 10 results
python long_term_crypto_finder.py --limit 30 --max-results 10

# Only consider cryptocurrencies with market cap > $500M
python long_term_crypto_finder.py --min-market-cap 500000000

# Output results in JSON format
python long_term_crypto_finder.py --output json
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--limit` | 50 | Number of top cryptocurrencies to analyze |
| `--min-market-cap` | 100000000 | Minimum market cap in USD ($100M) |
| `--max-results` | 20 | Maximum number of results to display |
| `--output` | console | Output format: 'console' or 'json' |

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
ATH: $69000.00 (Date: 2021-11-10)
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
```

## Methodology

### 1. Data Collection
- Uses CoinGecko API for comprehensive crypto data
- Historical price data (365 days) for technical analysis
- Real-time market metrics and trading volumes

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

## Rate Limiting

The program includes built-in rate limiting to respect CoinGecko API limits:
- 1 second delay between API requests
- Exponential backoff retry logic for rate limit errors
- Maximum 3 retry attempts per request

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
