# Crypto Finance — Current Pipeline (Dec 2025)

## Current Focus
- **Short-Term Crypto Finder (`short_term_crypto_finder.py`)**: Strict RR ≥ 2, ATR(7) gated with tiered caps (3k USD plus bps tiers ≈325/350/400/450 by price bands; tighter cap wins), hard SL/TP. Runs on USDC pairs; logs to `logs/short_term_crypto_finder/`.
- **Breakout Autotrade (`scripts/run_breakout_autotrade.py`)**: Hourly scan of USDC majors (BTC/ETH/SOL/XRP/ADA/DOT/AVAX/LINK/LTC/DOGE/OP/ARB/ATOM/UNI/AAVE/MKR/INJ) with fixed 2R structure, $500 notional / 50x, 24h lock to avoid stacking. Writes finder-format output; uses Coinbase primary with Kraken fallback. Near-breakouts are logged.
- **Safety First**: Downtime is expected; no trades when RR/ATR gates fail. Flat is acceptable.

## Active Tools
- **Finders**: `short_term_crypto_finder.py`, `scripts/symbol_snapshot.py` for targeted snapshots. `scripts/symbol_snapshot.py --gate-scan [--scan-limit N]` shows which symbols are closest to clearing RR/ATR gates without loosening filters.
- **Breakout Suite**: `scripts/breakout_scanner.py` (finder-style output, near-breakout logs), `scripts/run_breakout_autotrade.py` (cron-friendly runner with lock).
- **Watchdogs/Closers**: `watchdog_close_old_positions.py` (optional 24h timeout), `watchdog_dashboard.py` for monitoring (Streamlit).
- **Support**: `add_position_from_finder.py` to stage/execute trades from finder-format text.

## What’s Retired/Optional
- Forced daily trading or balanced 5-trade baskets—no longer required.
- Mandatory 24h holds across all strategies—used only when explicitly configured (e.g., breakout autotrade lock).
- Reservoir/multi-basket experiments—on pause; sticking to finder + breakout playbooks.
- Legacy bots/tools (not maintained): simplified_trading_bot v1.2.1f, simplified_trading_bot_v1.2.2.py, reservoir/multi-basket scripts. Kept for reference only.

## Quick Start (Current Flow)
1) Run snapshots: `python scripts/symbol_snapshot.py --symbols BTC,ETH --profile focused_no_llm_100`
   - Gate proximity view (optional): `python scripts/symbol_snapshot.py --gate-scan --profile focused_no_llm_100 --top 15 --scan-limit 100`
2) Let cron handle:
   - `short_term_crypto_finder.py` (daily) → feed results to `add_position_from_finder.py` or `add_top5_from_finder.py` to stage/execute trades.
   - `run_breakout_autotrade.py` (hourly) for 2R breakouts on USDC majors.
3) Check `logs/breakout_autotrade.log` for near-breakouts/triggers; finder logs under `logs/short_term_crypto_finder/`.
4) Optional: enforce a 24h timeout on positions with `watchdog_close_old_positions.py`.

## Environment & Requirements
- Python 3.11, ccxt ≥ 4.2, pandas ≥ 2.3, numpy ≥ 1.24, TA-Lib ≥ 0.6.7, openai ≥ 1.109.1 (installed), full list in `requirements.txt`.
- Configure API keys in `.env` (Coinbase primary: `API_KEY`/`API_SECRET`; Kraken fallback: `KRAKEN_API_KEY`/`KRAKEN_API_SECRET`). PEM secrets normalize `\\n`.

## Install/Setup

```bash
conda create -n trade python=3.11
conda activate trade
python scripts/install_requirements.py
```

> macOS: needs Homebrew for TA-Lib; Windows: install TA-Lib manually.

## Prerequisites

### Python Environment Setup

```bash
conda create -n myenv python=3.11
conda activate myenv
```

### Install Dependencies

```bash
python scripts/install_requirements.py
```

The helper script downloads and compiles the TA-Lib C library (v0.6.4) into `~/.local` when it is not already present, then installs all Python packages from `requirements.txt`. If you only need the C library, use:

```bash
python scripts/install_requirements.py --skip-pip
```

> **macOS:** The script expects Homebrew to be available (`brew install ta-lib` under the hood).
>
> **Windows:** Automatic TA-Lib setup is not supported. Install the TA-Lib binary manually, then rerun the script.

## Installation

1. Clone the repository:
   ```bash
git clone https://github.com/spirosrap/crypto-finance.git
cd crypto-finance
   ```

2. Install the required dependencies (handles TA-Lib automatically):
   ```bash
   python scripts/install_requirements.py
   ```

3. Create a `config.py` file in the root directory with your API keys.

4. (Optional) Install additional model weights:
   ```bash
   python setup_models.py
   ```

## API Keys Configuration

The bot requires several API keys for full functionality. Create a `config.py` file in the root directory with the following structure:

```python
class Config:
    # Required API Keys
    COINBASE = {
        'API_KEY': 'your_coinbase_api_key',
        'API_SECRET': 'your_coinbase_api_secret'
    }

    # Optional - For Enhanced Analysis
    NEWS_API_KEY = 'your_news_api_key'  # For sentiment analysis
    
    # Optional - For Social Sentiment
    TWITTER = {
        'BEARER_TOKEN': 'your_twitter_bearer_token',
        'CONSUMER_KEY': 'your_twitter_consumer_key',
        'CONSUMER_SECRET': 'your_twitter_consumer_secret',
        'ACCESS_TOKEN': 'your_twitter_access_token',
        'ACCESS_TOKEN_SECRET': 'your_twitter_access_token_secret'
    }
    
    # Optional - For AI-Enhanced Analysis
    AI_MODELS = {
        'OPENAI_KEY': 'your_openai_api_key',
        'DEEPSEEK_KEY': 'your_deepseek_api_key',
        'OPENROUTER_KEY': 'your_openrouter_api_key',
        'XAI_KEY': 'your_xai_api_key',
        'HYPERBOLIC_KEY': 'your_hyperbolic_api_key',
        'GROK_KEY': 'your_grok_api_key'
    }

    # Trading Parameters
    TRADING = {
        'DEFAULT_LEVERAGE': 1,
        'MAX_LEVERAGE': 20,
        'DEFAULT_STOP_LOSS_PCT': 1.0,
        'DEFAULT_TAKE_PROFIT_PCT': 2.0,
        'RISK_PER_TRADE': 0.01
    }

    # Analysis Parameters
    ANALYSIS = {
        'DEFAULT_TIMEFRAMES': ['ONE_MINUTE', 'FIVE_MINUTE', 'ONE_HOUR'],
        'SENTIMENT_ENABLED': True,
        'AI_ANALYSIS_ENABLED': True,
        'HFT_ENABLED': False
    }
```

### Required API Keys
1. **Coinbase API** (Required)
   - Get your API credentials from [Coinbase Advanced Trade](https://www.coinbase.com/settings/api)
   - Required for all trading functionality
   - Set `COINBASE['API_KEY']` and `COINBASE['API_SECRET']`

### Optional API Keys
2. **News API** (Optional)
   - Get your API key from [NewsAPI](https://newsapi.org/)
   - Used for news sentiment analysis
   - Set `NEWS_API_KEY`

3. **Twitter API** (Optional)
   - Get your credentials from [Twitter Developer Portal](https://developer.twitter.com/)
   - Used for social sentiment analysis
   - Configure all Twitter-related keys in the `TWITTER` dictionary

4. **AI Model Keys** (Optional)
   - Each key enables different AI analysis capabilities:
     - OpenAI: Advanced market analysis ([Get Key](https://platform.openai.com/))
     - DeepSeek: Pattern recognition ([Get Key](https://platform.deepseek.ai))
     - OpenRouter: Multi-model analysis ([Get Key](https://openrouter.ai/))
     - XAI: Explainable AI analysis ([Get Key](https://xai.com))
     - Hyperbolic: Price prediction ([Get Key](https://hyperbolic.ai))
     - Grok: Real-time market insights ([Get Key](https://grok.x.ai))

### Configuration Notes
- Store sensitive keys securely
- Never commit `config.py` to version control
- Use environment variables in production
- Consider using a `.env` file for local development
- Rotate API keys periodically
- Monitor API usage and limits

### Environment Variables
You can also use environment variables instead of `config.py`:
```bash
export COINBASE_API_KEY="your_key"
export COINBASE_API_SECRET="your_secret"
# ... other environment variables
```

## Usage (Legacy Tools)

The following tools are legacy/not part of the current short-term finder + breakout pipeline. Keep them for reference; they are not actively maintained.

### Basic Trading Bot (legacy, `base.py`)

```bash
# Basic usage with default settings
python base.py

# Backtest with specific date range
python base.py --start_date 2023-01-01 --end_date 2023-12-31

# Live trading with specific product
python base.py --product_id ETH-USD --live

# High-frequency trading mode
python base.py --hft --interval ONE_MINUTE
```

### Market Analysis Tools (legacy)

1. **Market Analyzer**
   ```bash
   python market_analyzer.py
   ```

2. **Scalping Analyzer**
   ```bash
   python scalping_analyzer.py
   ```

3. **Memecoin Analyzer**
   ```bash
   python memecoin_analyzer.py
   ```

### AI-Enhanced Analysis (legacy)

1. **AI Market Analysis**
   ```bash
   python prompt_market.py
   ```

2. **AI Trade Recommendations**
   ```bash
   python ai_trade_rec.py --product_id BTC-USD
   ```

### Output Format

The tool provides clear, actionable trading recommendations in the following format:
- Trading Action (BUY/SELL/HOLD)
- Price Target or Entry Point
- Concise rationale for the recommendation

### Prerequisites

- OpenAI API key (set in `config.py`)
- Market analyzer dependencies
- Python 3.11 or higher


# Legacy Tools (Not Maintained)

The sections below (Prompt Market, Memecoin Analyzer, simplified bots, UI experiments) are legacy/not part of the current short-term finder + breakout pipeline. Keep them for reference only. API keys can also be set via `config.py` if you use these tools, but they are not actively maintained.

# Memecoin Analyzer

A Python-based tool for analyzing and monitoring memecoin opportunities in the cryptocurrency market. This tool tracks social metrics, price movements, and various other indicators to help identify potential memecoin trading opportunities.

## Features

- Real-time monitoring of popular memecoins (DOGE, SHIB, PEPE, FLOKI, BONK)
- Social media metrics analysis (Twitter, Reddit, Telegram)
- Price momentum and volume analysis
- Pump pattern detection
- Risk level assessment
- Opportunity scoring system
- Continuous monitoring with customizable intervals

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/memecoin-analyzer.git
cd memecoin-analyzer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the analyzer:
```bash
python memecoin_analyzer.py
```

The tool will start monitoring memecoin opportunities and display reports at regular intervals.

### Sample Output

```
==================================================
Memecoin Opportunities Report - 2024-03-21 14:30:00
==================================================

Coin: DOGE (Dogecoin)
Price: $0.12345678
24h Change: 5.43%
24h Volume: $1,234,567.89
Social Score: 75.50
Risk Level: MEDIUM
Opportunity Score: 65.32
------------------------------
```

## Metrics Explained

- **Social Score**: Weighted combination of Twitter followers, Reddit subscribers, and Telegram members
- **Risk Level**: Categorized as VERY LOW, LOW, MEDIUM, HIGH, or VERY HIGH based on price volatility, volume, and social metrics
- **Opportunity Score**: Overall score (0-100) considering price action, volume, and social engagement

## Disclaimer

This tool is for informational purposes only. Cryptocurrency trading involves substantial risk, and memecoins are particularly volatile. Always conduct your own research before making investment decisions.

## Advanced Market Analyzer

The Advanced Market Analyzer is a sophisticated tool that provides comprehensive market analysis using multiple timeframes, advanced pattern recognition, and sentiment analysis. It combines technical, fundamental, and sentiment data to generate detailed market insights.

### Features

- **Multi-Timeframe Analysis**
  - Primary and secondary timeframe analysis
  - Cross-timeframe confirmation
  - Trend alignment detection

- **Advanced Pattern Recognition**
  - Double Top/Bottom
  - Head and Shoulders
  - Triangle Patterns (Ascending, Descending, Symmetrical)
  - Flag Patterns
  - Cup and Handle
  - Rising/Falling Wedges

- **Market Regime Detection**
  - Trending markets
  - Mean-reverting markets
  - Random walk detection
  - Volatility regime classification
  - Market state transitions

- **Sentiment Analysis**
  - News sentiment integration
  - Social media metrics
  - Market sentiment scoring
  - Sentiment volatility tracking

- **Risk Metrics**
  - Value at Risk (VaR)
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown
  - Volatility Analysis

### Usage

1. **Basic Analysis:**
   ```bash
   python advanced_market_analyzer.py
   ```

2. **Custom Product Analysis:**
   ```bash
   python advanced_market_analyzer.py --product_id ETH-USDC
   ```

3. **Different Timeframe:**
   ```bash
   python advanced_market_analyzer.py --interval FIFTEEN_MINUTE
   ```

4. **JSON Output:**
   ```bash
   python advanced_market_analyzer.py --json
   ```

### Sample Output

```
================================================================================
MARKET ANALYSIS REPORT - BTC-USDC
Generated at: 2024-03-21T15:30:00+00:00
================================================================================

Current Price: $65,432.10

📊 MARKET REGIME
----------------------------------------
Type: Trending
Confidence: 85.00%
Volatility: 2.30%
Trend Strength: 0.65

🎭 MARKET SENTIMENT
----------------------------------------
Category: Bullish
Score: 0.75
Confidence: 80.00%

⏱️ TIMEFRAME ANALYSIS
----------------------------------------
ONE_HOUR:
  Signal: STRONG_BUY
  Confidence: 85.00%
  Pattern: Ascending Triangle
  Pattern Confidence: 78.50%
  Volatility: 2.15%

⚠️ RISK METRICS
----------------------------------------
Volatility: 2.30%
Value at Risk (95%): -3.20%
Max Drawdown: 5.40%
Sharpe Ratio: 2.15
Sortino Ratio: 2.45

💡 TRADE RECOMMENDATIONS
----------------------------------------
Position: LONG
Confidence: 82.50%
Entry Points: $65,400, $65,200, $65,000
Stop Loss: $64,500
Take Profit: $66,500
```

### Key Components

1. **Market Regime Detection**
   - Uses Hurst exponent for trend strength
   - Implements stationarity tests
   - Calculates volatility regimes
   - Tracks regime transitions

2. **Pattern Recognition**
   - Advanced geometric pattern detection
   - Volume confirmation analysis
   - Pattern symmetry scoring
   - Confidence metrics

3. **Sentiment Analysis**
   - News API integration
   - Sentiment scoring system
   - Volatility adjustment
   - Confidence weighting

4. **Risk Management**
   - Dynamic position sizing
   - ATR-based stop losses
   - Multiple take-profit levels
   - Risk-reward optimization

### Integration Example

```python
from advanced_market_analyzer import AdvancedMarketAnalyzer

# Initialize analyzer
analyzer = AdvancedMarketAnalyzer(
    product_id='BTC-USDC',
    primary_interval='ONE_HOUR',
    secondary_intervals=['FIFTEEN_MINUTE', 'SIX_HOUR', 'ONE_DAY']
)

# Get comprehensive analysis
analysis = analyzer.get_advanced_analysis()

# Access specific components
market_regime = analysis['market_regime']
sentiment = analysis['sentiment']
signals = analysis['timeframe_analysis']
risk_metrics = analysis['risk_metrics']
recommendations = analysis['trade_recommendations']
```

### Best Practices

1. **Multi-Timeframe Analysis**
   - Use primary timeframe for main signals
   - Confirm with higher timeframes
   - Check lower timeframes for entry/exit

2. **Pattern Trading**
   - Wait for pattern completion
   - Confirm with volume
   - Use pattern confidence scores
   - Consider market regime

3. **Risk Management**
   - Follow position sizing rules
   - Use suggested stop losses
   - Scale into positions
   - Monitor risk metrics

4. **Sentiment Integration**
   - Consider sentiment direction
   - Watch sentiment volatility
   - Use as confirmation
   - Don't trade against strong sentiment

### Warning

The Advanced Market Analyzer provides sophisticated analysis but should not be used as the sole basis for trading decisions. Always combine with proper risk management and consider market conditions before trading.

## Trading Scripts

### trade_btc_perp.py
A command-line tool for placing leveraged BTC-PERP-INTX trades on Coinbase with take profit and stop loss orders.

**Features:**
- Place leveraged market orders with take profit and stop loss
- Position size validation based on available margin
- Order confirmation with detailed summary
- Automatic size conversion from USD to BTC
- Leverage range from 1x to 20x

**Usage:**
```bash
python trade_btc_perp.py --side [BUY/SELL] --size [USD_AMOUNT] --leverage [1-20] --tp [PRICE] --sl [PRICE] [--no-confirm]
```

**Arguments:**
- `--side`: Trade direction (BUY or SELL)
- `--size`: Position size in USD
- `--leverage`: Leverage amount (between 1-20x)
- `--tp`: Take profit price in USD
- `--sl`: Stop loss price in USD
- `--no-confirm`: Optional flag to skip order confirmation

**Example:**
```bash
python trade_btc_perp.py --side BUY --size 1000 --leverage 5 --tp 45000 --sl 43000
```

### cancel_orders.py
A utility script to cancel all open orders on your Coinbase account.

**Features:**
- Cancels all open orders across all products
- Logging of cancellation process
- Error handling and reporting

**Usage:**
```bash
python cancel_orders.py
```

### close_positions.py
A utility script to close all open positions on your Coinbase account.

**Features:**
- Cancels all open orders first
- Closes all open positions
- Sequential execution to ensure proper order
- Detailed logging of the process

**Usage:**
```bash
python close_positions.py
```

**Process Flow:**
1. Cancels all open orders to prevent conflicts
2. Retrieves all open positions
3. Closes each position with market orders
4. Logs the entire process

### trade_tracker.py
A comprehensive trade tracking and analysis tool that monitors your trading activity in real-time.

**Features:**
- Real-time trade monitoring and logging
- Performance metrics calculation
- Trade history visualization
- PnL tracking and analysis
- Risk metrics computation
- Export capabilities for trade data

**Usage:**
```bash
python trade_tracker.py [--days DAYS] [--export FORMAT]
```

**Arguments:**
- `--days`: Number of days of trade history to analyze (default: 30)
- `--export`: Export format for trade data (CSV/JSON/XLSX)

**Key Metrics Tracked:**
- Win/Loss ratio
- Average profit/loss
- Maximum drawdown
- Sharpe ratio
- Risk-adjusted returns
- Trade duration statistics

### process_trade.py
A utility script for processing and analyzing individual trades with detailed execution analysis.

**Features:**
- Trade execution quality analysis
- Slippage calculation
- Fee analysis and optimization
- Entry/exit timing evaluation
- Trade context recording
- Market impact assessment

**Usage:**
```bash
python process_trade.py --trade-id [TRADE_ID] [--detailed]
```

**Arguments:**
- `--trade-id`: Specific trade ID to analyze
- `--detailed`: Flag for detailed analysis output

**Analysis Components:**
1. Execution Quality
   - Price improvement/slippage
   - Fill rate analysis
   - Timing efficiency

2. Cost Analysis
   - Fee breakdown
   - Cost optimization suggestions
   - Impact on overall PnL

3. Market Context
   - Market conditions during trade
   - Volatility impact
   - Liquidity analysis

**Example Output:**
```
Trade Analysis Report - ID: 12345
================================
Entry Price: $44,500.00
Exit Price: $45,200.00
Slippage: 0.05%
Execution Time: 1.2s
Fee Impact: $12.50
Market Impact: Minimal
Timing Efficiency: 92%
```

**Note:** All trading scripts require valid Coinbase API credentials to be set in `config.py`. Exercise caution when using these scripts as they can affect your real trading positions and orders.

### Risk Warning

These trading scripts execute real trades on your Coinbase account. Please ensure you:
- Understand the risks of leveraged trading
- Double-check all parameters before confirming trades
- Have sufficient funds for the intended positions
- Test with small amounts first
- Monitor your positions after execution

## Market Analyzer UI

The Market Analyzer UI (`market_ui.py`) provides a sophisticated graphical interface for cryptocurrency market analysis and trading. It combines real-time price tracking, technical analysis, and trading capabilities in a user-friendly desktop application.

### Features

- **Real-Time Price Tracking**
  - Live price updates for multiple cryptocurrencies
  - Automatic price refresh with error handling
  - Last update timestamp display

- **Trading Interface**
  - Quick LONG/SHORT market orders
  - Configurable leverage (1x-20x)
  - Adjustable margin size
  - Customizable TP/SL percentages
  - Limit/Market order options
  - One-click position closing

- **Analysis Tools**
  - Multiple timeframe analysis (5m, 1h)
  - Support for various AI models:
    - O1 Mini
    - O3 Mini
    - DeepSeek
    - Grok
    - GPT-4o
  - Real-time analysis output display

- **Auto-Trading Capabilities**
  - Automated trading based on analysis
  - Configurable trading parameters
  - Safety controls and monitoring
  - Auto-stop on successful trade

### Usage

1. Start the Market Analyzer UI:
   ```bash
   python market_ui.py
   ```

2. Configure Trading Settings:
   - Select cryptocurrency pair
   - Set margin amount
   - Adjust leverage
   - Configure TP/SL percentages
   - Choose limit/market order type

3. Analysis Options:
   - Run 5-minute analysis for short-term trading
   - Run 1-hour analysis for longer timeframes
   - Enable auto-trading for automated execution

4. Trading Actions:
   - Use LONG/SHORT buttons for quick market orders
   - Close all positions with one click
   - Monitor real-time price updates
   - View detailed analysis output

### Auto-Trading

The auto-trading feature automatically:
- Runs analysis every 20 minutes
- Monitors for trading opportunities
- Executes trades based on signals
- Stops after successful trade execution
- Provides detailed logging

### Risk Management

The UI includes several risk management features:
- Configurable TP/SL levels
- Position size limits
- Leverage controls
- Quick position closing
- Real-time price monitoring

### Requirements

- Python 3.11 or higher
- CustomTkinter library
- Active Coinbase API credentials
- Stable internet connection

### Installation

1. Install required dependencies:
   ```bash
   pip install customtkinter requests
   ```

2. Ensure Coinbase API credentials are configured in `config.py`

3. Launch the UI:
   ```bash
   python market_ui.py
   ```

# Trading Performance Analyzer

The Trading Performance Analyzer (`trade_analyzer.py`) is a powerful tool for evaluating trading performance metrics from your trading history. It processes trading data stored in markdown format and generates comprehensive performance reports.

### Features

- **Basic Trading Metrics**
  - Total number of trades
  - Win rate and win/loss ratio
  - Average profit per trade
  - Total profit/loss

- **Risk-Adjusted Returns**
  - Sharpe ratio (annualized)
  - Maximum drawdown analysis
  - Standard deviation of returns
  - Risk/Reward ratio statistics

- **Risk Management Metrics**
  - Average trade probability
  - Average leverage used
  - Drawdown period analysis
  - Position sizing statistics

### Usage

1. Prepare your trading data in markdown format:
```markdown
| No. | Timestamp | SIDE | ENTRY | Take Profit | Stop Loss | Probability | Confidence | R/R Ratio | Volume Strength | Outcome | Outcome % | Leverage | Margin |
|-----|-----------|------|-------|-------------|-----------|-------------|------------|------------|-----------------|----------|-----------|----------|---------|
| 1   | 2024-03-21| LONG | 65000 | 66000      | 64000    | 75%        | Strong     | 2.0       | High           | SUCCESS  | 1.5      | 10x      | 100    |
```

2. Run the analyzer:
```bash
python trade_analyzer.py
```

### Sample Output
```
=== Trading Performance Report ===

Basic Metrics:
Total Trades: 50
Win Rate: 65.00%
Win/Loss Ratio: 1.86
Average Profit per Trade: 2.45%
Total Profit: 122.50%

Risk Metrics:
Sharpe Ratio (Annualized): 3.25
Maximum Drawdown: -15.40%
Standard Deviation: 4.82%
Average R/R Ratio: 2.15
Average Trade Probability: 72.50%
Average Leverage: 10.00x

Largest Drawdown Periods:
From 2024-02-01 to 2024-02-15: -15.40%
```

### Key Benefits

1. **Performance Tracking**: Monitor your trading strategy's effectiveness through key performance indicators.
2. **Risk Assessment**: Evaluate risk-adjusted returns and identify potential areas of improvement.
3. **Pattern Recognition**: Identify periods of strong performance and challenging drawdowns.
4. **Strategy Optimization**: Use metrics to refine your trading approach and risk management.

# Markdown to CSV Converter

This simple Python script converts a markdown table to CSV format.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your markdown file (automated_trades.md) in the same directory as the script
2. Run the script:
```bash
python convert_to_csv.py
```

The script will generate `automated_trades.csv` in the same directory.

## Input Format

The script expects a markdown table with the following format:
- Table should start with a header row
- Columns should be separated by pipes (|)
- The first line of content should be the header
- The second line should be the markdown separator (|----|)

## Analysis Components

### Technical Analysis

The `TechnicalAnalysis` class implements various technical indicators and analysis methods:

- Multiple technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Trend identification and analysis
- Volume analysis and On-Balance Volume (OBV)
- Market condition analysis (Bull/Bear market detection)
- Dynamic support/resistance levels
- Volatility analysis and ATR calculations
- Fibonacci retracements
- Ichimoku Cloud analysis

### Machine Learning Integration

The `MLSignal` class integrates machine learning models for enhanced prediction:

- XGBoost price prediction
- Random Forest classification
- Ensemble methods
- Feature engineering from technical indicators
- Real-time model updates
- Confidence scoring

### AI-Enhanced Analysis

The AI analysis components leverage multiple models for advanced insights:

1. **Market Analysis Models**
   - GPT-4: Advanced pattern recognition
   - DeepSeek: Price prediction
   - Grok: Real-time market insights
   - Custom ensemble predictions

2. **Pattern Recognition**
   - Complex pattern detection
   - Multi-timeframe confirmation
   - Volume profile analysis
   - Market regime detection

3. **Sentiment Analysis**
   - News sentiment scoring
   - Social media analysis
   - Market sentiment indicators
   - Sentiment-based signals

### High-Frequency Trading

The HFT components provide tools for rapid trading:

1. **Order Book Analysis**
   - Real-time depth analysis
   - Liquidity detection
   - Spread analysis
   - Order flow patterns

2. **Execution Engine**
   - Low-latency order placement
   - Smart order routing
   - Anti-gaming logic
   - Execution quality analysis

3. **Risk Management**
   - Real-time position monitoring
   - Dynamic stop-loss adjustment
   - Exposure limits
   - Risk metrics calculation

## Trading Tools

### Market Analyzer UI

The graphical interface provides:

1. **Real-Time Monitoring**
   - Price and volume charts
   - Technical indicators
   - Order book visualization
   - Position tracking

2. **Trading Controls**
   - Quick order entry
   - Position management
   - Risk parameter adjustment
   - Strategy selection

3. **Analysis Tools**
   - Multiple timeframe analysis
   - Pattern detection
   - Sentiment indicators
   - Risk metrics

### Performance Analytics

The performance tracking system offers:

1. **Trade Analysis**
   - Win/loss statistics
   - Risk-adjusted returns
   - Drawdown analysis
   - Position sizing effectiveness

2. **Risk Metrics**
   - Sharpe ratio
   - Sortino ratio
   - Maximum drawdown
   - Value at Risk (VaR)

3. **Strategy Evaluation**
   - Strategy performance comparison
   - Market regime analysis
   - Parameter optimization
   - Backtest results

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

1. **Code Style**
   - Follow PEP 8 guidelines
   - Use type hints
   - Write comprehensive docstrings
   - Maintain test coverage

2. **Testing**
   - Write unit tests for new features
   - Include integration tests
   - Test with different market conditions
   - Verify performance impact

3. **Documentation**
   - Update README.md
   - Add function/class documentation
   - Include usage examples
   - Document configuration options

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Disclaimer

This Bitcoin Trading Bot is for educational and research purposes only. It is not intended to be used for actual trading. Always consult with a qualified financial advisor before making any investment decisions. The authors and contributors are not responsible for any financial losses incurred from using this software.

### Risk Warning

- Cryptocurrency trading involves substantial risk
- High-frequency trading can result in significant losses
- Always start with small amounts
- Test strategies thoroughly before live trading
- Monitor positions and risk levels continuously
- Keep API keys secure and never share them

## Tools

### run_backtests.py

This tool runs multiple backtests over predefined date ranges and collects all the results into a single report file. It's designed to evaluate strategy performance across different market periods. This tool specifically refers to the `simplified_trading_bot.py` implementation.

#### Usage

```bash
python run_backtests.py
```

#### Functionality

The script:
1. Runs backtests for multiple predefined date ranges (e.g., Jan-Feb 2023, Mar-Apr 2023, etc.)
2. Executes `backtest_trading_bot.py` for each date range with appropriate parameters
3. Captures the output and extracts the backtest results
4. Combines all results into a single report file with timestamp
5. Handles timeouts and errors gracefully

#### Output

The tool generates a single text file (`all_backtest_reports_TIMESTAMP.txt`) containing:
- A header with generation timestamp
- Separate sections for each backtest period
- Complete backtest results for each period
- Clear separation between different backtest periods

This consolidated report makes it easy to compare strategy performance across different market conditions and timeframes.
