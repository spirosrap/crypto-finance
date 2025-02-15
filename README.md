# Bitcoin Trading Bot

## Overview

This Bitcoin Trading Bot implements a sophisticated trading strategy using traditional technical analysis signals, machine learning models (XGBoost), and advanced market analysis techniques for price prediction. It offers both backtesting capabilities and live trading simulations.

## Key Components

1. **Backtester**: Allows historical performance analysis and live trading simulations.
2. **Technical Analysis**: Implements various technical indicators and analysis methods.
3. **CoinbaseService**: Handles interactions with the Coinbase API.
4. **HistoricalData**: Manages retrieval and storage of historical price data.
5. **API**: Provides an interface for external interactions with the trading bot.
6. **MLSignal**: Implements machine learning models for price prediction and signal generation.
7. **BitcoinPredictionModel**: A specialized model for Bitcoin price prediction.
8. **HighFrequencyStrategy**: Implements high-frequency trading strategies.

## Features

- Backtesting over specified date ranges
- Live (paper) trading simulations
- Multiple technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Advanced market condition analysis (Bull/Bear market detection)
- Dynamic trade sizing based on market conditions and volatility
- Risk management with trailing stop-loss and ATR-based position sizing
- State persistence for continuous operation
- Visual trade and balance history plots
- Machine learning integration for enhanced prediction accuracy
- High-frequency trading capabilities
- Sentiment analysis integration

## Prerequisites

### Python Environment Setup

```bash
conda create -n myenv python=3.11
conda activate myenv
```

### Required Libraries

```bash
pip install coinbase-advanced-py statsmodels yfinance newsapi-python schedule hmmlearn scikit-learn scikit-fuzzy xgboost joblib scikit-optimize shap pandas numpy matplotlib tqdm requests talib
```

### TA-Lib Installation

TA-Lib is required for technical analysis calculations. Install it based on your operating system:

#### Linux
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

#### macOS
```bash
brew install ta-lib
```

After installing TA-Lib, install the Python wrapper:
```bash
pip install TA-Lib
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bitcoin-trading-bot.git
   cd bitcoin-trading-bot
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `config.py` file in the root directory of the project with your API keys.

## API Keys Configuration

The bot requires several API keys for full functionality. Create a `config.py` file in the root directory with the following structure:

```python
# Coinbase API credentials
API_KEY = "your_coinbase_api_key"
API_SECRET = "your_coinbase_api_secret"

# News API for sentiment analysis
NEWS_API_KEY = "your_news_api_key"

# Twitter API credentials (optional - for social sentiment analysis)
BEARER_TOKEN = "your_twitter_bearer_token"
CONSUMER_KEY = "your_twitter_consumer_key"
CONSUMER_SECRET = "your_twitter_consumer_secret"
ACCESS_TOKEN = "your_twitter_access_token"
ACCESS_TOKEN_SECRET = "your_twitter_access_token_secret"

# AI Model API Keys (optional - for enhanced analysis)
OPENAI_KEY = "your_openai_api_key"
DEEPSEEK_KEY = "your_deepseek_api_key"
OPENROUTER_API_KEY = "your_openrouter_api_key"
XAI_KEY = "your_xai_api_key"
HYPERBOLIC_KEY = "your_hyperbolic_api_key"
```

### Required API Keys
- **Coinbase API**: Required for trading functionality
  - Get your API credentials from [Coinbase Advanced Trade](https://www.coinbase.com/settings/api)
  - Set `API_KEY` and `API_SECRET`

### Optional API Keys
- **News API**: For news sentiment analysis
  - Get your API key from [NewsAPI](https://newsapi.org/)
  - Set `NEWS_API_KEY`

- **Twitter API**: For social sentiment analysis
  - Get your credentials from [Twitter Developer Portal](https://developer.twitter.com/)
  - Set all Twitter-related keys

- **AI Model Keys**: For enhanced market analysis
  - OpenAI: Get from [OpenAI Platform](https://platform.openai.com/)
  - DeepSeek: Get from DeepSeek's platform
  - OpenRouter: Get from OpenRouter's platform
  - XAI: Get from XAI's platform
  - Hyperbolic: Get from Hyperbolic's platform

### Security Notes
- Never commit your `config.py` file to version control
- Keep your API keys secure and rotate them periodically
- Use environment variables in production environments
- Consider using a `.env` file for local development

## Usage

The main program for the Bitcoin Trading Bot is `base.py`. Here are some common usage scenarios:

### Backtesting Options

1. Specific date range:
   ```bash
   python base.py --start_date 2023-01-01 --end_date 2023-12-31
   ```

2. Bear market period:
   ```bash
   python base.py --bearmarket
   ```

3. Bull market period:
   ```bash
   python base.py --bullmarket
   ```

4. Year-to-date:
   ```bash
   python base.py --ytd
   ```

### Live Trading Simulation

To run the program in live trading simulation mode:

```bash
python base.py
```

### Additional Options

- Skip backtesting:
  ```bash
  python base.py --skip_backtest
  ```

- Specify a different product ID (default is BTC-USDC):
  ```bash
  python base.py --product_id ETH-USD
  ```

- Change granularity:
  ```bash
  python base.py --granularity ONE_MINUTE
  ```

For a full list of options, run:
```bash
python base.py --help
```

## Advanced Usage

### High-Frequency Trading

To use the high-frequency trading strategy:

```bash
python high_frequency_strategy.py
```

### Running All Commands

To run a series of backtests with different parameters:

```bash
python run_all_commands.py
```

### Continuous Backtesting

For continuous backtesting:

```bash
python run_base.py
```

## Technical Analysis

The `TechnicalAnalysis` class in `technicalanalysis.py` is a core component of our trading strategy. It implements various technical indicators and analysis methods to generate trading signals and assess market conditions.

### Key Features

- Multiple technical indicators (RSI, MACD, Bollinger Bands, Stochastic Oscillator, etc.)
- Trend identification and analysis
- Volume analysis and On-Balance Volume (OBV)
- Market condition analysis (Bull/Bear market detection)
- Dynamic support/resistance levels
- Volatility analysis and ATR calculations
- Fibonacci retracements
- Ichimoku Cloud analysis

## Machine Learning Integration

The `MLSignal` class in `ml_model.py` integrates machine learning models to enhance prediction accuracy. It uses ensemble methods combining Random Forest, Gradient Boosting, and Logistic Regression models.

## Contributing

Contributions to improve the Bitcoin Trading Bot are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Disclaimer

This Bitcoin Trading Bot is for educational and research purposes only. It is not intended to be used for actual trading. Always consult with a qualified financial advisor before making any investment decisions. The authors and contributors are not responsible for any financial losses incurred from using this software.

## 🚀 Quick Start

1. **Set up environment:**
   ```bash
   conda create -n btc-trader python=3.11
   conda activate btc-trader
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys:**
   Create `config.py` with your API credentials:
   ```python
   API_KEY = "your_coinbase_api_key"
   API_SECRET = "your_coinbase_api_secret"
   NEWS_API_KEY = "your_news_api_key"  # Optional for sentiment analysis
   ```

   Replace `your_coinbase_api_key`, `your_coinbase_api_secret`, and `your_news_api_key` with your actual API keys from Coinbase and NewsAPI respectively.

   Note: Keep your `config.py` file secure and never share it publicly or commit it to version control.

4. **Run basic simulation:**
   ```bash
   python base.py --ytd  # Backtest using year-to-date data
   ```

## 📊 Trading Strategies

The bot combines multiple analysis methods for trading decisions:

1. **Technical Analysis**
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - Volume analysis
   - Support/Resistance levels
   - ATR (Average True Range)

2. **Machine Learning**
   - XGBoost price prediction
   - Random Forest classification
   - Ensemble methods
   - Feature engineering from technical indicators

3. **Market Analysis**
   - Bull/Bear market detection
   - Volatility tracking
   - Trend analysis
   - News sentiment integration

## 💻 Usage Examples

### Basic Backtesting

```

## Market Analyzer

The Market Analyzer is a sophisticated tool for analyzing cryptocurrency markets and generating trading signals based on technical analysis. It provides comprehensive market analysis including:

### Features

- **Technical Indicators Analysis**
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - ADX (Average Directional Index)
  - Volume Analysis

- **Market Condition Detection**
  - Trend Analysis
  - Support/Resistance Levels
  - Consolidation Patterns
  - Breakout/Breakdown Detection

- **Trading Signals**
  - Strong Buy/Sell Signals
  - Conservative Buy/Sell Recommendations
  - Hold/Neutral Signals
  - Signal Confidence Levels

- **Risk Management**
  - Stop Loss Recommendations
  - Take Profit Targets
  - Position Sizing Suggestions
  - Risk-Reward Ratios

### Usage

1. **Basic Usage:**
   ```bash
   python market_analyzer.py
   ```

2. **Analyze Specific Product:**
   ```bash
   python market_analyzer.py --product_id BTC-USDC
   ```

3. **Change Time Interval:**
   ```bash
   python market_analyzer.py --granularity FIFTEEN_MINUTE
   ```

4. **List Available Options:**
   ```bash
   python market_analyzer.py --list-products
   python market_analyzer.py --list-granularities
   ```

### Supported Products

- BTC-USDC
- ETH-USDC
- SOL-USDC
- DOGE-USDC
- XRP-USDC
- ADA-USDC
- MATIC-USDC
- LINK-USDC
- DOT-USDC
- UNI-USDC
- SHIB-USDC
### Available Granularities

- ONE_MINUTE
- FIVE_MINUTE
- FIFTEEN_MINUTE
- THIRTY_MINUTE
- ONE_HOUR
- TWO_HOUR
- SIX_HOUR
- ONE_DAY

### Analysis Components

1. **Technical Indicators:**
   - Relative Strength Index (RSI)
   - Moving Average Convergence Divergence (MACD)
   - Bollinger Bands
   - Average Directional Index (ADX)
   - Moving Average Crossovers
   - Volume Analysis
   - Trend Strength Analysis

2. **Risk Metrics:**
   - Average True Range (ATR)
   - Volatility Assessment
   - Risk Level Classification
   - Suggested Stop Loss Levels
   - Recommended Take Profit Levels
   - Maximum Position Size Calculation

3. **Market Conditions:**
   - Bull Market
   - Bear Market
   - Bullish
   - Bearish
   - Neutral

4. **Signal Types:**
   - STRONG BUY
   - BUY
   - HOLD
   - SELL
   - STRONG SELL

### Sample Output

```
=== Market Analysis Report ===
Timestamp: 2024-11-15T20:47:29.866625
Product: BTC-USDC
Current Price: $44,123.45

Signal: BUY
Confidence: 75.5%
Market Condition: Bullish

Key Indicators:
  rsi: 58.32
  macd: 125.45
  macd_signal: 100.23
  macd_histogram: 25.22
  bollinger_upper: 45000.00
  bollinger_middle: 44000.00
  bollinger_lower: 43000.00
  adx: 28.45
  trend_direction: Uptrend

Risk Metrics:
  atr: 450.2345
  volatility: 0.0234
  risk_level: medium
  stop_loss: 43500.00
  take_profit: 45000.00
  max_position_size: 0.5000

Recommendation:
Bullish conditions detected. Look for entry points for a long position.
```

### Signal Generation Process

The Market Analyzer uses a sophisticated weighted approach to generate signals:

1. **Data Collection:**
   - Fetches historical data for the specified timeframe
   - Validates and formats candle data

2. **Technical Analysis:**
   - Calculates multiple technical indicators
   - Evaluates trend strength and direction
   - Assesses market conditions

3. **Signal Weighting:**
   - Applies product-specific weights to each indicator
   - Adjusts for market conditions
   - Considers volume and volatility

4. **Risk Assessment:**
   - Calculates risk metrics
   - Determines position sizing
   - Suggests entry/exit points

5. **Final Recommendation:**
   - Generates confidence-based signals
   - Provides detailed trading recommendations
   - Includes risk management suggestions

### Integration with Trading Bot

The Market Analyzer can be integrated with the trading bot for automated trading:

```python
from market_analyzer import MarketAnalyzer

analyzer = MarketAnalyzer(product_id='BTC-USDC', candle_interval='ONE_HOUR')
analysis = analyzer.get_market_signal()
signal = analysis['signal']
confidence = analysis['confidence']
```

## Scalping Analyzer

The Scalping Analyzer is a specialized tool designed for high-frequency trading and short-term price movements. It provides rapid analysis and real-time signals optimized for scalping strategies.

### Features

- **Real-Time Analysis**
  - Price action patterns
  - Volume profile analysis
  - Order book depth
  - Market microstructure
  - Tick-by-tick data analysis

- **Scalping-Specific Indicators**
  - Momentum indicators (1-5 minute timeframes)
  - Volume weighted average price (VWAP)
  - Order flow analysis
  - Price momentum index
  - Market depth imbalances
  - Tick volume analysis

- **Risk Management Tools**
  - Dynamic stop-loss calculation
  - Quick take-profit targets
  - Position sizing calculator
  - Risk exposure monitor
  - Slippage estimation
  - Spread analysis

- **Market Execution Analysis**
  - Entry/exit point optimization
  - Spread cost analysis
  - Execution speed monitoring
  - Liquidity analysis
  - Trading cost estimation

### Usage

1. **Basic Usage:**
   ```bash
   python scalping_analyzer.py
   ```

2. **Custom Configuration:**
   ```bash
   python scalping_analyzer.py --product_id ETH-USDC --interval ONE_MINUTE --risk 0.01
   ```

3. **Advanced Options:**
   ```bash
   python scalping_analyzer.py --depth 10 --volume_threshold 1.5 --momentum_period 3
   ```

### Supported Timeframes

- ONE_MINUTE
- THREE_MINUTE
- FIVE_MINUTE

### Key Components

1. **Price Action Analysis**
   - Support/resistance levels
   - Price patterns
   - Momentum shifts
   - Volatility breaks
   - Range analysis

2. **Volume Analysis**
   - Volume spikes
   - Buy/sell pressure
   - Volume profile
   - Cumulative volume delta
   - Time and sales analysis

3. **Order Book Analysis**
   - Depth imbalances
   - Large orders detection
   - Spread analysis
   - Liquidity pools
   - Order flow patterns

4. **Risk Parameters**
   - Maximum position size
   - Per-trade risk limit
   - Maximum drawdown
   - Quick exit rules
   - Profit targets

### Sample Output

```
=== Scalping Analysis Report ===
Timestamp: 2024-11-15T14:30:15.123456
Product: ETH-USDC
Current Price: $2,456.78

Market Conditions:
  Spread: 0.12
  Depth Ratio: 1.45
  Volume Delta: +2500
  Momentum: Strong Bullish
  
Quick Stats:
  1min ROC: +0.15%
  Volume Spike: 2.3x
  Depth Imbalance: 65% Buy
  
Entry Zones:
  Support 1: $2,455.50
  Support 2: $2,454.25
  Resistance 1: $2,458.00
  Resistance 2: $2,459.50

Risk Parameters:
  Stop Loss: $2,454.00 (-0.11%)
  Take Profit 1: $2,458.00 (+0.05%)
  Take Profit 2: $2,459.50 (+0.11%)
  Max Position: 0.75 ETH
```

### Best Practices for Scalping

1. **Pre-Trading Checklist**
   - Check spread levels
   - Verify sufficient liquidity
   - Confirm market volatility
   - Review recent price action
   - Check trading costs

2. **Risk Management**
   - Use tight stop losses
   - Define clear exit points
   - Limit position sizes
   - Monitor cumulative risk
   - Track win/loss ratio

3. **Execution Guidelines**
   - Wait for confirmation signals
   - Monitor order book changes
   - Use limit orders when possible
   - Avoid high spread periods
   - Exit quickly if wrong

### Integration Example

```python
from scalping_analyzer import ScalpingAnalyzer

# Initialize analyzer
scalper = ScalpingAnalyzer(
    product_id='ETH-USDC',
    interval='ONE_MINUTE',
    risk_per_trade=0.01,
    max_position_size=1.0
)

# Get real-time analysis
analysis = scalper.get_scalping_signals()

# Check for trading opportunities
if analysis['signal'] == 'LONG':
    entry_price = analysis['entry_price']
    stop_loss = analysis['stop_loss']
    take_profit = analysis['take_profit']
    position_size = analysis['position_size']
    # Execute trade...
```

### Warning

Scalping requires extreme attention to detail and quick decision-making. This strategy involves frequent trading and can result in significant transaction costs. It is recommended only for experienced traders who understand the risks and have tested their strategy thoroughly in a paper trading environment.

## AI-Powered Market Analysis

The `prompt_market.py` tool provides AI-powered market analysis and trading recommendations by combining the comprehensive market analysis from `market_analyzer.py` with advanced language models.

### Features

- AI-driven trading recommendations
- Natural language analysis of market conditions
- Clear BUY/SELL/HOLD signals with price targets
- Concise rationale for trading decisions

### Usage

1. **Basic Analysis:**
   ```bash
   python prompt_market.py
   ```

2. **Analyze Specific Product:**
   ```bash
   python prompt_market.py --product_id ETH-USDC
   ```

3. **Change Time Interval:**
   ```bash
   python prompt_market.py --granularity ONE_HOUR
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

**Note:** All trading scripts require valid Coinbase API credentials to be set in `config.py`. Exercise caution when using these scripts as they can affect your real trading positions and orders.

### Risk Warning

These trading scripts execute real trades on your Coinbase account. Please ensure you:
- Understand the risks of leveraged trading
- Double-check all parameters before confirming trades
- Have sufficient funds for the intended positions
- Test with small amounts first
- Monitor your positions after execution