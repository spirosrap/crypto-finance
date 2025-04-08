# Bitcoin Trading Bot

## Overview

This Bitcoin Trading Bot implements a sophisticated trading strategy using traditional technical analysis signals, machine learning models (XGBoost), and advanced market analysis techniques for price prediction. It offers both backtesting capabilities and live trading simulations, with support for high-frequency trading and AI-powered market analysis.

## Key Components

1. **Backtester**: Allows historical performance analysis and live trading simulations.
2. **Technical Analysis**: Implements various technical indicators and analysis methods.
3. **CoinbaseService**: Handles interactions with the Coinbase API.
4. **HistoricalData**: Manages retrieval and storage of historical price data.
5. **API**: Provides an interface for external interactions with the trading bot.
6. **MLSignal**: Implements machine learning models for price prediction and signal generation.
7. **BitcoinPredictionModel**: A specialized model for Bitcoin price prediction.
8. **HighFrequencyStrategy**: Implements high-frequency trading strategies.
9. **MarketAnalyzer**: Advanced market analysis with multi-timeframe support.
10. **ScalpingAnalyzer**: Specialized tool for high-frequency trading.
11. **MemeAnalyzer**: Tool for analyzing memecoin opportunities.
12. **PerformanceAnalyzer**: Trading performance tracking and analysis.

## Features

### Core Features
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

### Advanced Features
- AI-powered market analysis with multiple model support (GPT-4, DeepSeek, Grok)
- Real-time scalping analysis with order book depth integration
- Memecoin opportunity detection and analysis
- Multi-timeframe pattern recognition
- Advanced risk metrics (VaR, Sharpe Ratio, Sortino Ratio)
- Comprehensive performance tracking and analysis
- Graphical user interface for market analysis and trading
- Automated trading with configurable parameters

## Prerequisites

### Python Environment Setup

```bash
conda create -n myenv python=3.11
conda activate myenv
```

### Required Libraries

```bash
pip install coinbase-advanced-py statsmodels yfinance newsapi-python schedule hmmlearn scikit-learn scikit-fuzzy xgboost joblib scikit-optimize shap pandas numpy matplotlib tqdm requests talib customtkinter tensorflow torch transformers openai deepseek-api openrouter-py hyperbolic-api
```

### Additional Dependencies

#### TA-Lib Installation

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

#### GUI Dependencies
For the Market Analyzer UI:
```bash
pip install customtkinter pillow
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

## Usage

The bot provides multiple tools and interfaces for different trading strategies and analysis needs.

### Basic Trading Bot

The main program (`base.py`) supports both backtesting and live trading:

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

### Market Analysis Tools

1. **Market Analyzer**
   ```bash
   # Basic market analysis
   python market_analyzer.py
   
   # Analysis with specific timeframe
   python market_analyzer.py --interval FIFTEEN_MINUTE
   
   # Multi-timeframe analysis
   python market_analyzer.py --timeframes ONE_MINUTE FIVE_MINUTE ONE_HOUR
   ```

2. **Scalping Analyzer**
   ```bash
   # Real-time scalping analysis
   python scalping_analyzer.py
   
   # Custom configuration
   python scalping_analyzer.py --depth 10 --volume_threshold 1.5
   ```

3. **Memecoin Analyzer**
   ```bash
   # Monitor memecoin opportunities
   python memecoin_analyzer.py
   
   # Specific coin analysis
   python memecoin_analyzer.py --coin DOGE
   ```

### AI-Enhanced Analysis

1. **AI Market Analysis**
   ```bash
   # Basic AI analysis
   python prompt_market.py
   
   # Analysis with specific model
   python prompt_market.py --model gpt4
   ```

2. **Pattern Recognition**
   ```bash
   # Run pattern detection
   python pattern_recognition.py
   
   # Save pattern analysis
   python pattern_recognition.py --save_plots
   ```

### Trading Tools

1. **Trade BTC Perpetual**
   ```bash
   # Place a leveraged trade
   python trade_btc_perp.py --side BUY --size 1000 --leverage 5 --tp 45000 --sl 43000
   ```

2. **Position Management**
   ```bash
   # Close all positions
   python close_positions.py
   
   # Cancel all orders
   python cancel_orders.py
   ```

3. **Trade Tracking**
   ```bash
   # View recent trades
   python trade_tracker.py
   
   # Export trade history
   python trade_tracker.py --days 30 --export CSV
   ```

### Graphical Interface

Launch the Market Analyzer UI:
```bash
python market_ui.py
```

### Additional Tools

1. **Performance Analysis**
   ```bash
   # Analyze trading performance
   python trade_analyzer.py
   
   # Generate detailed report
   python trade_analyzer.py --detailed --export PDF
   ```

2. **Data Management**
   ```bash
   # Convert trading data format
   python convert_to_csv.py
   
   # Clean historical data
   python clean_data.py
   ```

### Common Options

Most tools support these common flags:
- `--help`: Show help message
- `--verbose`: Enable detailed logging
- `--config`: Specify custom config file
- `--output`: Set output directory
- `--debug`: Enable debug mode

### Environment Variables

You can override config settings with environment variables:
```bash
export TRADING_MODE=live
export RISK_LEVEL=conservative
python base.py
```

### Simplified Trading Bot

```bash
# Basic usage with default settings
python simplified_trading_bot.py

# Custom configuration
python simplified_trading_bot.py --product_id BTC-USDC --margin 100 --leverage 5
```

The simplified trading bot provides a streamlined trading experience with:
- Single coin trading (BTC-USDC by default)
- Fixed 5-minute timeframe analysis
- Simple strategy combining RSI, EMA, and volume indicators
- Automated trade execution with configurable TP/SL
- Risk management with position sizing and leverage control
- No AI/ML components for faster execution

### Market UI

```bash
# Launch the graphical user interface
python market_ui.py
```

The Market UI provides a comprehensive graphical interface for:
- Real-time price monitoring and market analysis
- Multiple trading modes (Auto-Trading, Simplified Trading)
- Configurable trading parameters (margin, leverage, TP/SL)
- Trade history and performance tracking
- Multi-timeframe analysis with alignment detection
- Order management and position monitoring
- Risk management controls
- Support for multiple trading pairs (BTC, ETH, DOGE, SOL, SHIB)

Key Features:
- **Auto-Trading Mode**: AI-powered trading with multiple timeframe analysis
- **Simplified Trading Mode**: Fast execution using basic technical indicators
- **Quick Market Orders**: One-click long/short positions with preset parameters
- **Trade Status Monitoring**: Real-time updates on open positions and orders
- **Risk Controls**: Configurable leverage, margin, and TP/SL levels
- **Market Analysis**: Support for multiple AI models and timeframe alignment
- **Performance Tracking**: Trade history with win/loss statistics

Configuration:
- Uses the same `config.py` file for API keys
- Supports both market and limit orders
- Customizable timeframes and analysis parameters
- Persistent settings saved between sessions

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