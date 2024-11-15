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

3. Create a `config.py` file in the root directory of the project with your API keys:
   ```python
   API_KEY = "your_coinbase_api_key"
   API_SECRET = "your_coinbase_api_secret"
   NEWS_API_KEY = "your_news_api_key"
   ```

   Replace `your_coinbase_api_key`, `your_coinbase_api_secret`, and `your_news_api_key` with your actual API keys from Coinbase and NewsAPI respectively.

   Note: Keep your `config.py` file secure and never share it publicly or commit it to version control.

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

The Market Analyzer is a powerful command-line tool that provides real-time technical analysis and trading signals for various cryptocurrency pairs. It analyzes market conditions using multiple technical indicators and generates detailed trading recommendations.

### Market Analyzer Features

- Real-time market analysis with multiple technical indicators
- Support for multiple cryptocurrency pairs
- Configurable time intervals (granularity)
- Risk metrics calculation
- Market condition assessment
- Confidence-based signals
- Detailed recommendations
- Performance monitoring

### Market Analyzer Usage

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
