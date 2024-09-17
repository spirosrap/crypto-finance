# Bitcoin Trading Bot

## Overview

This Bitcoin Trading Bot implements a sophisticated trading strategy using traditional technical analysis signals and machine learning models (XGBoost) for price prediction. It offers both backtesting capabilities and live trading simulations.

## Key Components

1. **Backtester**: Allows historical performance analysis and live trading simulations.
2. **Technical Analysis**: Implements various technical indicators and analysis methods.
3. **CoinbaseService**: Handles interactions with the Coinbase API.
4. **HistoricalData**: Manages retrieval and storage of historical price data.
5. **API**: Provides an interface for external interactions with the trading bot.

## Features

- Backtesting over specified date ranges
- Live (paper) trading simulations
- Multiple technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Market condition analysis (Bull/Bear market detection)
- Dynamic trade sizing based on market conditions
- Risk management with trailing stop-loss
- State persistence for continuous operation
- Visual trade and balance history plots

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

4. You're now ready to use the Bitcoin Trading Bot!

## Usage

The main program for the Bitcoin Trading Bot is `base.py`. Here are some common usage scenarios:

### Basic Backtesting

Run a 1-year backtest with default settings:

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

The live trading simulation mode doesn't execute real-time trades. Instead, it simulates trades as if they had started a few weeks ago, providing a more extensive historical context. This allows you to observe when the bot would perform its next buy or sell action based on current market conditions.

To run the program in live trading simulation mode:

```bash
python base.py
```

### Additional Options

- Skip backtesting:
  ```bash
  python base.py --skip_backtest
  ```

- Specify a different product ID (default is BTC-USD):
  ```bash
  python base.py --product_id ETH-USD
  ```

You can combine these options as needed. For example, to run a live simulation with ETH-USD:

For a full list of options, run:
```bash
python base.py --help
```

This will display all available command-line arguments and their descriptions.

## Technical Analysis

The `TechnicalAnalysis` class in `technicalanalysis.py` is a core component of our trading strategy. It implements various technical indicators and analysis methods to generate trading signals and assess market conditions.

### Key Features

- Calculates multiple technical indicators:
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Stochastic Oscillator
  - Average True Range (ATR)
  - Simple Moving Averages (SMA)
- Implements trend identification
- Provides volume analysis
- Generates combined trading signals based on multiple indicators
- Analyzes market conditions (Bull/Bear market, Bullish/Bearish trends)
- Implements dynamic support/resistance levels
- Detects pullbacks for potential entry points
- Adjusts strategies based on market volatility

### Main Methods

- `generate_combined_signal`: Produces a trading signal by combining multiple technical indicators and market conditions.
- `analyze_market_conditions`: Determines the overall market state (Bull Market, Bear Market, Bullish, Bearish, or Neutral).
- `compute_rsi`, `compute_macd`, `compute_bollinger_bands`: Calculate individual technical indicators.
- `identify_trend`: Determines the current market trend.
- `analyze_volume`: Assesses the current trading volume relative to recent averages.
- `detect_pullback`: Identifies potential pullback situations for trading opportunities.

### Usage

The `TechnicalAnalysis` class is typically used within the `Backtester` and main trading logic to generate signals and analyze market conditions. It's initialized with a `CoinbaseService` object:
```
python
from technicalanalysis import TechnicalAnalysis
from coinbaseservice import CoinbaseService
coinbase_service = CoinbaseService(api_key, api_secret)
ta = TechnicalAnalysis(coinbase_service)
```


### Generate a combined trading signal

```
signal = ta.generate_combined_signal(rsi, macd, signal, histogram, candles, market_conditions)
```

###  Analyze market conditions
```
market_state = ta.analyze_market_conditions(candles)
```

### Calculate technical indicators
```
rsi = ta.compute_rsi(candles)
macd = ta.compute_macd(candles)
signal = ta.compute_signal(candles)
histogram = ta.compute_histogram(candles)
```


