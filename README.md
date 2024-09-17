# Bitcoin Trading Bot

* Trading strategy using traditional signals.
* Price prediction using ml models (xgboost).

## Backtester

The Backtester is a crucial component of our trading system, allowing for historical performance analysis and live trading simulations. It's implemented in the `backtester.py` file.

### Key Features

- Performs backtesting over specified date ranges
- Runs live trading simulations using real-time market data
- Implements various trading constraints:
  - Cooldown period between trades
  - Maximum trades per day
  - Minimum price change threshold
  - Drawdown threshold for stop-loss
- Adjusts trade size based on market conditions
- Uses trailing stop-loss for risk management
- Saves and loads trading state for continuity
- Generates visual plots of trades and balance history

### Usage

To use the Backtester, you'll need to initialize it with a Trader object:

## Running the Trading Bot

The main program for the Bitcoin Trading Bot is `base.py`. It provides various options for backtesting and live trading simulation. Here are some examples of how to run the program:

### Basic Usage

To run the program with default settings (1-year backtest):

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

To run the program in live trading mode:

```bash
python base.py --live
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


