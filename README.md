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
