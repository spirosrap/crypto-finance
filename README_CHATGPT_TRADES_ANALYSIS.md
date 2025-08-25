# ChatGPT Trades Analysis

This repository contains two programs to analyze trades from `chatgpt_trades.csv` and determine if they were wins or losses based on whether the price action (including wicks) reached the take profit or stop loss levels.

## Programs Overview

### 1. Simple Analyzer (`analyze_chatgpt_trades_simple.py`)
- **Purpose**: Basic analysis without requiring API access
- **Features**: 
  - Loads and analyzes trades from CSV
  - Calculates risk/reward ratios and statistics
  - Provides trade outcome simulations
  - Exports results to CSV
- **Use Case**: When you want quick analysis without historical price data

### 2. Full Analyzer (`analyze_chatgpt_trades.py`)
- **Purpose**: Complete analysis with actual historical price data
- **Features**:
  - Fetches historical candle data for each trade
  - Analyzes wicks to determine if TP/SL was hit
  - Calculates actual trade outcomes
  - Provides detailed analysis and statistics
  - Handles both long and short positions
- **Use Case**: When you want accurate trade outcomes based on real market data

## Requirements

### For Simple Analyzer
- Python 3.7+
- pandas
- numpy

### For Full Analyzer
- Python 3.7+
- pandas
- numpy
- Coinbase API credentials (configured in `config.py`)

## Installation

1. Clone or download the repository
2. Install required packages:
```bash
pip install pandas numpy
```

3. For the full analyzer, ensure your Coinbase API credentials are configured in `config.py`:
```python
API_KEY_PERPS = "your_api_key"
API_SECRET_PERPS = "your_api_secret"
```

## Usage

### Simple Analyzer

Run the simple analyzer to get basic statistics and simulations:

```bash
python analyze_chatgpt_trades_simple.py
```

**Output**:
- Console summary with trade characteristics, risk metrics, and simulations
- CSV file: `chatgpt_trades_simple_analysis.csv`

### Full Analyzer

Run the full analyzer to get actual trade outcomes based on historical data:

```bash
python analyze_chatgpt_trades.py
```

**Output**:
- Console summary with actual trade outcomes
- CSV file: `chatgpt_trades_analysis_results.csv`
- Log file: `chatgpt_trades_analysis.log`

## CSV File Format

The programs expect a CSV file named `chatgpt_trades.csv` with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| timestamp | Trade entry time | 2025-08-25T15:58:09.362797+00:00 |
| strategy | Trading strategy name | Range-Fade-Buy |
| symbol | Trading pair | BTC-PERP-INTX |
| side | Trade direction | BUY/SELL |
| entry_price | Entry price | 112160.0 |
| stop_loss | Stop loss price | 111960 |
| take_profit | Take profit price | 112501 |
| position_size_usd | Position size in USD | 5000 |
| margin | Margin used | 250 |
| leverage | Leverage multiplier | 20 |

## Analysis Features

### Simple Analyzer Features

1. **Trade Characteristics**:
   - Total trades, position sizes, leverage distribution
   - Risk/reward ratios and statistics

2. **Risk Metrics**:
   - Average risk and reward percentages
   - Leveraged risk and reward calculations

3. **Simulations**:
   - Trade outcome simulations with different win rates
   - Profit/loss scenarios

### Full Analyzer Features

1. **Historical Data Analysis**:
   - Fetches 5-minute candles for granular analysis
   - Analyzes wicks to determine TP/SL hits
   - 24-hour analysis window for each trade

2. **Trade Outcomes**:
   - WIN: Take profit was hit
   - LOSS: Stop loss was hit
   - OPEN: Neither TP nor SL was hit within 24 hours
   - ERROR: Data issues or API problems

3. **Performance Metrics**:
   - Win rate and profit factor
   - Maximum favorable and adverse excursions
   - Dollar and percentage P&L

4. **Risk Analysis**:
   - Maximum drawdown periods
   - Risk-adjusted returns
   - Symbol and strategy breakdowns

## Output Files

### Simple Analyzer Output
- `chatgpt_trades_simple_analysis.csv`: Enhanced CSV with calculated metrics
- `chatgpt_trades_simple_analysis.log`: Log file

### Full Analyzer Output
- `chatgpt_trades_analysis_results.csv`: Trade outcomes and analysis
- `chatgpt_trades_analysis.log`: Detailed log file

## Example Results

### Simple Analyzer Output
```
TRADE CHARACTERISTICS:
  Total Trades: 1
  Average Position Size: $5000.0
  Risk/Reward Ratio: 1.7

POTENTIAL P&L (if all trades hit TP/SL):
  Total Potential Profit: $15.2
  Total Potential Loss: $8.92
```

### Full Analyzer Output
```
TRADE COUNT:
  Total Trades: 1
  Winning Trades: 1
  Losing Trades: 0

PERFORMANCE:
  Win Rate: 100.0%
  Total P&L: $15.2 (6.08%)
  Exit Reason: TAKE_PROFIT_HIT
```

## Key Concepts

### Wick Analysis
The full analyzer examines the high and low of each candle (wick) to determine if:
- **For LONG positions**: High wick reaches take profit OR low wick reaches stop loss
- **For SHORT positions**: Low wick reaches take profit OR high wick reaches stop loss

### Trade Exit Logic
1. **Take Profit Hit**: Price wick reaches or exceeds take profit level
2. **Stop Loss Hit**: Price wick reaches or falls below stop loss level
3. **Open Trade**: Neither level is reached within 24 hours (uses last close price)

### Risk Metrics
- **Maximum Favorable Excursion (MFE)**: Best unrealized profit during trade
- **Maximum Adverse Excursion (MAE)**: Worst unrealized loss during trade
- **Profit Factor**: Total profits / Total losses

## Troubleshooting

### Common Issues

1. **API Errors**: Ensure Coinbase API credentials are correct
2. **No Historical Data**: Check if the symbol exists and has data
3. **CSV Format**: Verify the CSV file has all required columns
4. **Date Range**: Ensure trade timestamps are within available data range

### Error Messages

- `'dict' object has no attribute 'start'`: Historical data format issue (fixed)
- `No candles returned`: Symbol not found or no data available
- `Missing required price data`: Invalid entry, TP, or SL values

## Contributing

To improve the analysis:

1. Add more granular timeframes (1-minute candles)
2. Implement additional exit conditions
3. Add more risk metrics (Sharpe ratio, drawdown analysis)
4. Support for more exchanges and data sources

## License

This project is for educational and analysis purposes. Please ensure compliance with Coinbase API terms of service when using the full analyzer.
