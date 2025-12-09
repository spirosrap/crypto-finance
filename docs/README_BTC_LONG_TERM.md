# Bitcoin Long-Term Buy & Hold Strategy

A comprehensive Bitcoin investment strategy focused on long-term accumulation (6-12 months+) with Dollar Cost Averaging (DCA) capabilities.

## 🎯 Strategy Overview

This strategy implements a **buy and hold forever** approach for Bitcoin:

- **Initial Investment**: Lump sum purchase to start the portfolio
- **Dollar Cost Averaging (DCA)**: Regular purchases at set intervals
- **No Selling**: Pure accumulation strategy - no exit strategy
- **Portfolio Tracking**: Comprehensive performance monitoring
- **Coinbase Integration**: Uses Coinbase spot trading (not perpetuals)

## 🚀 Key Features

### Investment Strategy
- **Initial Investment**: Configurable lump sum to start (default: $1,000)
- **DCA Schedule**: Weekly purchases (configurable frequency and amount)
- **Monthly Limits**: Prevents over-investment (default: max 4 DCA purchases/month)
- **Manual Buys**: Additional purchases when you feel like it

### Portfolio Management
- **Real-time Tracking**: Current portfolio value and performance
- **Trade History**: Complete record of all purchases
- **Performance Metrics**: Total return, percentage gains, average cost basis
- **Persistent Storage**: All data saved to JSON files

### Risk Management
- **No Leverage**: Uses spot trading only (no perpetuals)
- **DCA Limits**: Prevents emotional over-buying
- **Portfolio Monitoring**: Clear visibility into investment performance

## 📁 Files Created

- `btc_long_term_strategy.py` - Main strategy implementation
- `btc_portfolio.json` - Portfolio statistics (auto-created)
- `btc_trades.json` - Complete trade history (auto-created)
- `btc_long_term_strategy.log` - Detailed logging (auto-created)

## 🛠️ Usage

### Basic Usage

```bash
# Run with default settings (initial $1000, weekly $100 DCA)
python btc_long_term_strategy.py

# Show portfolio summary only
python btc_long_term_strategy.py --summary

# Manual buy of $500
python btc_long_term_strategy.py --manual-buy 500
```

### Advanced Configuration

```bash
# Custom initial investment and DCA settings
python btc_long_term_strategy.py \
    --initial 5000 \
    --dca-amount 200 \
    --dca-frequency 14

# Disable DCA (manual buys only)
python btc_long_term_strategy.py --disable-dca

# Weekly $50 DCA
python btc_long_term_strategy.py --dca-amount 50 --dca-frequency 7
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--initial` | Initial investment amount in USD | $1,000 |
| `--dca-amount` | DCA amount in USD | $100 |
| `--dca-frequency` | DCA frequency in days | 7 (weekly) |
| `--disable-dca` | Disable DCA functionality | False |
| `--summary` | Show portfolio summary only | False |
| `--manual-buy` | Execute manual buy with USD amount | None |

## 📊 Portfolio Summary Example

```
============================================================
🚀 BITCOIN LONG-TERM PORTFOLIO SUMMARY
============================================================
📅 Date: 2024-01-15 14:30:25
💰 Total Invested: $2,100.00
🪙 BTC Owned: 0.045123
💵 Current Value: $2,450.00
📈 Total Return: $350.00
📊 Return %: +16.67%
📉 Average Price: $46,550.00
📈 Current Price: $54,300.00
🔄 DCA Count: 11
📅 Last DCA: 2024-01-14T10:15:30
============================================================

📋 RECENT TRADES (Last 10):
--------------------------------------------------------------------------------
Date                 Type            BTC         USD         Price/BTC    
--------------------------------------------------------------------------------
2024-01-14 10:15    DCA_BUY         0.001234    100.00      81,000.00    
2024-01-07 10:15    DCA_BUY         0.001456    100.00      68,700.00    
2024-01-01 10:15    DCA_BUY         0.001789    100.00      55,900.00    
```

## 🔧 Setup Requirements

1. **API Credentials**: Ensure your Coinbase API credentials are in `config.py`
2. **Dependencies**: Install required packages from `requirements.txt`
3. **Coinbase Account**: Active Coinbase account with USD funding

## 📈 Strategy Benefits

### Long-Term Focus
- **Time in Market**: Emphasizes holding through market cycles
- **Emotional Control**: Automated DCA reduces emotional decision-making
- **Compounding**: Benefits from Bitcoin's long-term growth potential

### Risk Mitigation
- **DCA Effect**: Reduces impact of buying at market peaks
- **No Timing**: Eliminates need to predict market movements
- **Diversification**: Can be part of broader investment strategy

### Tax Efficiency
- **Long-term Holdings**: Potential for favorable capital gains treatment
- **No Frequent Trading**: Reduces tax complexity
- **Cost Basis Tracking**: Maintains detailed purchase records

## 🎯 Investment Philosophy

This strategy follows the **"HODL"** philosophy:

1. **Buy**: Regular accumulation through DCA
2. **Hold**: Never sell, regardless of market conditions
3. **Accumulate**: Focus on increasing Bitcoin holdings over time
4. **Ignore Noise**: Don't react to short-term price movements

## ⚠️ Important Notes

- **No Exit Strategy**: This is a pure accumulation strategy
- **Market Risk**: Bitcoin prices can be highly volatile
- **Long-term Horizon**: Designed for 6-12+ month holding periods
- **Not Financial Advice**: Use at your own risk and discretion

## 🔄 Automation

For automated execution, consider setting up a cron job:

```bash
# Weekly DCA execution (every Sunday at 10 AM)
0 10 * * 0 cd /path/to/crypto-finance && python btc_long_term_strategy.py

# Daily portfolio check (every day at 9 AM)
0 9 * * * cd /path/to/crypto-finance && python btc_long_term_strategy.py --summary
```

## 📞 Support

For questions or issues:
1. Check the log file: `btc_long_term_strategy.log`
2. Review trade history: `btc_trades.json`
3. Verify portfolio data: `btc_portfolio.json`

---

**Remember**: This is a long-term strategy. Focus on accumulation, not short-term gains. The goal is to build a substantial Bitcoin position over time through consistent, disciplined investing.
