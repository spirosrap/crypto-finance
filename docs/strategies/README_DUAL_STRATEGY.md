# BTC Dual Strategy Alert Monitor

This system implements two complementary Bitcoin trading strategies based on the provided trading plans:

## 📊 Strategy Overview

### Plan A: Breakout Long (Momentum)
- **Trigger**: 1-hour close ≥ $119,650 (conservative) or ≥ $120,150 (aggressive)
- **Entry**: Buy-stop order at trigger price
- **Stop Loss**: $118,800 (below breakout bar)
- **Take Profit 1**: $120,900
- **Take Profit 2**: $122,500-$123,500
- **Volume Condition**: ≥ 1.25x 20-period SMA
- **RSI Filter**: ≤ 70 (not overbought)

### Plan B: Pullback Long (Fade to Support)
- **Bid Zone**: $117,350-$117,600 (wick + quick reclaim only)
- **Entry**: After price reclaims the zone
- **Stop Loss**: $116,600
- **Take Profit**: $118,900-$119,500 (trail if strength persists)
- **Conditions**: 
  - Zone reclaimed within 1-2 candles
  - No fill on heavy sell volume (≤2x average)

## 🚀 Features

- **Dual Strategy Monitoring**: Simultaneously monitors both breakout and pullback opportunities
- **Volume Confirmation**: Ensures trades are backed by sufficient volume
- **RSI Filtering**: Prevents entry into overbought conditions
- **State Management**: Tracks which strategies have been triggered to avoid duplicate trades
- **Alert System**: Plays sound alerts when trade conditions are met
- **Robust Error Handling**: Retry logic with exponential backoff for API calls
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

## 📁 Files

- `crypto_alert_monitor.py` - Main monitoring script
- `test_dual_strategy.py` - Test script to verify strategy logic
- `btc_dual_strategy_trigger_state.json` - State file (created automatically)
- `btc_dual_strategy_alert_debug.log` - Log file (created automatically)

## 🛠️ Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create Alert Sound**:
   ```bash
   python synthesize_alert_sound.py
   ```

3. **Configure API Credentials**:
   Ensure your `config.py` file contains valid Coinbase API credentials:
   ```python
   API_KEY_PERPS = "your_api_key"
   API_SECRET_PERPS = "your_api_secret"
   ```

## 🎯 Usage

### Start Monitoring
```bash
python crypto_alert_monitor.py
```

### Test Strategy Logic
```bash
python test_dual_strategy.py
```

## 📈 How It Works

### Plan A (Breakout) Logic
1. Monitors 1-hour candles for price action
2. Checks if close price exceeds trigger levels ($119,650 or $120,150)
3. Confirms volume is ≥1.25x 20-period average
4. Ensures RSI ≤ 70 (not overbought)
5. Executes buy-stop order at trigger price
6. Sets stop loss at $118,800 and take profit targets

### Plan B (Pullback) Logic
1. Monitors for price wicks into bid zone ($117,350-$117,600)
2. Waits for quick reclaim (close above $117,600)
3. Confirms no heavy sell volume (≤2x average)
4. Enters long position at current market price
5. Sets stop loss at $116,600 and take profit targets

## 🔧 Configuration

### Trade Parameters
- **Margin**: $250 USD
- **Leverage**: 20x
- **Product**: BTC-PERP-INTX
- **Timeframe**: 1-hour candles

### Risk Management
- Each strategy can only trigger once per session
- State is persisted to prevent duplicate trades
- Stop losses are automatically set
- Take profit targets are predefined

## 📊 Monitoring

The system provides comprehensive logging including:
- Current market conditions
- Strategy analysis for both plans
- Trade execution details
- Error handling and retry attempts
- Performance metrics

## ⚠️ Risk Disclaimer

This is an automated trading system that executes real trades with real money. Please:
- Test thoroughly with small amounts first
- Monitor the system while it's running
- Understand the risks involved in leveraged trading
- Ensure your API credentials are secure
- Have proper risk management in place

## 🔄 State Management

The system maintains state in `btc_dual_strategy_trigger_state.json`:
```json
{
  "plan_a_triggered": false,
  "plan_b_triggered": false,
  "last_trigger_ts": null
}
```

This prevents duplicate trades and allows for session management.

## 🐛 Troubleshooting

### Common Issues
1. **API Connection Errors**: The system will retry with exponential backoff
2. **Insufficient Data**: Requires at least 25 candles for analysis
3. **Sound Alert Issues**: Ensure `alert_sound.wav` exists
4. **Trade Execution Failures**: Check API permissions and account balance

### Log Files
- Check `btc_dual_strategy_alert_debug.log` for detailed error information
- Monitor console output for real-time status updates

## 📝 Example Output

```
🚀 BTC/USD Dual Strategy Alert

📊 Plan A: Breakout Long (Momentum)
   • Trigger: 1h close ≥ $119,650 (conservative) or ≥ $120,150 (aggressive)
   • Entry: Buy-stop at $119,650 or $120,150
   • Stop Loss: $118,800 (below breakout bar)
   • TP1: $120,900
   • TP2: $122,500-$123,500
   • Volume Condition: ≥ 1.25x 20-period SMA

📊 Plan B: Pullback Long (Fade to Support)
   • Bid Zone: $117,350-$117,600 (wick + quick reclaim only)
   • Stop Loss: $116,600
   • Take Profit: $118,900-$119,500 (trail if strength persists)
   • Conditions: Zone reclaimed within 1-2 candles, no fill on heavy sell volume

Current 1-Hour Candle: close=$119,800, high=$120,100, low=$119,500
Volume: 1,500, Avg20: 1,200, Rel_Vol: 1.25
RSI: 65.2

🔍 Plan A (Breakout) Analysis:
   • Price trigger (≥$119,650 or ≥$120,150): ✅
   • Volume ≥ 1.25x avg: ✅
   • RSI ≤ 70: ✅
   • Not already triggered: ✅
   • Plan A Ready: 🎯 YES
```

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system. 