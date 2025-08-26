# ETH Intraday Trading Strategy - Aug 26, 2025

## Overview
This document describes the ETH intraday trading strategy implemented in `crypto_alert_monitor_eth.py` for August 26, 2025.

## Market Context
- **Current Price**: ~$4,430
- **24h Range**: $4,316–$4,706
- **Pivot Levels**: P 4,426.7 / R1 4,449.7 / S1 4,411.3 / S2 4,388
- **Funding**: Mixed, slightly positive on several venues

## Trading Strategies

### LONG Setups

#### 1. Breakout Long
- **Trigger**: M5 close > R1 4,449.7 with rising volume
- **Entry Zone**: 4,450–4,460
- **Entry Price**: 4,455 (mid of zone)
- **Stop Loss**: < 4,430
- **Targets**: 
  - TP1: 4,475
  - TP2: 4,505
  - TP3: 4,565
- **Volume Requirement**: ≥1.5× 5-min volume SMA
- **Notes**: Confirm ≥1.5× 5-min vol SMA before entry

#### 2. Support Long
- **Trigger**: Hold S1 4,411–4,415 with higher low
- **Entry Zone**: 4,412–4,420
- **Entry Price**: 4,416 (mid of zone)
- **Stop Loss**: < 4,400
- **Targets**:
  - TP1: 4,427 (Pivot)
  - TP2: 4,450 (R1)
- **Volume Requirement**: ≥1.25× 5-min volume SMA
- **Notes**: Skip if funding flips negative and OI spikes

### SHORT Setups

#### 3. Rejection Short
- **Trigger**: Wick/reject at 4,450–4,465
- **Entry Zone**: 4,445–4,460
- **Entry Price**: 4,452.5 (mid of zone)
- **Stop Loss**: > 4,470
- **Targets**:
  - TP1: 4,427 (Pivot)
  - TP2: 4,411 (S1)
  - TP3: 4,388 (S2)
- **Volume Requirement**: ≥1.5× 5-min volume SMA
- **Notes**: Favor if funding tilts + and spot lags perp

#### 4. Momentum Short
- **Trigger**: M5 close < 24h low 4,316
- **Entry Zone**: 4,305–4,315
- **Entry Price**: 4,310 (mid of zone)
- **Stop Loss**: > 4,335
- **Targets**:
  - TP1: 4,260
  - TP2: 4,210
- **Volume Requirement**: ≥2.0× 5-min volume SMA (strong volume expansion)
- **Notes**: Only on strong volume expansion

## Position Sizing
- **Margin**: $250 USD
- **Leverage**: 20x
- **Position Size**: $5,000 USD (250 × 20)
- **Risk Management**: ≤1% per trade, R/R ≥ 2:1, hard stops

## Execution Rules
- Use M5 structure and volume confirmation
- Pivots from prior session guide S/R
- Confirm volume requirements before entry
- Use hard stops for risk management

## Technical Implementation
- **Primary Timeframe**: 5-minute candles for triggers
- **Secondary Timeframe**: 15-minute candles for analysis
- **Volume Period**: 20-period SMA for volume confirmation
- **State Tracking**: JSON files for each strategy to prevent duplicate triggers
- **Logging**: All trades logged to `chatgpt_trades.csv`

## Usage
```bash
# Monitor both LONG and SHORT strategies (default)
python crypto_alert_monitor_eth.py

# Monitor only LONG strategies
python crypto_alert_monitor_eth.py --direction LONG

# Monitor only SHORT strategies
python crypto_alert_monitor_eth.py --direction SHORT

# Test CSV logging functionality
python crypto_alert_monitor_eth.py --test-csv
```

## Files
- **Main Script**: `crypto_alert_monitor_eth.py`
- **Trade Log**: `chatgpt_trades.csv`
- **Alert Log**: `eth_alert_debug.log`
- **State Files**: 
  - `eth_breakout_trigger_state.json`
  - `eth_support_trigger_state.json`
  - `eth_rejection_trigger_state.json`
  - `eth_momentum_trigger_state.json`

## Key Changes from Previous Version
- Updated price levels to Aug 26, 2025 market context
- Changed from ATH-focused strategy to intraday pivot-based strategy
- Implemented 4 distinct setups with specific volume requirements
- Enhanced volume confirmation with different thresholds per strategy
- Improved logging and state management
- Fixed position sizing to exactly $5,000 USD (250 × 20 leverage)
