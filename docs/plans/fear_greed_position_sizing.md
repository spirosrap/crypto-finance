# Fear & Greed Index Position Sizing Module

## Implementation Plan for Crypto-Finance Trading System

**Date:** February 8, 2026  
**Status:** Planned Enhancement  
**Priority:** Medium-High  
**Exchange:** Coinbase Perps (exchange-agnostic design)

---

## Overview

Integrate the Crypto Fear & Greed Index as a position sizing modifier to enhance trade conviction through sentiment alignment. This module applies dynamic multipliers to base position sizes based on market sentiment, increasing size when sentiment aligns with trade direction and decreasing when opposed.

**Key Principle:** Fear & Greed affects position SIZE, not signal generation. Entry signals continue to come from existing regime detection (BTC vs 20 EMA).

---

## Core Logic

### Sentiment Zones & Multipliers

| Zone | Index Range | Long Multiplier | Short Multiplier | Interpretation |
|------|-------------|-----------------|------------------|----------------|
| Extreme Fear | 0-20 | 2.0x | 0.25x | Maximum conviction for longs |
| Fear | 21-40 | 1.5x | 0.5x | Increased long sizing |
| Neutral | 41-60 | 1.0x | 1.0x | No modification |
| Greed | 61-80 | 0.5x | 1.5x | Increased short sizing |
| Extreme Greed | 81-100 | 0.25x | 2.0x | Maximum conviction for shorts |

### Position Sizing Formula

```
adjusted_risk = base_risk × regime_tilt × fear_greed_multiplier
final_risk = clamp(adjusted_risk, min_risk, max_risk)
position_size = final_risk / stop_distance
```

Where:
- `base_risk`: $15 (configurable base risk per trade)
- `regime_tilt`: 0.7 (trending side) or 0.3 (counter-trend)
- `fear_greed_multiplier`: From table above based on alignment
- `min_risk`: $3.75 (0.25x base)
- `max_risk`: $30 (2x base)

---

## Python Implementation

### FearGreedSizier Class

```python
import requests
from datetime import datetime, timedelta
from typing import Dict, Literal
import logging

logger = logging.getLogger(__name__)


class FearGreedSizier:
    """
    Position sizing modifier based on Crypto Fear & Greed Index.
    
    Integrates with regime-based trading systems to apply sentiment-based
    position sizing multipliers. Works with any perpetual futures exchange
    (Coinbase, Hyperliquid, etc.).
    """
    
    API_URL = "https://api.alternative.me/fng/?limit=1"
    CACHE_TTL_HOURS = 12
    
    MULTIPLIERS = {
        'extreme_fear': {'long': 2.0, 'short': 0.25},
        'fear': {'long': 1.5, 'short': 0.5},
        'neutral': {'long': 1.0, 'short': 1.0},
        'greed': {'long': 0.5, 'short': 1.5},
        'extreme_greed': {'long': 0.25, 'short': 2.0}
    }
    
    ZONES = [
        (20, 'extreme_fear'),
        (40, 'fear'),
        (60, 'neutral'),
        (80, 'greed'),
        (100, 'extreme_greed')
    ]
    
    def __init__(self, 
                 max_multiplier: float = 2.0,
                 min_multiplier: float = 0.25,
                 timeout: int = 10):
        """
        Initialize Fear & Greed sizer.
        
        Args:
            max_multiplier: Hard cap on position sizing (default: 2.0x)
            min_multiplier: Hard floor on position sizing (default: 0.25x)
            timeout: API request timeout in seconds
        """
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        self.timeout = timeout
        self._cache_value: int = 50  # Start neutral
        self._cache_time: datetime = datetime.min
    
    def _get_sentiment_zone(self, index: int) -> str:
        """Map Fear & Greed index (0-100) to sentiment zone."""
        for threshold, zone in self.ZONES:
            if index <= threshold:
                return zone
        return 'extreme_greed'
    
    def fetch_index(self) -> int:
        """
        Fetch current Fear & Greed Index from alternative.me API.
        
        Returns:
            Fear & Greed Index value (0-100)
            
        Falls back to cached value if API fails or cache is fresh.
        """
        # Check cache freshness
        if datetime.now() - self._cache_time < timedelta(hours=self.CACHE_TTL_HOURS):
            logger.debug(f"Using cached F&G index: {self._cache_value}")
            return self._cache_value
        
        try:
            response = requests.get(
                self.API_URL, 
                timeout=self.timeout,
                headers={'Accept': 'application/json'}
            )
            response.raise_for_status()
            
            data = response.json()
            index = int(data['data'][0]['value'])
            classification = data['data'][0]['value_classification']
            
            # Update cache
            self._cache_value = index
            self._cache_time = datetime.now()
            
            logger.info(f"Fetched F&G Index: {index} ({classification})")
            return index
            
        except requests.RequestException as e:
            logger.warning(f"F&G API request failed: {e}. Using cached value: {self._cache_value}")
            return self._cache_value
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"F&G API response parsing failed: {e}. Using cached value: {self._cache_value}")
            return self._cache_value
    
    def get_size_multiplier(self, side: Literal['long', 'short']) -> float:
        """
        Get position sizing multiplier for given trade side.
        
        Args:
            side: 'long' or 'short'
            
        Returns:
            Multiplier between min_multiplier and max_multiplier
        """
        index = self.fetch_index()
        zone = self._get_sentiment_zone(index)
        multiplier = self.MULTIPLIERS[zone][side]
        
        # Apply hard caps
        clamped = max(self.min_multiplier, min(self.max_multiplier, multiplier))
        
        logger.debug(f"F&G multiplier for {side}: {clamped} (zone: {zone}, raw: {multiplier})")
        return clamped
    
    def calculate_position_size(self,
                                base_risk: float,
                                regime_tilt: float,
                                side: Literal['long', 'short'],
                                stop_distance: float) -> float:
        """
        Calculate final position size with Fear & Greed adjustment.
        
        Args:
            base_risk: Base dollar risk per trade (e.g., $15)
            regime_tilt: Regime allocation (0.3 or 0.7)
            side: 'long' or 'short'
            stop_distance: Stop loss distance as decimal (e.g., 0.02 for 2%)
            
        Returns:
            Position size in base currency units
        """
        fng_multiplier = self.get_size_multiplier(side)
        
        # Apply multipliers
        adjusted_risk = base_risk * regime_tilt * fng_multiplier
        
        # Apply hard risk limits
        min_risk = base_risk * self.min_multiplier
        max_risk = base_risk * self.max_multiplier
        final_risk = max(min_risk, min(max_risk, adjusted_risk))
        
        # Convert to position size
        position_size = final_risk / stop_distance
        
        logger.info(
            f"Position sizing: base=${base_risk}, tilt={regime_tilt}, "
            f"fng={fng_multiplier:.2f}x, final_risk=${final_risk:.2f}, "
            f"size={position_size:.4f}"
        )
        
        return position_size
```

---

## Integration with Existing Trading System

### Example Usage

```python
from fear_greed_sizier import FearGreedSizier

# Initialize (once per trading session)
fng_sizer = FearGreedSizier(
    max_multiplier=2.0,
    min_multiplier=0.25
)

# In your trade execution logic:
def calculate_trade_size(signal, base_risk=15.0):
    """
    Calculate position size with regime tilt and Fear & Greed adjustment.
    """
    # Determine regime tilt
    btc_price = get_btc_price()
    ema20 = get_btc_ema20()
    
    if signal['side'] == 'long':
        regime_tilt = 0.7 if btc_price > ema20 else 0.3
    else:  # short
        regime_tilt = 0.7 if btc_price < ema20 else 0.3
    
    # Get stop distance from your existing logic
    stop_distance = signal['stop_distance']  # e.g., 0.02 for 2%
    
    # Calculate position size with F&G adjustment
    position_size = fng_sizer.calculate_position_size(
        base_risk=base_risk,
        regime_tilt=regime_tilt,
        side=signal['side'],
        stop_distance=stop_distance
    )
    
    return position_size
```

### Integration Point

Insert the Fear & Greed calculation **after** regime tilt application but **before** final position sizing:

```
Signal Generation → Regime Tilt → Fear & Greed Multiplier → Position Sizing → Order Execution
         ↑                                                        ↑
    (BTC vs EMA20)                                         (Stop distance → size)
```

---

## Risk Management Considerations

### 1. Drawdown Impact

**Concern:** Fear & Greed can increase position sizes up to 2x, potentially amplifying drawdowns.

**Mitigation:**
- Hard caps at 2x base risk regardless of sentiment
- Correlation check: Extreme sentiment often coincides with high volatility — consider reducing base_risk during VIX spikes
- Portfolio heat limit: Ensure total portfolio risk doesn't exceed max acceptable drawdown

### 2. Correlation with Regime Tilt

**Scenario:** Both tilt and F&G can size up simultaneously
- Trend long + Extreme Fear = 0.7 × 2.0 = 1.4x base (not 2.0x)
- Counter-trend long + Extreme Fear = 0.3 × 2.0 = 0.6x base

**Mitigation:** The multiplicative approach naturally dampens extreme sizing through regime tilt allocation.

### 3. Whipsaw Protection

**Concern:** Fresh signals during volatile sentiment shifts can lead to false entries.

**Mitigation:**
```python
# Only apply F&G sizing if signal age > 24 hours
if signal_age_hours > 24:
    fng_multiplier = fng_sizer.get_size_multiplier(side)
else:
    fng_multiplier = 1.0  # Neutral for fresh signals
```

### 4. API Failure Handling

**Current:** Falls back to cached value, then neutral (50).

**Monitoring:** Log API failures; if >3 failures in 24h, disable F&G sizing and alert.

---

## Configuration Parameters

```yaml
# config/fear_greed.yaml

fear_greed:
  enabled: true
  
  # Multipliers for each zone
  multipliers:
    extreme_fear: {long: 2.0, short: 0.25}
    fear: {long: 1.5, short: 0.5}
    neutral: {long: 1.0, short: 1.0}
    greed: {long: 0.5, short: 1.5}
    extreme_greed: {long: 0.25, short: 2.0}
  
  # Hard limits
  max_multiplier: 2.0
  min_multiplier: 0.25
  
  # API settings
  api_url: "https://api.alternative.me/fng/?limit=1"
  cache_ttl_hours: 12
  timeout_seconds: 10
  
  # Safety settings
  require_signal_age_hours: 0  # Set to 24 to enable whipsaw protection
  disable_on_api_failures: 3   # Disable after N failures in 24h
```

---

## Testing Approach

### Phase 1: Unit Tests

```python
def test_sentiment_zone_mapping():
    sizer = FearGreedSizier()
    assert sizer._get_sentiment_zone(10) == 'extreme_fear'
    assert sizer._get_sentiment_zone(35) == 'fear'
    assert sizer._get_sentiment_zone(50) == 'neutral'
    assert sizer._get_sentiment_zone(75) == 'greed'
    assert sizer._get_sentiment_zone(95) == 'extreme_greed'

def test_multiplier_clamping():
    sizer = FearGreedSizier(max_multiplier=1.5)
    # Even with extreme fear, should cap at 1.5
    assert sizer.get_size_multiplier('long') <= 1.5

def test_api_fallback():
    sizer = FearGreedSizier()
    # Simulate API failure by using invalid URL
    sizer.API_URL = "https://invalid.url"
    result = sizer.fetch_index()
    assert result == 50  # Should fallback to neutral
```

### Phase 2: Backtesting

1. **Download historical F&G data:**
   - alternative.me provides historical data
   - Or use: `https://api.alternative.me/fng/?limit=1000` for last 1000 days

2. **Simulate on past trades:**
   ```python
   for trade in historical_trades:
       fng_at_time = get_historical_fng(trade['date'])
       original_size = trade['position_size']
       adjusted_size = apply_fng_sizing(original_size, fng_at_time, trade['side'])
       compare_performance(original_size, adjusted_size, trade['pnl'])
   ```

3. **Metrics to track:**
   - Win rate change
   - Average winner/loser size
   - Max drawdown
   - Sharpe ratio
   - Expectancy

### Phase 3: Paper Trading

**Week 1-2:** Log F&G sizing but don't apply
- Compare hypothetical vs actual sizes
- Verify API reliability
- Check for any edge case bugs

**Week 3-4:** Apply with 0.5x max multiplier
- Reduced risk while validating live behavior
- Monitor for unexpected sizing

**Week 5+:** Full deployment
- Enable full 2.0x max multiplier
- Daily review of sizing decisions

---

## Potential Pitfalls & Mitigations

| Pitfall | Impact | Mitigation |
|---------|--------|------------|
| F&G API goes down | Falls back to cached/neutral | 12h cache + neutral fallback + monitoring alerts |
| Extreme sentiment whipsaws | Oversized losing trades | Require signal age > 24h or reduce base_risk in extreme zones |
| Multiplier stacking | 2x F&G × 2x tilt = 4x risk | Regime tilt is 0.3/0.7, not 2x, so max is 0.7×2.0=1.4x |
| Sentiment lag | F&G updates daily, market moves hourly | Accept as feature — daily sentiment is the signal, not intraday |
| Overfitting multipliers | Optimize for past data that won't repeat | Use conservative multipliers, backtest on multiple market cycles |
| Exchange-specific issues | Coinbase vs Hyperliquid sizing differences | Keep sizer exchange-agnostic; exchange logic in separate layer |

---

## Future Enhancements

1. **Multi-timeframe F&G:** Weight 7-day average more than daily
2. **Asset-specific sentiment:** BTC F&G for BTC pairs, ETH F&G for ETH pairs
3. **Combined with funding rates:** Size up more when funding favors your direction
4. **Volatility adjustment:** Reduce base_risk when VIX > threshold
5. **On-chain sentiment:** Integrate with Nansen/Arkham for on-chain fear/greed signals

---

## References

- **Fear & Greed API:** https://alternative.me/crypto/fear-and-greed-index/
- **API Documentation:** https://api.alternative.me/
- **Original DCA Strategy:** https://x.com/bsmokes/status/2020557128738701822

---

## Implementation Checklist

- [ ] Create `fear_greed_sizier.py` module
- [ ] Add unit tests
- [ ] Create configuration YAML
- [ ] Integrate with existing position sizing logic
- [ ] Download historical F&G data for backtesting
- [ ] Run backtest on 2022-2025 data
- [ ] Paper trade for 2 weeks
- [ ] Deploy to live with monitoring
- [ ] Document in TRADING_PROGRESSION.md

---

## Backtest Results (February 8, 2026)

**Status:** ✅ COMPLETED — Results advise AGAINST implementation with current multipliers

### Test Configuration
- **Dataset:** 485 trades (watchdog + paper_finder logs)
- **Fear & Greed Data:** 1000 days from API
- **Test Period:** December 2025 - February 2026
- **Trade Attribution:** `opened_at` date used for F&G lookup

### Overall Performance Comparison

| Metric | Baseline | F&G Adjusted | Change |
|--------|----------|--------------|--------|
| **Total PnL** | -$8.06 | -$111.96 | **-1,289%** 📉 |
| **Win Rate** | 48.45% | 48.45% | 0% |
| **Avg Winner** | $4.14 | $3.50 | -15.4% |
| **Avg Loser** | -$4.44 | -$4.23 | +4.6% (improved) |
| **Expectancy** | -$0.28 | -$0.48 | -71.6% |
| **Profit Factor** | 0.99 | 0.88 | -11.2% |
| **Max Drawdown** | $195.72 | $222.52 | +13.7% |
| **Sharpe Ratio** | -0.07 | -0.95 | **-1,196%** |

### Per-Zone Performance (F&G Adjusted)

| Zone | Side | Trades | Total PnL | Win Rate | Sharpe |
|------|------|--------|-----------|----------|--------|
| **Extreme Fear** | Long | 25 | **-$147.58** | 20.0% | -3.69 |
| **Extreme Fear** | Short | 67 | -$2.26 | 47.8% | -0.23 |
| **Fear** | Long | 110 | -$25.56 | 41.8% | -0.30 |
| **Fear** | Short | 166 | **+$74.90** | 60.2% | **+2.55** |
| **Neutral** | Long | 54 | **-$114.91** | 27.8% | -3.31 |
| **Neutral** | Short | 42 | **+$58.77** | 64.3% | **+1.90** |
| **Greed** | Long | 12 | -$23.72 | 8.3% | -3.80 |
| **Greed** | Short | 9 | **+$68.40** | 100.0% | **+8.20** |
| **Extreme Greed** | Long | 0 | N/A | N/A | N/A |
| **Extreme Greed** | Short | 0 | N/A | N/A | N/A |

### Key Findings

1. **The Multipliers Amplified Losses**
   - The baseline strategy had **negative expectancy** (-$0.28 per trade)
   - Applying multipliers >1x to losing trades amplified losses significantly:
     - Extreme Fear Longs: Sized up 2.0x → Lost $147.58 (worst segment)
     - Fear Longs: Sized up 1.5x → Lost $25.56
     - Neutral Longs: Kept at 1.0x → Still lost $114.91 (baseline underperformance)

2. **Shorts Outperformed Across All Zones**
   - Interestingly, **shorts were profitable in Fear, Neutral, and Greed zones**:
     - Fear Shorts (0.5x): +$74.90 with 2.55 Sharpe
     - Neutral Shorts (1.0x): +$58.77 with 1.90 Sharpe  
     - Greed Shorts (1.5x): +$68.40 with 8.20 Sharpe (100% win rate!)
   - This suggests the underlying strategy has a **short bias** that the F&G sizing inadvertently reduced

3. **Win Rate Unaffected**
   - As expected, win rates remained identical (48.45%) because multipliers scale position size, not outcome probability
   - The deterioration came from larger losses on losing longs and smaller wins on winning longs

4. **Drawdown Increased**
   - Max drawdown worsened from $195.72 to $222.52 (+13.7%), primarily driven by oversized losing longs in Extreme Fear zone

### Root Cause Analysis

**The fundamental issue:** Fear & Greed multipliers were designed to increase long exposure during fear (2.0x multiplier), but the strategy **loses money on longs**:
- Longs total: -$196.97
- Shorts total: +$188.91

The multipliers amplified the strategy's biggest weakness rather than its strength.

### Alternative Approaches to Consider

Based on these results, consider these modifications if re-testing:

1. **Inverse Multipliers (Counter-Trend Sizing)**
   - Since shorts work better, consider sizing UP shorts in fear zones
   - Or use F&G to size DOWN the losing side (longs) rather than up

2. **Volatility-Based Sizing**
   - Size down when F&G indicates high uncertainty (extreme zones)
   - Rather than sentiment alignment, use it as a risk control

3. **Regime-Conditional Application**
   - Only apply F&G sizing when baseline Sharpe > 0 in that zone
   - Disable multipliers in zones where the strategy historically loses

4. **Short-Only Enhancement**
   - Given the strong short performance, consider using F&G primarily to enhance short sizing
   - Reduce or neutralize long multipliers

### Recommendation

**VERDICT: DO NOT IMPLEMENT** ❌ (with current multipliers)

The Fear & Greed position sizing significantly degraded performance because the baseline strategy loses money on longs, but F&G multipliers were designed to increase long exposure during fear zones.

**For future testing:**
- [ ] Invert multipliers (test counter-trend sizing)
- [ ] Apply only to shorts (which are profitable)
- [ ] Use as risk control (size down in extreme zones) rather than opportunity sizing
- [ ] Fix long profitability first, then re-test sentiment alignment

### Files Generated

- **Backtest Script:** `/home/spiros/clawd/fear_greed_backtest.py`
- **Results JSON:** `backtest_results.json`
- **F&G Cache:** `fear_greed_history.json`

---

*Document created: February 8, 2026*  
*Author: Rook (OpenClaw Agent)*  
*Status: TESTED — Results advise against implementation with current configuration*
