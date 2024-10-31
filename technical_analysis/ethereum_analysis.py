from .base import BaseTechnicalAnalysis, SignalType, SignalResult, MarketRegime
from typing import List, Dict
import time
import numpy as np

class EthereumAnalysis(BaseTechnicalAnalysis):
    """Technical analysis specifically tuned for Ethereum trading."""
    
    def analyze(self, candles: List[Dict]) -> SignalResult:
        if not self.validate_data(candles):
            raise ValueError("Invalid candle data")
        
        # Calculate indicators
        rsi = self.calculate_rsi(candles)
        macd, signal, hist = self.calculate_macd(candles)
        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(candles)
        volatility = self.calculate_volatility(candles)
        market_regime = self.identify_market_regime(candles)
        
        # Current price and recent prices
        current_price = float(candles[-1]['close'])
        recent_prices = [float(candle['close']) for candle in candles[-10:]]
        short_term_trend = np.mean(recent_prices) < current_price
        
        # Initialize signal strength components
        signal_components = []
        
        # RSI Signal (30% weight)
        if rsi < 30:
            signal_components.append(0.3)  # Strong buy
        elif rsi < 40:
            signal_components.append(0.15)  # Moderate buy
        elif rsi > 70:
            signal_components.append(-0.3)  # Strong sell
        elif rsi > 60:
            signal_components.append(-0.15)  # Moderate sell
        else:
            signal_components.append(0)
            
        # MACD Signal (30% weight)
        if hist > 0 and macd > signal:
            signal_components.append(0.3)  # Strong buy
        elif hist > 0:
            signal_components.append(0.15)  # Moderate buy
        elif hist < 0 and macd < signal:
            signal_components.append(-0.3)  # Strong sell
        elif hist < 0:
            signal_components.append(-0.15)  # Moderate sell
        else:
            signal_components.append(0)
            
        # Bollinger Bands Signal (20% weight)
        bb_position = (current_price - lower_bb) / (upper_bb - lower_bb)
        if bb_position < 0.2:
            signal_components.append(0.2)  # Strong buy
        elif bb_position > 0.8:
            signal_components.append(-0.2)  # Strong sell
        else:
            signal_components.append(0)
            
        # Trend Following (20% weight)
        if short_term_trend:
            signal_components.append(0.2)
        else:
            signal_components.append(-0.2)
            
        # Calculate final signal strength (-1 to 1)
        signal_strength = sum(signal_components)
        
        # Adjust signal based on market regime
        if market_regime == MarketRegime.HIGH_VOLATILITY:
            signal_strength *= 0.5  # Reduce signal strength in high volatility
        elif market_regime == MarketRegime.TRENDING_UP:
            signal_strength *= 1.2  # Amplify signals in trending markets
        elif market_regime == MarketRegime.TRENDING_DOWN:
            signal_strength *= 1.2
            
        # Determine signal type and confidence
        signal_type = self._determine_signal_type(signal_strength)
        confidence = min(abs(signal_strength), 1.0)
        
        # Print debug information
        print(f"\nSignal Analysis:")
        print(f"RSI: {rsi:.2f}")
        print(f"MACD: {macd:.2f}, Signal: {signal:.2f}, Hist: {hist:.2f}")
        print(f"BB Position: {bb_position:.2f}")
        print(f"Market Regime: {market_regime}")
        print(f"Signal Strength: {signal_strength:.2f}")
        print(f"Signal Type: {signal_type}")
        print(f"Confidence: {confidence:.2f}")
        
        # Add debug information
        print(f"\nAnalyzing candle at timestamp: {candles[-1]['timestamp']}")
        print(f"Price: ${current_price:.2f}")
        print(f"Signal Components:")
        print(f"RSI Signal: {signal_components[0]:.2f}")
        print(f"MACD Signal: {signal_components[1]:.2f}")
        print(f"BB Signal: {signal_components[2]:.2f}")
        print(f"Trend Signal: {signal_components[3]:.2f}")
        print(f"Final Signal Strength (after adjustments): {signal_strength:.2f}")
        print(f"Market Regime Adjustment: {1.2 if market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN] else 0.5 if market_regime == MarketRegime.HIGH_VOLATILITY else 1.0}")
        
        return SignalResult(
            signal_type=signal_type,
            confidence=confidence,
            indicators={
                'rsi': rsi,
                'macd': macd,
                'macd_signal': signal,
                'macd_hist': hist,
                'bb_upper': upper_bb,
                'bb_middle': middle_bb,
                'bb_lower': lower_bb,
                'volatility': volatility,
                'current_price': current_price
            },
            market_regime=market_regime,
            timestamp=time.time()
        )
    
    def _determine_signal_type(self, signal_strength: float) -> SignalType:
        """Convert signal strength to SignalType."""
        if signal_strength >= 0.5:
            return SignalType.STRONG_BUY
        elif signal_strength >= 0.2:
            return SignalType.BUY
        elif signal_strength <= -0.5:
            return SignalType.STRONG_SELL
        elif signal_strength <= -0.2:
            return SignalType.SELL
        else:
            return SignalType.HOLD