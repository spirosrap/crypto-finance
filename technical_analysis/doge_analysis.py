from .base import BaseTechnicalAnalysis, SignalType, SignalResult, MarketRegime, TechnicalIndicatorConfig
from typing import Dict, Optional
import pandas as pd
import numpy as np
import talib

class DogeAnalysis(BaseTechnicalAnalysis):
    def __init__(self, config: Optional[TechnicalIndicatorConfig] = None):
        super().__init__(config)
        self.required_history = 100

    def analyze(self, candles: list[Dict]) -> Optional[SignalResult]:
        """
        Enhanced DOGE analysis with less restrictive conditions.
        """
        if len(candles) < self.required_history:
            return None

        if not self.validate_data(candles):
            self.logger.error("Invalid candle data")
            return None

        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(candles)
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)

            # Calculate basic indicators
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            macd, macd_signal, macd_hist = self.calculate_macd(candles)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(candles)
            
            # Calculate EMAs
            ema_9 = talib.EMA(df['close'], timeperiod=9)
            ema_21 = talib.EMA(df['close'], timeperiod=21)
            
            # Volume analysis
            volume_sma = df['volume'].rolling(window=20).mean()
            relative_volume = df['volume'] / volume_sma
            
            # Get current values
            current_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            current_price = current_candle['close']
            current_rsi = df['rsi'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2]
            
            # Price momentum
            price_change = (current_price - prev_candle['close']) / prev_candle['close']
            
            # Market regime determination
            market_regime = self.identify_market_regime(candles)
            
            # Less restrictive trading conditions
            long_conditions = (
                # RSI conditions
                (current_rsi < 45 or  # Moderately oversold
                (current_rsi < 50 and current_rsi > prev_rsi)) and  # Or RSI turning up below midpoint
                
                # Trend conditions (need only one)
                (macd > macd_signal or  # MACD bullish
                ema_9.iloc[-1] > ema_9.iloc[-2] or  # Short-term uptrend
                price_change > 0.005) and  # Positive momentum
                
                # Volume confirmation
                relative_volume.iloc[-1] > 1.2 and  # Above average volume
                
                # Market regime check
                market_regime in [MarketRegime.BULLISH, MarketRegime.NEUTRAL, MarketRegime.VOLATILE]
            )
            
            short_conditions = (
                # RSI conditions
                (current_rsi > 55 or  # Moderately overbought
                (current_rsi > 50 and current_rsi < prev_rsi)) and  # Or RSI turning down above midpoint
                
                # Trend conditions (need only one)
                (macd < macd_signal or  # MACD bearish
                ema_9.iloc[-1] < ema_9.iloc[-2] or  # Short-term downtrend
                price_change < -0.005) and  # Negative momentum
                
                # Volume confirmation
                relative_volume.iloc[-1] > 1.2 and  # Above average volume
                
                # Market regime check
                market_regime in [MarketRegime.BEARISH, MarketRegime.NEUTRAL, MarketRegime.VOLATILE]
            )
            
            if long_conditions:
                # Calculate confidence based on signal strength
                rsi_factor = (45 - current_rsi) / 45 if current_rsi < 45 else 0.5
                volume_factor = min((relative_volume.iloc[-1] - 1) * 0.5, 1)
                trend_factor = 0.5 + (0.5 * (1 if macd > macd_signal else 0))
                
                confidence = min(0.9, (rsi_factor + volume_factor + trend_factor) / 3)
                
                return SignalResult(
                    signal_type=SignalType.LONG,
                    confidence=confidence,
                    indicators={
                        'rsi': current_rsi,
                        'macd': macd,
                        'macd_signal': macd_signal,
                        'volume_ratio': relative_volume.iloc[-1],
                        'price_momentum': price_change
                    },
                    market_regime=market_regime,
                    timestamp=float(current_candle['timestamp'])
                )
                
            elif short_conditions:
                # Calculate confidence based on signal strength
                rsi_factor = (current_rsi - 55) / 45 if current_rsi > 55 else 0.5
                volume_factor = min((relative_volume.iloc[-1] - 1) * 0.5, 1)
                trend_factor = 0.5 + (0.5 * (1 if macd < macd_signal else 0))
                
                confidence = min(0.9, (rsi_factor + volume_factor + trend_factor) / 3)
                
                return SignalResult(
                    signal_type=SignalType.SHORT,
                    confidence=confidence,
                    indicators={
                        'rsi': current_rsi,
                        'macd': macd,
                        'macd_signal': macd_signal,
                        'volume_ratio': relative_volume.iloc[-1],
                        'price_momentum': price_change
                    },
                    market_regime=market_regime,
                    timestamp=float(current_candle['timestamp'])
                )
            
            return None

        except Exception as e:
            self.logger.error(f"Error in analysis: {str(e)}")
            return None

    def _calculate_volume_metrics(self, df: pd.DataFrame, period: int = 20):
        """Calculate volume-based metrics."""
        df['volume_sma'] = df['volume'].rolling(window=period).mean()