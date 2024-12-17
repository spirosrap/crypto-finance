import logging
from typing import Dict, List
from datetime import datetime, timedelta, UTC
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis, SignalType, TechnicalAnalysisConfig
from config import API_KEY, API_SECRET
from historicaldata import HistoricalData
from coinbase.rest import RESTClient
import argparse
import time
import numpy as np
from enum import Enum

# Add valid choices for granularity and products
VALID_GRANULARITIES = [
    "ONE_MINUTE",
    "FIVE_MINUTE",
    "FIFTEEN_MINUTE",
    "THIRTY_MINUTE",
    "ONE_HOUR",
    "TWO_HOUR",
    "SIX_HOUR",
    "ONE_DAY"
]

VALID_PRODUCTS = [
    "BTC-USDC",
    "ETH-USDC",
    "SOL-USDC",
    "DOGE-USDC",
    "XRP-USDC",
    "ADA-USDC",
    "MATIC-USDC",
    "LINK-USDC",
    "DOT-USDC",
    "UNI-USDC"
]

class PatternType(Enum):
    DOUBLE_TOP = "Double Top"
    DOUBLE_BOTTOM = "Double Bottom"
    HEAD_SHOULDERS = "Head and Shoulders"
    INV_HEAD_SHOULDERS = "Inverse Head and Shoulders"
    TRIANGLE = "Triangle"
    WEDGE = "Wedge"
    NONE = "None"

class MarketRegime(Enum):
    TRENDING = "Trending"
    RANGING = "Ranging"
    VOLATILE = "Volatile"
    BREAKOUT = "Breakout"
    REVERSAL = "Reversal"

class MarketAnalyzer:
    """
    A class that analyzes market conditions and generates trading signals
    using technical analysis indicators.
    """

    def __init__(self, product_id: str = 'DOGE-USDC', candle_interval: str = 'ONE_HOUR'):
        self.product_id = product_id
        self.candle_interval = candle_interval
        self.coinbase_service = CoinbaseService(API_KEY, API_SECRET)
        self.client = RESTClient(API_KEY, API_SECRET)
        self.historical_data = HistoricalData(self.client)
        
        # Initialize custom technical analysis configuration
        self.ta_config = TechnicalAnalysisConfig(
            rsi_overbought=70,
            rsi_oversold=30,
            volatility_threshold=0.02,
            risk_per_trade=0.02,
            atr_multiplier=2.5
        )
        
        self.technical_analysis = TechnicalAnalysis(
            self.coinbase_service,
            config=self.ta_config,
            candle_interval=candle_interval,
            product_id=product_id
        )
        
        self._current_candles = []
        self.logger = logging.getLogger(__name__)
        
        # Add new attributes
        self.pattern_memory = []
        self.sentiment_scores = []
        self.volatility_history = []
        self.max_pattern_memory = 10
        
        # Initialize risk parameters
        self.base_risk = 0.02  # 2% base risk
        self.max_risk = 0.04   # 4% maximum risk
        self.min_risk = 0.01   # 1% minimum risk

    def get_market_signal(self) -> Dict:
        """
        Analyze the market and generate a trading signal based on the last month of data.
        
        Returns:
            Dict containing the signal analysis results
        """
        try:
            # Get candle data for the last month
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(days=30)
            
            # Get historical data using HistoricalData class
            candles = self.historical_data.get_historical_data(
                self.product_id,
                start_time,
                end_time,
                self.candle_interval
            )

            if not candles:
                raise ValueError("No candle data available")

            # Format candles to match expected structure
            formatted_candles = self._format_candles(candles)
            
            # Store the formatted candles
            self._current_candles = formatted_candles

            # Get comprehensive market analysis
            analysis = self.technical_analysis.analyze_market(formatted_candles)
            
            # Get additional trend information
            adx_value, trend_direction = self.technical_analysis.get_trend_strength(formatted_candles)
            
            # Calculate key indicators
            rsi = self.technical_analysis.compute_rsi(self.product_id, formatted_candles)
            macd, signal, histogram = self.technical_analysis.compute_macd(self.product_id, formatted_candles)
            upper_band, middle_band, lower_band = self.technical_analysis.compute_bollinger_bands(formatted_candles)
            current_price = float(formatted_candles[-1]['close'])
            
            # Add signal stability information
            signal_stability = "High" if len(self.technical_analysis.signal_history) >= self.technical_analysis.trend_confirmation_period and \
                                  all(s['strength'] > 0 for s in self.technical_analysis.signal_history) or \
                                  all(s['strength'] < 0 for s in self.technical_analysis.signal_history) else "Low"
            
            # Get volume confirmation analysis
            volume_info = self.technical_analysis.analyze_volume_confirmation(formatted_candles)
            
            # Create detailed analysis result
            result = {
                'timestamp': datetime.now(UTC).isoformat(),
                'product_id': self.product_id,
                'current_price': current_price,
                'signal': analysis['signal'].signal_type.value,
                'position': 'LONG' if analysis['signal'].signal_type in [SignalType.STRONG_BUY, SignalType.BUY] 
                           else 'SHORT' if analysis['signal'].signal_type in [SignalType.STRONG_SELL, SignalType.SELL]
                           else 'NEUTRAL',
                'confidence': analysis['signal'].confidence,
                'market_condition': analysis['signal'].market_condition.value,
                'risk_metrics': analysis['risk_metrics'],
                'indicators': {
                    'rsi': round(rsi, 2),
                    'macd': round(macd, 2),
                    'macd_signal': round(signal, 2),
                    'macd_histogram': round(histogram, 2),
                    'bollinger_upper': round(upper_band, 2),
                    'bollinger_middle': round(middle_band, 2),
                    'bollinger_lower': round(lower_band, 2),
                    'adx': round(adx_value, 2),
                    'trend_direction': trend_direction
                },
                'recommendation': self._generate_recommendation(analysis['signal'].signal_type),
                'signal_stability': signal_stability,
                'signals_analyzed': len(self.technical_analysis.signal_history),
                'time_since_last_change': time.time() - (self.technical_analysis.last_signal_time or time.time()),
                'volume_analysis': {
                    'change': round(volume_info['volume_change'], 1),
                    'trend': volume_info['volume_trend'],
                    'strength': volume_info['strength'],
                    'is_confirming': volume_info['is_confirming'],
                    'price_change': round(volume_info['price_change'], 1)
                }
            }

            # Add pattern recognition
            patterns = self.detect_chart_patterns(self._current_candles)
            result['patterns'] = {
                'type': patterns['type'].value,
                'confidence': patterns['confidence'],
                'target': patterns['target'],
                'stop_loss': patterns['stop_loss']
            }
            
            # Add dynamic risk calculation
            dynamic_risk = self.calculate_dynamic_risk(self._current_candles)
            result['risk_metrics']['dynamic_risk'] = dynamic_risk
            
            # Add pattern history
            result['pattern_history'] = [
                {
                    'timestamp': p['timestamp'].isoformat(),
                    'pattern': p['pattern']['type'].value,
                    'confidence': p['pattern']['confidence']
                }
                for p in self.pattern_memory[-5:]  # Last 5 patterns
            ]
            
            # Calculate probability of success
            probability = self.calculate_success_probability(
                result['indicators'],
                result['volume_analysis'],
                result['patterns']
            )
            
            result['probability_analysis'] = probability
            
            # Add momentum analysis
            momentum_analysis = self.calculate_momentum_score(formatted_candles)
            result['momentum_analysis'] = momentum_analysis
            
            # Add regime analysis
            regime_analysis = self.detect_market_regime(formatted_candles)
            result['regime_analysis'] = {
                'regime': regime_analysis['regime'].value,
                'confidence': regime_analysis['confidence'],
                'metrics': regime_analysis['metrics']
            }
            
            return result

        except Exception as e:
            self.logger.error(f"Error generating market signal: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now(UTC).isoformat(),
                'product_id': self.product_id,
                'signal': 'HOLD',
                'position': 'NEUTRAL',
                'confidence': 0.0,
                'current_price': 0.0,
                'recommendation': 'Unable to generate signal due to error'
            }

    def _format_candles(self, candles: List[Dict]) -> List[Dict]:
        """Format candles to match the expected structure."""
        formatted_candles = []
        for candle in candles:
            formatted_candle = {
                'time': candle['start'],
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'volume': float(candle['volume'])
            }
            formatted_candles.append(formatted_candle)
        return formatted_candles

    def _generate_recommendation(self, signal_type: SignalType) -> str:
        """Generate a detailed trading recommendation including consolidation patterns and bias."""
        try:
            if not self._current_candles:
                return "No market data available for recommendation"
            
            # Get current price and indicators
            current_price = float(self._current_candles[-1]['close'])
            atr = self.technical_analysis.compute_atr(self._current_candles)
            consolidation_info = self.technical_analysis.detect_consolidation(self._current_candles)
            volume_info = self.technical_analysis.analyze_volume_confirmation(self._current_candles)
            
            # Calculate market bias
            rsi = self.technical_analysis.compute_rsi(self.product_id, self._current_candles)
            macd, signal, histogram = self.technical_analysis.compute_macd(self.product_id, self._current_candles)
            adx_value, trend_direction = self.technical_analysis.get_trend_strength(self._current_candles)
            
            # Calculate price change
            prices = self.technical_analysis.extract_prices(self._current_candles)
            price_change = ((prices[-1] - prices[-2]) / prices[-2]) * 100 if len(prices) > 1 else 0.0
            
            # Calculate key levels
            stop_loss_atr = atr * self.ta_config.atr_multiplier
            take_profit_1r = stop_loss_atr * 1.5  # 1.5:1 reward-risk
            take_profit_2r = stop_loss_atr * 2.0  # 2:1 reward-risk
            take_profit_3r = stop_loss_atr * 3.0  # 3:1 reward-risk
            
            # Calculate support and resistance levels
            resistance_level = consolidation_info['upper_channel']
            support_level = consolidation_info['lower_channel']
            
            # Determine bias based on multiple indicators
            bias_factors = []
            
            # RSI bias
            if rsi > 50:
                bias_factors.append(("Bullish", (rsi - 50) / 50))
            else:
                bias_factors.append(("Bearish", (50 - rsi) / 50))
            
            # MACD bias
            if macd > signal:
                bias_factors.append(("Bullish", abs(histogram / macd) if macd != 0 else 0.1))
            else:
                bias_factors.append(("Bearish", abs(histogram / macd) if macd != 0 else 0.1))
            
            # Trend direction bias
            if trend_direction == "Uptrend":
                bias_factors.append(("Bullish", adx_value / 100))
            elif trend_direction == "Downtrend":
                bias_factors.append(("Bearish", adx_value / 100))
            
            # Volume trend bias
            if volume_info['volume_trend'] == "Increasing":
                if price_change > 0:
                    bias_factors.append(("Bullish", 0.3))
                else:
                    bias_factors.append(("Bearish", 0.3))
                
            # Calculate overall bias
            bullish_strength = sum(strength for direction, strength in bias_factors if direction == "Bullish")
            bearish_strength = sum(strength for direction, strength in bias_factors if direction == "Bearish")
            
            # Determine dominant bias
            if bullish_strength > bearish_strength:
                bias = f"Neutral with Bullish Bias (Strength: {(bullish_strength - bearish_strength):.1f})"
            elif bearish_strength > bullish_strength:
                bias = f"Neutral with Bearish Bias (Strength: {(bearish_strength - bullish_strength):.1f})"
            else:
                bias = "Neutral with No Clear Bias"
            
            # Define recommendations dictionary
            recommendations = {
                SignalType.STRONG_BUY: {
                    'position': 'LONG',
                    'message': f"Strong buy signal detected. Consider opening a LONG position:\n"
                              f"• Entry Price: ${current_price:.4f}\n"
                              f"• Position Type: LONG\n"
                              f"• Leverage: 1-2x maximum\n\n"
                              f"Support Level: ${support_level:.4f}\n"
                              f"Resistance Level: ${resistance_level:.4f}\n\n"
                              f"Stop Losses:\n"
                              f"• Conservative: ${(current_price - stop_loss_atr):.4f} (-{(stop_loss_atr/current_price)*100:.1f}%)\n"
                              f"• Aggressive: ${(current_price - (stop_loss_atr*0.7)):.4f} (-{(stop_loss_atr*0.7/current_price)*100:.1f}%)\n\n"
                              f"Take Profit Targets:\n"
                              f"• Target 1 (1.5R): ${(current_price + take_profit_1r):.4f} (+{(take_profit_1r/current_price)*100:.1f}%)\n"
                              f"• Target 2 (2R): ${(current_price + take_profit_2r):.4f} (+{(take_profit_2r/current_price)*100:.1f}%)\n"
                              f"• Target 3 (3R): ${(current_price + take_profit_3r):.4f} (+{(take_profit_3r/current_price)*100:.1f}%)"
                },
                SignalType.STRONG_SELL: {
                    'position': 'SHORT',
                    'message': f"Strong sell signal detected. Consider opening a SHORT position:\n"
                              f"• Entry Price: ${current_price:.4f}\n"
                              f"• Position Type: SHORT\n"
                              f"• Leverage: 1-2x maximum\n\n"
                              f"Support Level: ${support_level:.4f}\n"
                              f"Resistance Level: ${resistance_level:.4f}\n\n"
                              f"Stop Losses:\n"
                              f"• Conservative: ${(current_price + stop_loss_atr):.4f} (+{(stop_loss_atr/current_price)*100:.1f}%)\n"
                              f"• Aggressive: ${(current_price + (stop_loss_atr*0.7)):.4f} (+{(stop_loss_atr*0.7/current_price)*100:.1f}%)\n\n"
                              f"Take Profit Targets:\n"
                              f"• Target 1 (1.5R): ${(current_price - take_profit_1r):.4f} (-{(take_profit_1r/current_price)*100:.1f}%)\n"
                              f"• Target 2 (2R): ${(current_price - take_profit_2r):.4f} (-{(take_profit_2r/current_price)*100:.1f}%)\n"
                              f"• Target 3 (3R): ${(current_price - take_profit_3r):.4f} (-{(take_profit_3r/current_price)*100:.1f}%)"
                },
                SignalType.BUY: {
                    'position': 'LONG',
                    'message': f"Bullish conditions detected. Consider a conservative LONG position:\n"
                              f"• Entry Price: ${current_price:.4f}\n"
                              f"• Position Type: LONG\n"
                              f"• Leverage: 1x only\n\n"
                              f"Support Level: ${support_level:.4f}\n"
                              f"Resistance Level: ${resistance_level:.4f}\n\n"
                              f"Stop Loss: Place below support at ${(support_level - (atr * 0.5)):.4f}\n"
                              f"Take Profit Targets:\n"
                              f"• Target 1 (1.5R): ${(current_price + take_profit_1r):.4f}\n"
                              f"• Target 2 (2R): ${(current_price + take_profit_2r):.4f}"
                },
                SignalType.SELL: {
                    'position': 'SHORT',
                    'message': f"Bearish conditions detected. Consider a conservative SHORT position:\n"
                              f"• Entry Price: ${current_price:.4f}\n"
                              f"• Position Type: SHORT\n"
                              f"• Leverage: 1x only\n\n"
                              f"Support Level: ${support_level:.4f}\n"
                              f"Resistance Level: ${resistance_level:.4f}\n\n"
                              f"Stop Loss: Place above resistance at ${(resistance_level + (atr * 0.5)):.4f}\n"
                              f"Take Profit Targets:\n"
                              f"• Target 1 (1.5R): ${(current_price - take_profit_1r):.4f}\n"
                              f"• Target 2 (2R): ${(current_price - take_profit_2r):.4f}"
                },
                SignalType.HOLD: {
                    'position': 'NEUTRAL',
                    'message': f"Market conditions are neutral:\n"
                              f"• Current Price: ${current_price:.4f}\n"
                              f"• ATR: ${atr:.4f}\n\n"
                              f"Market Bias:\n"
                              f"• Current Bias: {bias}\n"
                              f"• RSI Position: {'Bullish' if rsi > 50 else 'Bearish'} ({rsi:.1f})\n"
                              f"• MACD Status: {'Bullish' if macd > signal else 'Bearish'}\n"
                              f"• Trend Direction: {trend_direction}\n"
                              f"• Volume Trend: {volume_info['volume_trend']}\n\n"
                              f"Key Levels:\n"
                              f"• Support: ${support_level:.4f}\n"
                              f"• Resistance: ${resistance_level:.4f}\n\n"
                              f"Recommendation:\n"
                              f"• Hold existing positions\n"
                              f"• Wait for price to break ${resistance_level:.4f} resistance or\n"
                              f"  ${support_level:.4f} support with volume confirmation\n"
                              f"• Prepare for potential {'bullish' if bullish_strength > bearish_strength else 'bearish'} move"
                }
            }
            
            # Add volume analysis message
            volume_message = f"\nVolume Analysis:\n" \
                            f"• Volume Change: {volume_info['volume_change']:.1f}%\n" \
                            f"• Volume Trend: {volume_info['volume_trend']}\n" \
                            f"• Signal Strength: {volume_info['strength']}\n" \
                            f"• Price Change: {volume_info['price_change']:.1f}%\n" \
                            f"• Volume Confirmation: {'Yes' if volume_info['is_confirming'] else 'No'}"
            
            # Add volume message to each recommendation
            for key in recommendations:
                recommendations[key]['message'] = recommendations[key]['message'] + volume_message
            
            rec = recommendations.get(signal_type, {
                'position': 'NEUTRAL',
                'message': "No clear trading opportunity. Wait for better setup." + volume_message
            })
            
            # Add consolidation information if relevant
            consolidation_message = ""
            if consolidation_info['is_consolidating']:
                if consolidation_info['pattern'] == "Breakout":
                    target = consolidation_info['upper_channel'] + (consolidation_info['upper_channel'] - consolidation_info['lower_channel'])
                    consolidation_message = f"\nBreakout Pattern Detected:\n" \
                                          f"• Breakout Level: ${consolidation_info['upper_channel']:.4f}\n" \
                                          f"• Volume Confirmation: {'Yes' if consolidation_info['volume_confirmed'] else 'No'}\n" \
                                          f"• Suggested Stop: ${consolidation_info['channel_middle']:.4f}\n" \
                                          f"• Target: ${target:.4f}"
                elif consolidation_info['pattern'] == "Breakdown":
                    target = consolidation_info['lower_channel'] - (consolidation_info['upper_channel'] - consolidation_info['lower_channel'])
                    consolidation_message = f"\nBreakdown Pattern Detected:\n" \
                                          f"• Breakdown Level: ${consolidation_info['lower_channel']:.4f}\n" \
                                          f"• Volume Confirmation: {'Yes' if consolidation_info['volume_confirmed'] else 'No'}\n" \
                                          f"• Suggested Stop: ${consolidation_info['channel_middle']:.4f}\n" \
                                          f"• Target: ${target:.4f}"
                else:
                    consolidation_message = f"\nConsolidation Phase Detected:\n" \
                                          f"• Upper Channel: ${consolidation_info['upper_channel']:.4f}\n" \
                                          f"• Lower Channel: ${consolidation_info['lower_channel']:.4f}\n" \
                                          f"• Channel Middle: ${consolidation_info['channel_middle']:.4f}\n" \
                                          f"• Strength: {consolidation_info['strength']*100:.1f}%"
            
            # Add rejection analysis if detected
            if consolidation_info.get('rejection_event'):
                rejection = consolidation_info['rejection_event']
                rejection_time = datetime.fromtimestamp(float(rejection['timestamp'])).strftime('%Y-%m-%d %H:%M:%S')
                
                confirmation_strength = "Strong" if rejection['confirming_candles'] >= 2 and rejection['volume_confirmation'] else \
                                      "Moderate" if rejection['confirming_candles'] >= 1 or rejection['volume_confirmation'] else \
                                      "Weak"
                
                if rejection['type'] == 'resistance':
                    consolidation_message += f"\nMost Recent Resistance Rejection:\n" \
                                           f"• Time: {rejection_time}\n" \
                                           f"• Rejection Level: ${rejection['price_level']:.4f}\n" \
                                           f"• Current Price: ${rejection['price']:.4f}\n" \
                                           f"• Distance from Level: {rejection['distance_from_level']:.1f}%\n" \
                                           f"• Rejection Volume: {rejection['volume']:.2f}\n" \
                                           f"• Volume vs Average: {rejection['volume_ratio']:.1f}x\n" \
                                           f"��� Confirming Candles: {rejection['confirming_candles']}\n" \
                                           f"• Volume Confirmation: {'Yes' if rejection['volume_confirmation'] else 'No'}\n" \
                                           f"• Confirmation Strength: {confirmation_strength}"
                else:  # support rejection
                    consolidation_message += f"\nMost Recent Support Bounce:\n" \
                                           f"• Time: {rejection_time}\n" \
                                           f"• Support Level: ${rejection['price_level']:.4f}\n" \
                                           f"• Current Price: ${rejection['price']:.4f}\n" \
                                           f"• Distance from Level: {rejection['distance_from_level']:.1f}%\n" \
                                           f"• Bounce Volume: {rejection['volume']:.2f}\n" \
                                           f"• Volume vs Average: {rejection['volume_ratio']:.1f}x\n" \
                                           f"• Confirming Candles: {rejection['confirming_candles']}\n" \
                                           f"• Volume Confirmation: {'Yes' if rejection['volume_confirmation'] else 'No'}\n" \
                                           f"• Confirmation Strength: {confirmation_strength}"
            
            return f"Position: {rec['position']}\n\n{rec['message']}{consolidation_message}"
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {str(e)}")
            return "Error generating recommendation"

    def _determine_signal_type(self, signal_strength: float) -> SignalType:
        """
        Determine signal type based on signal strength.
        More sensitive thresholds for more frequent signals.
        
        Args:
            signal_strength: Float value between -10 and 10
            
        Returns:
            SignalType: The determined signal type
        """
        if signal_strength >= 4:  # Reduced from 7
            return SignalType.STRONG_BUY
        elif signal_strength >= 1.5:  # Reduced from 3
            return SignalType.BUY
        elif signal_strength <= -4:  # Changed from -7
            return SignalType.STRONG_SELL
        elif signal_strength <= -1.5:  # Changed from -3
            return SignalType.SELL
        else:
            # Check if there's a slight bias even in the "neutral" zone
            if signal_strength > 0.5:  # Slight bullish bias
                return SignalType.BUY
            elif signal_strength < -0.5:  # Slight bearish bias
                return SignalType.SELL
            return SignalType.HOLD

    def _calculate_base_signal(self, indicators: Dict[str, float], market_condition: str) -> float:
        """Calculate the base signal without stability checks."""
        try:
            weights = self.set_product_weights()
            signal_strength = 0.0
            
            # RSI Signal (-1 to 1 scale)
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if rsi > self.config.rsi_overbought:
                    signal_strength -= weights['rsi']
                elif rsi < self.config.rsi_oversold:
                    signal_strength += weights['rsi']
                else:
                    # More sensitive to RSI movements around the middle
                    signal_strength += weights['rsi'] * ((rsi - 50) / 20)  # More sensitive

            # MACD Signal
            if all(k in indicators for k in ['macd', 'macd_signal', 'macd_histogram']):
                macd = indicators['macd']
                macd_signal = indicators['macd_signal']
                histogram = indicators['macd_histogram']
                
                # More sensitive MACD signals
                if macd > macd_signal:
                    signal_strength += weights['macd'] * (1 + abs(histogram/macd) * 0.5)
                else:
                    signal_strength -= weights['macd'] * (1 + abs(histogram/macd) * 0.5)

            # Bollinger Bands Signal
            if 'bollinger' in indicators:
                signal_strength += weights['bollinger'] * indicators['bollinger'] * 1.5  # Increased sensitivity

            # ADX Signal - More sensitive to trend strength
            if 'adx' in indicators:
                adx = indicators['adx']
                if adx > 15:  # Lowered from 20
                    if indicators.get('trend_direction') == "Uptrend":
                        signal_strength += weights['adx'] * (adx/40)  # More sensitive
                    elif indicators.get('trend_direction') == "Downtrend":
                        signal_strength -= weights['adx'] * (adx/40)

            # MA Crossover Signal
            if 'ma_crossover' in indicators:
                signal_strength += weights['ma_crossover'] * indicators['ma_crossover'] * 1.25

            # Market Condition Adjustment
            condition_multipliers = {
                'Bull Market': 1.3,
                'Bear Market': 0.7,
                'Bullish': 1.2,
                'Bearish': 0.8,
                'Neutral': 1.0
            }
            
            # Apply market condition multiplier
            multiplier = condition_multipliers.get(market_condition, 1.0)
            signal_strength *= multiplier

            # Add volume confirmation boost
            if 'volume_trend' in indicators and indicators['volume_trend'] == "Increasing":
                signal_strength *= 1.2

            # Normalize signal strength to be between -10 and 10
            signal_strength = max(min(signal_strength * 1.5, 10), -10)  # Increased sensitivity

            self.logger.debug(f"Base signal strength calculated: {signal_strength}")
            return signal_strength

        except Exception as e:
            self.logger.error(f"Error calculating base signal: {str(e)}")
            return 0.0

    def detect_chart_patterns(self, candles: List[Dict]) -> Dict:
        """Detect common chart patterns in the price data."""
        try:
            prices = np.array([c['close'] for c in candles])
            highs = np.array([c['high'] for c in candles])
            lows = np.array([c['low'] for c in candles])
            
            patterns = {
                'type': PatternType.NONE,
                'confidence': 0.0,
                'target': None,
                'stop_loss': None
            }
            
            # Double Top Detection
            if self._is_double_top(highs[-30:]):
                patterns['type'] = PatternType.DOUBLE_TOP
                patterns['confidence'] = 0.8
                patterns['target'] = min(lows[-30:])
                patterns['stop_loss'] = max(highs[-30:]) + (max(highs[-30:]) - min(lows[-30:])) * 0.1
            
            # Double Bottom Detection
            elif self._is_double_bottom(lows[-30:]):
                patterns['type'] = PatternType.DOUBLE_BOTTOM
                patterns['confidence'] = 0.8
                patterns['target'] = max(highs[-30:])
                patterns['stop_loss'] = min(lows[-30:]) - (max(highs[-30:]) - min(lows[-30:])) * 0.1
            
            # Add pattern to memory
            if patterns['type'] != PatternType.NONE:
                self.pattern_memory.append({
                    'timestamp': datetime.now(UTC),
                    'pattern': patterns
                })
                
                # Maintain pattern memory size
                if len(self.pattern_memory) > self.max_pattern_memory:
                    self.pattern_memory.pop(0)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting chart patterns: {str(e)}")
            return {'type': PatternType.NONE, 'confidence': 0.0}

    def calculate_dynamic_risk(self, candles: List[Dict]) -> float:
        """Calculate dynamic risk based on market volatility."""
        try:
            # Calculate current volatility
            returns = np.diff(np.log([c['close'] for c in candles]))
            current_vol = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            # Store volatility
            self.volatility_history.append(current_vol)
            if len(self.volatility_history) > 30:
                self.volatility_history.pop(0)
            
            # Calculate relative volatility
            avg_vol = np.mean(self.volatility_history)
            vol_ratio = current_vol / avg_vol if avg_vol != 0 else 1
            
            # Adjust risk based on volatility
            dynamic_risk = self.base_risk * (1 / vol_ratio)
            
            # Constrain risk within bounds
            return max(min(dynamic_risk, self.max_risk), self.min_risk)
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic risk: {str(e)}")
            return self.base_risk

    def _is_double_top(self, prices: np.ndarray, threshold: float = 0.02) -> bool:
        """Detect double top pattern."""
        try:
            peaks = self._find_peaks(prices)
            if len(peaks) < 2:
                return False
                
            # Get last two peaks
            last_peaks = peaks[-2:]
            peak_prices = prices[last_peaks]
            
            # Check if peaks are within threshold
            price_diff = abs(peak_prices[0] - peak_prices[1]) / peak_prices[0]
            time_diff = last_peaks[1] - last_peaks[0]
            
            return price_diff < threshold and 5 <= time_diff <= 20
            
        except Exception as e:
            self.logger.error(f"Error detecting double top: {str(e)}")
            return False

    def _is_double_bottom(self, prices: np.ndarray, threshold: float = 0.02) -> bool:
        """Detect double bottom pattern."""
        try:
            troughs = self._find_troughs(prices)
            if len(troughs) < 2:
                return False
                
            # Get last two troughs
            last_troughs = troughs[-2:]
            trough_prices = prices[last_troughs]
            
            # Check if troughs are within threshold
            price_diff = abs(trough_prices[0] - trough_prices[1]) / trough_prices[0]
            time_diff = last_troughs[1] - last_troughs[0]
            
            return price_diff < threshold and 5 <= time_diff <= 20
            
        except Exception as e:
            self.logger.error(f"Error detecting double bottom: {str(e)}")
            return False

    def _find_peaks(self, prices: np.ndarray, window: int = 5) -> List[int]:
        """Find peaks in price data."""
        peaks = []
        for i in range(window, len(prices) - window):
            if all(prices[i] > prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] > prices[i+j] for j in range(1, window+1)):
                peaks.append(i)
        return peaks

    def _find_troughs(self, prices: np.ndarray, window: int = 5) -> List[int]:
        """Find troughs in price data."""
        troughs = []
        for i in range(window, len(prices) - window):
            if all(prices[i] < prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] < prices[i+j] for j in range(1, window+1)):
                troughs.append(i)
        return troughs

    def _generate_error_response(self) -> Dict:
        """Generate standardized error response."""
        return {
            'error': 'Analysis error',
            'timestamp': datetime.now(UTC).isoformat(),
            'product_id': self.product_id,
            'signal': 'HOLD',
            'position': 'NEUTRAL',
            'confidence': 0.0,
            'patterns': {'type': PatternType.NONE.value, 'confidence': 0.0},
            'risk_metrics': {'dynamic_risk': self.base_risk}
        }

    def calculate_success_probability(self, indicators: Dict, volume_info: Dict, patterns: Dict) -> Dict:
        """Calculate probability of success for the suggested direction with detailed move analysis."""
        try:
            probability_factors = []
            move_characteristics = {}
            
            # Trend alignment (0-20%)
            trend_direction = indicators.get('trend_direction', 'Unknown')
            adx = indicators.get('adx', 0)
            if trend_direction == "Uptrend":
                probability_factors.append(("Trend", 20 if adx > 25 else 10))
                move_characteristics['trend_quality'] = {
                    'strength': 'Strong' if adx > 25 else 'Moderate' if adx > 15 else 'Weak',
                    'duration': 'Established' if adx > 30 else 'Developing',
                    'momentum': 'Accelerating' if adx > indicators.get('prev_adx', 0) else 'Decelerating'
                }
            elif trend_direction == "Downtrend":
                probability_factors.append(("Trend", 20 if adx > 25 else 10))
                move_characteristics['trend_quality'] = {
                    'strength': 'Strong' if adx > 25 else 'Moderate' if adx > 15 else 'Weak',
                    'duration': 'Established' if adx > 30 else 'Developing',
                    'momentum': 'Accelerating' if adx > indicators.get('prev_adx', 0) else 'Decelerating'
                }
            else:
                probability_factors.append(("Trend", 5))
                move_characteristics['trend_quality'] = {
                    'strength': 'Weak',
                    'duration': 'Undefined',
                    'momentum': 'Neutral'
                }

            # RSI alignment with detailed momentum analysis (0-15%)
            rsi = indicators.get('rsi', 50)
            rsi_momentum = {
                'condition': 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral',
                'strength': abs(rsi - 50) / 50,
                'divergence': 'None'  # Could be calculated with price comparison
            }
            if rsi > 70 or rsi < 30:
                probability_factors.append(("RSI", 15))
            elif 40 <= rsi <= 60:
                probability_factors.append(("RSI", 5))
            else:
                probability_factors.append(("RSI", 10))
            move_characteristics['momentum'] = rsi_momentum

            # MACD confirmation with trend strength (0-15%)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            histogram = indicators.get('macd_histogram', 0)
            
            macd_analysis = {
                'crossover_type': 'Bullish' if macd > macd_signal else 'Bearish' if macd < macd_signal else 'None',
                'histogram_strength': abs(histogram) / abs(macd) if macd != 0 else 0,
                'momentum_quality': 'Increasing' if histogram > 0 and histogram > indicators.get('prev_histogram', 0) else
                                  'Decreasing' if histogram < 0 and histogram < indicators.get('prev_histogram', 0) else 'Neutral'
            }
            
            if macd != 0 or macd_signal != 0:
                macd_diff = abs(macd - macd_signal)
                macd_strength = min(15, (macd_diff / (abs(macd_signal) + 0.00001)) * 15)
                probability_factors.append(("MACD", macd_strength))
            else:
                probability_factors.append(("MACD", 0))
            move_characteristics['macd_analysis'] = macd_analysis

            # Volume confirmation with detailed analysis (0-20%)
            volume_change = volume_info.get('volume_change', 0)
            is_confirming = volume_info.get('is_confirming', False)
            
            volume_quality = {
                'trend': volume_info.get('volume_trend', 'Neutral'),
                'strength': 'Strong' if abs(volume_change) > 50 else 'Moderate' if abs(volume_change) > 20 else 'Weak',
                'consistency': 'High' if is_confirming else 'Low',
                'price_alignment': 'Confirmed' if is_confirming else 'Divergent'
            }
            
            if is_confirming:
                vol_strength = min(20, abs(volume_change) / 5)
                probability_factors.append(("Volume", vol_strength))
            else:
                probability_factors.append(("Volume", 5))
            move_characteristics['volume_quality'] = volume_quality

            # Pattern recognition with failure points (0-15%)
            pattern_type = patterns.get('type', 'None')
            pattern_confidence = patterns.get('confidence', 0)
            
            pattern_analysis = {
                'type': pattern_type,
                'reliability': pattern_confidence,
                'completion': patterns.get('completion_percentage', 0),
                'failure_points': {
                    'immediate': patterns.get('stop_loss', None),
                    'pattern_invalidation': patterns.get('invalidation_level', None)
                }
            }
            
            if pattern_type != "None":
                pattern_strength = min(15, pattern_confidence * 15)
                probability_factors.append(("Pattern", pattern_strength))
            else:
                probability_factors.append(("Pattern", 0))
            move_characteristics['pattern_analysis'] = pattern_analysis

            # Market condition analysis (0-15%)
            market_condition = indicators.get('market_condition', 'Unknown')
            
            market_context = {
                'condition': market_condition,
                'volatility': indicators.get('volatility', 'Normal'),
                'liquidity': 'High' if volume_info.get('volume_change', 0) > 0 else 'Normal',
                'support_resistance_proximity': indicators.get('price_level_proximity', 'Far')
            }
            
            if market_condition in ["Bull Market", "Bear Market"]:
                probability_factors.append(("Market", 15))
            elif market_condition in ["Bullish", "Bearish"]:
                probability_factors.append(("Market", 10))
            else:
                probability_factors.append(("Market", 5))
            move_characteristics['market_context'] = market_context

            # Calculate total probability
            total_probability = sum(factor[1] for factor in probability_factors)
            total_probability = max(0, min(100, total_probability))
            
            # Determine move quality characteristics
            move_quality = {
                'expected_speed': 'Rapid' if total_probability > 80 else 'Moderate' if total_probability > 60 else 'Slow',
                'expected_volatility': 'High' if market_context['volatility'] == 'High' else 'Normal',
                'continuation_probability': f"{total_probability:.1f}%",
                'reversal_risk': 'Low' if total_probability > 75 else 'Moderate' if total_probability > 50 else 'High'
            }

            # Calculate confidence level
            confidence_level = "Very High" if total_probability >= 85 else \
                             "High" if total_probability >= 70 else \
                             "Moderate" if total_probability >= 50 else \
                             "Low" if total_probability >= 30 else "Very Low"

            return {
                'total_probability': total_probability,
                'confidence_level': confidence_level,
                'factors': [(factor[0], round(factor[1], 1)) for factor in probability_factors],
                'move_characteristics': move_characteristics,
                'move_quality': move_quality,
                'failure_points': {
                    'immediate_stop': pattern_analysis['failure_points']['immediate'],
                    'trend_reversal_point': pattern_analysis['failure_points']['pattern_invalidation'],
                    'momentum_failure_level': indicators.get('key_reversal_level', None)
                }
            }

        except Exception as e:
            self.logger.error(f"Error calculating success probability: {str(e)}")
            return {
                'total_probability': 0,
                'confidence_level': "Low",
                'factors': [("Error", 0)],
                'move_characteristics': {},
                'move_quality': {},
                'failure_points': {}
            }

    def calculate_momentum_score(self, candles: List[Dict]) -> Dict:
        """
        Calculate market momentum score using multiple indicators.
        Returns score between -100 (strong bearish) to +100 (strong bullish).
        """
        try:
            # Extract price data
            prices = np.array([c['close'] for c in candles])
            volumes = np.array([c['volume'] for c in candles])
            
            # Calculate momentum indicators
            rsi = self.technical_analysis.compute_rsi(self.product_id, candles)
            macd, signal, histogram = self.technical_analysis.compute_macd(self.product_id, candles)
            adx_value, trend_direction = self.technical_analysis.get_trend_strength(candles)
            
            # Calculate rate of change
            roc = ((prices[-1] - prices[-20]) / prices[-20]) * 100
            
            # Volume momentum
            vol_sma = np.mean(volumes[-20:])
            vol_momentum = ((volumes[-1] - vol_sma) / vol_sma) * 100
            
            # Calculate component scores
            rsi_score = ((rsi - 50) * 2)  # -100 to +100
            macd_score = (histogram / abs(macd) if abs(macd) > 0 else 0) * 100
            trend_score = adx_value * (1 if trend_direction == "Uptrend" else -1)
            roc_score = min(max(roc * 2, -100), 100)
            vol_score = min(max(vol_momentum, -100), 100)
            
            # Weight the components
            weights = {
                'rsi': 0.2,
                'macd': 0.25,
                'trend': 0.25,
                'roc': 0.2,
                'volume': 0.1
            }
            
            # Calculate final score
            momentum_score = (
                rsi_score * weights['rsi'] +
                macd_score * weights['macd'] +
                trend_score * weights['trend'] +
                roc_score * weights['roc'] +
                vol_score * weights['volume']
            )
            
            return {
                'total_score': round(momentum_score, 2),
                'components': {
                    'rsi_score': round(rsi_score, 2),
                    'macd_score': round(macd_score, 2),
                    'trend_score': round(trend_score, 2),
                    'roc_score': round(roc_score, 2),
                    'volume_score': round(vol_score, 2)
                },
                'interpretation': 'Strong Bullish' if momentum_score > 70 else
                                'Bullish' if momentum_score > 30 else
                                'Neutral' if momentum_score > -30 else
                                'Bearish' if momentum_score > -70 else
                                'Strong Bearish'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {str(e)}")
            return {'total_score': 0, 'components': {}, 'interpretation': 'Error'}

    def detect_market_regime(self, candles: List[Dict]) -> Dict:
        """
        Detect current market regime using volatility, trend strength, and price patterns.
        """
        try:
            prices = np.array([c['close'] for c in candles])
            
            # Calculate volatility
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            # Get trend strength
            adx_value, trend_direction = self.technical_analysis.get_trend_strength(candles)
            
            # Calculate price range
            price_range = (np.max(prices) - np.min(prices)) / np.mean(prices)
            
            # Detect regime
            regime = MarketRegime.RANGING
            confidence = 0.0
            
            if volatility > 0.04:  # High volatility threshold
                regime = MarketRegime.VOLATILE
                confidence = min((volatility - 0.04) * 10, 1.0)
            elif adx_value > 25:  # Strong trend threshold
                regime = MarketRegime.TRENDING
                confidence = min((adx_value - 25) / 75, 1.0)
            elif price_range < 0.02:  # Tight range threshold
                regime = MarketRegime.RANGING
                confidence = min((0.02 - price_range) * 50, 1.0)
            
            # Check for breakouts
            bb_upper, bb_middle, bb_lower = self.technical_analysis.compute_bollinger_bands(candles)
            if prices[-1] > bb_upper or prices[-1] < bb_lower:
                regime = MarketRegime.BREAKOUT
                confidence = min(abs(prices[-1] - bb_middle) / (bb_upper - bb_middle), 1.0)
            
            # Check for reversals
            momentum = self.calculate_momentum_score(candles)
            if abs(momentum['total_score']) > 70 and np.sign(momentum['total_score']) != np.sign(returns[-1]):
                regime = MarketRegime.REVERSAL
                confidence = min(abs(momentum['total_score']) / 100, 1.0)
            
            return {
                'regime': regime,
                'confidence': round(confidence * 100, 2),
                'metrics': {
                    'volatility': round(volatility * 100, 2),
                    'trend_strength': round(adx_value, 2),
                    'price_range': round(price_range * 100, 2)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return {'regime': MarketRegime.RANGING, 'confidence': 0.0, 'metrics': {}}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crypto Market Analyzer')
    
    parser.add_argument(
        '--product_id',
        type=str,
        choices=VALID_PRODUCTS,
        default='DOGE-USDC',
        help='Product ID to analyze (e.g., BTC-USDC, ETH-USDC)'
    )
    
    parser.add_argument(
        '--granularity',
        type=str,
        choices=VALID_GRANULARITIES,
        default='ONE_HOUR',
        help='Candle interval granularity'
    )
    
    parser.add_argument(
        '--list-products',
        action='store_true',
        help='List all available products'
    )
    
    parser.add_argument(
        '--list-granularities',
        action='store_true',
        help='List all available granularities'
    )
    
    return parser.parse_args()

def list_options():
    """Print available options for products and granularities."""
    print("\nAvailable Products:")
    for product in VALID_PRODUCTS:
        print(f"  - {product}")
    
    print("\nAvailable Granularities:")
    for granularity in VALID_GRANULARITIES:
        print(f"  - {granularity}")
    print()

def main():
    # Configure logging with more detail
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    args = parse_arguments()
    
    if args.list_products or args.list_granularities:
        list_options()
        return

    analyzer = MarketAnalyzer(
        product_id=args.product_id,
        candle_interval=args.granularity
    )
    
    try:
        analysis = analyzer.get_market_signal()
        
        # Enhanced formatted output
        print("\n====== Comprehensive Market Analysis Report ======")
        print(f"Timestamp: {analysis['timestamp']}")
        print(f"Product: {analysis['product_id']}")
        print(f"Current Price: ${analysis['current_price']:,.4f}")
        
        # Market Overview Section
        print("\n=== Market Overview ===")
        print(f"Signal: {analysis['signal']}")
        print(f"Position: {analysis['position']}")
        print(f"Confidence: {analysis['confidence']*100:.1f}%")
        print(f"Market Condition: {analysis['market_condition']}")
        print(f"Signal Stability: {analysis['signal_stability']}")
        
        # Technical Indicators Section
        print("\n=== Technical Indicators ===")
        indicators = analysis['indicators']
        print(f"RSI: {indicators['rsi']:.2f} ({'Overbought' if indicators['rsi'] > 70 else 'Oversold' if indicators['rsi'] < 30 else 'Neutral'})")
        print(f"MACD: {indicators['macd']:.4f}, MACD Signal: {indicators['macd_signal']:.4f}, MACD Histogram: {indicators['macd_histogram']:.4f}")
        print(f"ADX: {indicators['adx']:.2f} ({'Strong Trend' if indicators['adx'] > 25 else 'Weak Trend'})")
        print(f"Trend Direction: {indicators['trend_direction']}")
        
        # Bollinger Bands
        print("\n=== Price Channels ===")
        print(f"Bollinger Upper: ${indicators['bollinger_upper']:.4f}, Bollinger Middle: ${indicators['bollinger_middle']:.4f}, Bollinger Lower: ${indicators['bollinger_lower']:.4f}")
        
        # Volume Analysis Section
        print("\n=== Volume Analysis ===")
        volume = analysis['volume_analysis']
        print(f"Volume Change: {volume['change']:.1f}%")
        print(f"Volume Trend: {volume['trend']}")
        print(f"Volume Strength: {volume['strength']}")
        print(f"Price Change: {volume['price_change']:.1f}%")
        print(f"Volume Confirmation: {'Yes' if volume['is_confirming'] else 'No'}")
        
        # Pattern Recognition Section
        print("\n=== Pattern Analysis ===")
        patterns = analysis['patterns']
        print(f"Current Pattern: {patterns['type']}")
        if patterns['type'] != "None":
            print(f"Pattern Confidence: {patterns['confidence']*100:.1f}%")
            if patterns['target']:
                print(f"Pattern Target: ${patterns['target']:.4f}")
            if patterns['stop_loss']:
                print(f"Suggested Stop Loss: ${patterns['stop_loss']:.4f}")
        
        # Pattern History
        print("\n=== Recent Pattern History ===")
        for pattern in analysis['pattern_history'][-3:]:  # Show last 3 patterns
            print(f"• {pattern['pattern']} (Confidence: {pattern['confidence']*100:.1f}%) - {pattern['timestamp']}")
        
        # Add Regime Analysis Section after Pattern History
        print("\n=== Market Regime Analysis ===")
        regime = analysis['regime_analysis']
        print(f"Current Regime: {regime['regime']}")
        print(f"Confidence: {regime['confidence']:.1f}%")
        print("\nRegime Metrics:")
        print(f"• Volatility: {regime['metrics']['volatility']:.1f}%")
        print(f"• Trend Strength: {regime['metrics']['trend_strength']:.1f}")
        print(f"• Price Range: {regime['metrics']['price_range']:.1f}%")
        
        # Add Momentum Analysis Section
        print("\n=== Momentum Analysis ===")
        momentum = analysis['momentum_analysis']
        print(f"Overall Momentum: {momentum['interpretation']}")
        print(f"Total Score: {momentum['total_score']:.1f}")
        print("\nComponent Scores:")
        for component, score in momentum['components'].items():
            print(f"• {component.replace('_', ' ').title()}: {score:.1f}")
        
        # Risk Metrics Section
        print("\n=== Risk Analysis ===")
        risk = analysis['risk_metrics']
        print(f"Dynamic Risk Level: {risk['dynamic_risk']*100:.1f}%")
        if 'volatility' in risk:
            print(f"Current Volatility: {risk['volatility']*100:.1f}%")
        if 'risk_reward_ratio' in risk:
            print(f"Risk/Reward Ratio: {risk['risk_reward_ratio']:.2f}")
        
        # Trading Recommendation Section
        print("\n=== Trading Recommendation ===")
        print(analysis['recommendation'])
        
        # Key Levels and Potential Moves
        print("\n=== Key Levels & Potential Moves ===")
        atr = indicators.get('atr', (indicators['bollinger_upper'] - indicators['bollinger_lower']) / 4)
        current_price = analysis['current_price']
        
        print(f"Potential Bullish Targets:")
        print(f"• Conservative: ${current_price * 1.01:.4f} (+1%)")
        print(f"• Moderate: ${current_price * 1.02:.4f} (+2%)")
        print(f"• Aggressive: ${current_price * 1.05:.4f} (+5%)")
        
        print(f"\nPotential Bearish Targets:")
        print(f"• Conservative: ${current_price * 0.99:.4f} (-1%)")
        print(f"• Moderate: ${current_price * 0.98:.4f} (-2%)")
        print(f"• Aggressive: ${current_price * 0.95:.4f} (-5%)")
        
        # Add directional bias analysis with move specifics
        print("\n=== Directional Bias & Move Analysis ===")
        bullish_points = 0
        bearish_points = 0
        
        # RSI Analysis
        if indicators['rsi'] > 50:
            bullish_points += 1
        else:
            bearish_points += 1
            
        # MACD Analysis
        if indicators['macd'] > indicators['macd_signal']:
            bullish_points += 1
        else:
            bearish_points += 1
            
        # Trend Direction
        if indicators['trend_direction'] == "Uptrend":
            bullish_points += 2
        elif indicators['trend_direction'] == "Downtrend":
            bearish_points += 2
            
        # Volume Analysis
        if volume['is_confirming'] and volume['price_change'] > 0:
            bullish_points += 1
        elif volume['is_confirming'] and volume['price_change'] < 0:
            bearish_points += 1
            
        # Price relative to Bollinger Bands
        if current_price > indicators['bollinger_middle']:
            bullish_points += 1
        else:
            bearish_points += 1
            
        # Calculate confidence percentage
        total_points = bullish_points + bearish_points
        bullish_confidence = (bullish_points / total_points * 100) if total_points > 0 else 50
        bearish_confidence = (bearish_points / total_points * 100) if total_points > 0 else 50
        
        # Calculate move specifics
        atr = indicators.get('atr', (indicators['bollinger_upper'] - indicators['bollinger_lower']) / 4)
        bb_width = indicators['bollinger_upper'] - indicators['bollinger_lower']
        price_volatility = bb_width / indicators['bollinger_middle']
        
        # Define move characteristics
        move_speed = "Rapid" if price_volatility > 0.03 else "Gradual"
        move_strength = "Strong" if abs(indicators['macd']) > abs(indicators['macd_signal']) * 1.5 else "Moderate"
        
        print("Move Analysis:")
        if bullish_points > bearish_points:
            print(f"BULLISH with {bullish_confidence:.1f}% confidence")
            print("\nMove Characteristics:")
            print(f"• Expected Move Type: {move_speed} {move_strength} Advance")
            print(f"• Momentum: {'Accelerating' if indicators['macd_histogram'] > 0 else 'Decelerating'}")
            print(f"• Volume Profile: {'Supporting' if volume['is_confirming'] else 'Lacking'}")
            
            print("\nPrice Targets:")
            print(f"• Initial Target: ${(current_price + atr):.4f} (+{(atr/current_price)*100:.1f}%)")
            print(f"• Secondary Target: ${indicators['bollinger_upper']:.4f} (+{((indicators['bollinger_upper']-current_price)/current_price)*100:.1f}%)")
            print(f"• Extended Target: ${(indicators['bollinger_upper'] + atr):.4f} (+{((indicators['bollinger_upper']+atr-current_price)/current_price)*100:.1f}%)")
            
            print("\nSupporting Factors:")
            if indicators['rsi'] > 50:
                print(f"• RSI showing upward momentum ({indicators['rsi']:.1f})")
            if indicators['macd'] > indicators['macd_signal']:
                print(f"• MACD bullish crossover (Spread: {(indicators['macd']-indicators['macd_signal']):.4f})")
            if indicators['trend_direction'] == "Uptrend":
                print("• Established uptrend with higher lows")
            if volume['is_confirming'] and volume['price_change'] > 0:
                print(f"• Volume increased by {volume['change']:.1f}% supporting price action")
            if current_price > indicators['bollinger_middle']:
                print("• Price trading above BB middle band showing strength")
                
        elif bearish_points > bullish_points:
            print(f"BEARISH with {bearish_confidence:.1f}% confidence")
            print("\nMove Characteristics:")
            print(f"• Expected Move Type: {move_speed} {move_strength} Decline")
            print(f"• Momentum: {'Accelerating' if indicators['macd_histogram'] < 0 else 'Decelerating'}")
            print(f"• Volume Profile: {'Supporting' if volume['is_confirming'] else 'Lacking'}")
            
            print("\nPrice Targets:")
            print(f"• Initial Target: ${(current_price - atr):.4f} (-{(atr/current_price)*100:.1f}%)")
            print(f"• Secondary Target: ${indicators['bollinger_lower']:.4f} (-{((current_price-indicators['bollinger_lower'])/current_price)*100:.1f}%)")
            print(f"• Extended Target: ${(indicators['bollinger_lower'] - atr):.4f} (-{((current_price-(indicators['bollinger_lower']-atr))/current_price)*100:.1f}%)")
            
            print("\nSupporting Factors:")
            if indicators['rsi'] < 50:
                print(f"• RSI showing downward momentum ({indicators['rsi']:.1f})")
            if indicators['macd'] < indicators['macd_signal']:
                print(f"• MACD bearish crossover (Spread: {(indicators['macd_signal']-indicators['macd']):.4f})")
            if indicators['trend_direction'] == "Downtrend":
                print("• Established downtrend with lower highs")
            if volume['is_confirming'] and volume['price_change'] < 0:
                print(f"• Volume increased by {volume['change']:.1f}% supporting price action")
            if current_price < indicators['bollinger_middle']:
                print("• Price trading below BB middle band showing weakness")
        else:
            print("NEUTRAL - No Clear Directional Bias")
            print("\nConsolidation Analysis:")
            print(f"• Price Range: ${(current_price - atr):.4f} to ${(current_price + atr):.4f}")
            print(f"• Volatility: {'High' if price_volatility > 0.03 else 'Low'} ({price_volatility*100:.1f}%)")
            print(f"• Volume Profile: {volume['trend']} on {volume['change']:.1f}% change")
            print("\nBreakout Levels:")
            print(f"• Bullish Breakout Above: ${indicators['bollinger_upper']:.4f}")
            print(f"• Bearish Breakdown Below: ${indicators['bollinger_lower']:.4f}")
            print("\nRecommendation:")
            print("• Consider waiting for stronger directional signals")
            print("• Monitor for breakout of recent trading range")
            print("• Prepare for potential volatility expansion")

        # Enhanced Probability Analysis Section
        print("\n====== Detailed Move Analysis ======")
        prob = analysis['probability_analysis']
        print(f"\nOverall Success Probability: {prob['total_probability']:.1f}%")
        print(f"Confidence Level: {prob['confidence_level']}")
        
        print("\nMove Quality:")
        move_quality = prob['move_quality']
        print(f"• Expected Speed: {move_quality['expected_speed']}")
        print(f"• Expected Volatility: {move_quality['expected_volatility']}")
        print(f"• Continuation Probability: {move_quality['continuation_probability']}")
        print(f"• Reversal Risk: {move_quality['reversal_risk']}")
        
        print("\nMove Characteristics:")
        chars = prob['move_characteristics']
        
        print("\nTrend Quality:")
        trend = chars['trend_quality']
        print(f"• Strength: {trend['strength']}")
        print(f"• Duration: {trend['duration']}")
        print(f"• Momentum: {trend['momentum']}")
        
        print("\nMomentum Analysis:")
        momentum = chars['momentum']
        print(f"• Condition: {momentum['condition']}")
        print(f"• Strength: {momentum['strength']:.2f}")
        print(f"• Divergence: {momentum['divergence']}")
        
        print("\nVolume Quality:")
        volume = chars['volume_quality']
        print(f"• Trend: {volume['trend']}")
        print(f"• Strength: {volume['strength']}")
        print(f"• Consistency: {volume['consistency']}")
        print(f"• Price Alignment: {volume['price_alignment']}")
        
        print("\nPattern Analysis:")
        pattern = chars['pattern_analysis']
        print(f"• Type: {pattern['type']}, • Reliability: {pattern['reliability']:.2f}, • Completion: {pattern['completion']:.1f}%")
        
        print("\nFailure Points:")
        failure = prob['failure_points']
        if failure['immediate_stop']:
            print(f"• Immediate Stop: ${failure['immediate_stop']:.4f}")
        if failure['trend_reversal_point']:
            print(f"• Trend Reversal: ${failure['trend_reversal_point']:.4f}")
        if failure['momentum_failure_level']:
            print(f"• Momentum Failure: ${failure['momentum_failure_level']:.4f}")
        
        print("\nContributing Factors: " + ", ".join([f"• {factor}: {value:.1f}%" for factor, value in prob['factors']]))

    except Exception as e:
        print(f"\nError running market analysis: {str(e)}")

if __name__ == "__main__":
    main() 