import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta, UTC
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis, SignalType, MarketCondition, TechnicalAnalysisConfig
from config import API_KEY, API_SECRET, NEWS_API_KEY
from historicaldata import HistoricalData
from coinbase.rest import RESTClient
import argparse
import time

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
        """Generate a detailed trading recommendation including consolidation patterns."""
        try:
            if not self._current_candles:
                return "No market data available for recommendation"
            
            # Get current price and indicators
            current_price = float(self._current_candles[-1]['close'])
            atr = self.technical_analysis.compute_atr(self._current_candles)
            consolidation_info = self.technical_analysis.detect_consolidation(self._current_candles)
            volume_info = self.technical_analysis.analyze_volume_confirmation(self._current_candles)
            
            # Calculate key levels
            stop_loss_atr = atr * self.ta_config.atr_multiplier
            take_profit_1r = stop_loss_atr * 1.5  # 1.5:1 reward-risk
            take_profit_2r = stop_loss_atr * 2.0  # 2:1 reward-risk
            take_profit_3r = stop_loss_atr * 3.0  # 3:1 reward-risk
            
            # Calculate support and resistance levels
            resistance_level = consolidation_info['upper_channel']
            support_level = consolidation_info['lower_channel']
            
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
                              f"Key Levels:\n"
                              f"• Support: ${support_level:.4f}\n"
                              f"• Resistance: ${resistance_level:.4f}\n\n"
                              f"Recommendation:\n"
                              f"• Hold existing positions\n"
                              f"• Wait for price to break ${resistance_level:.4f} resistance or\n"
                              f"  ${support_level:.4f} support with volume confirmation"
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
            
            return f"Position: {rec['position']}\n\n{rec['message']}{consolidation_message}"
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {str(e)}")
            return "Error generating recommendation"

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
        level=logging.INFO,  # Change to DEBUG level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create analyzer instance with command line arguments
    args = parse_arguments()
    
    if args.list_products or args.list_granularities:
        list_options()
        return

    analyzer = MarketAnalyzer(
        product_id=args.product_id,
        candle_interval=args.granularity
    )
    
    try:
        # Get and print market analysis
        analysis = analyzer.get_market_signal()
        
        # Print formatted results
        print("\n=== Market Analysis Report ===")
        print(f"Timestamp: {analysis['timestamp']}")
        print(f"Product: {analysis['product_id']}")
        print(f"Current Price: ${analysis['current_price']:,.2f}")
        print(f"\nSignal: {analysis['signal']}")
        print(f"Position: {analysis['position']}")
        print(f"Confidence: {analysis['confidence']*100:.1f}%")
        print(f"Market Condition: {analysis['market_condition']}")
        
        if 'indicators' in analysis:
            print("\nKey Indicators:")
            for name, value in analysis['indicators'].items():
                print(f"  {name}: {value}")
        
        if 'risk_metrics' in analysis:
            print("\nRisk Metrics:")
            for name, value in analysis['risk_metrics'].items():
                if isinstance(value, float):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")
        
        print(f"\nRecommendation:")
        print(analysis['recommendation'])

    except Exception as e:
        print(f"\nError running market analysis: {str(e)}")

if __name__ == "__main__":
    main() 