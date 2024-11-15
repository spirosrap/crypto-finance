import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta, UTC
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis, SignalType, MarketCondition, TechnicalAnalysisConfig
from config import API_KEY, API_SECRET, NEWS_API_KEY
from historicaldata import HistoricalData
from coinbase.rest import RESTClient
import argparse

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

            # Get comprehensive market analysis
            analysis = self.technical_analysis.analyze_market(formatted_candles)
            
            # Get additional trend information
            adx_value, trend_direction = self.technical_analysis.get_trend_strength(formatted_candles)
            
            # Calculate key indicators
            rsi = self.technical_analysis.compute_rsi(self.product_id, formatted_candles)
            macd, signal, histogram = self.technical_analysis.compute_macd(self.product_id, formatted_candles)
            upper_band, middle_band, lower_band = self.technical_analysis.compute_bollinger_bands(formatted_candles)
            current_price = float(formatted_candles[-1]['close'])
            
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
                'recommendation': self._generate_recommendation(analysis['signal'].signal_type)
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
        """Generate a detailed trading recommendation including position type and risk management."""
        recommendations = {
            SignalType.STRONG_BUY: {
                'position': 'LONG',
                'message': "Strong buy signal detected. Consider opening a LONG position:\n"
                          "• Entry: Current market price\n"
                          "• Position Type: LONG\n"
                          "• Leverage: 1-2x maximum\n"
                          "• Stop Loss: Place below recent support or -2.5 ATR\n"
                          "• Take Profit: Set multiple targets at 1.5:1 and 2:1 risk-reward ratios\n"
                          "• Risk Management: Size position to risk only 1-2% of portfolio"
            },
            SignalType.BUY: {
                'position': 'LONG',
                'message': "Bullish conditions detected. Consider a conservative LONG position:\n"
                          "• Entry: Look for pullbacks to support levels\n"
                          "• Position Type: LONG\n"
                          "• Leverage: 1x only\n"
                          "• Stop Loss: Place below entry support level\n"
                          "• Take Profit: Set target at 1.5:1 risk-reward ratio\n"
                          "• Risk Management: Size position to risk only 1% of portfolio"
            },
            SignalType.HOLD: {
                'position': 'NEUTRAL',
                'message': "Market conditions are neutral:\n"
                          "• Action: Hold existing positions or stay in cash\n"
                          "• Watch for: Consolidation breakout or breakdown\n"
                          "• Strategy: Wait for clearer directional signals\n"
                          "• Risk Management: Maintain tight stops on any existing positions"
            },
            SignalType.SELL: {
                'position': 'SHORT',
                'message': "Bearish conditions detected. Consider a conservative SHORT position:\n"
                          "• Entry: Look for rallies to resistance levels\n"
                          "• Position Type: SHORT\n"
                          "• Leverage: 1x only\n"
                          "• Stop Loss: Place above entry resistance level\n"
                          "• Take Profit: Set target at 1.5:1 risk-reward ratio\n"
                          "• Risk Management: Size position to risk only 1% of portfolio"
            },
            SignalType.STRONG_SELL: {
                'position': 'SHORT',
                'message': "Strong sell signal detected. Consider opening a SHORT position:\n"
                          "• Entry: Current market price\n"
                          "• Position Type: SHORT\n"
                          "• Leverage: 1-2x maximum\n"
                          "• Stop Loss: Place above recent resistance or +2.5 ATR\n"
                          "• Take Profit: Set multiple targets at 1.5:1 and 2:1 risk-reward ratios\n"
                          "• Risk Management: Size position to risk only 1-2% of portfolio"
            }
        }
        
        rec = recommendations.get(signal_type, {
            'position': 'NEUTRAL',
            'message': "No clear trading opportunity. Wait for better setup."
        })
        
        return f"Position: {rec['position']}\n\n{rec['message']}"

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
    # Parse command line arguments
    args = parse_arguments()
    
    # If user requested to list options, show them and exit
    if args.list_products or args.list_granularities:
        list_options()
        return
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create analyzer instance with command line arguments
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