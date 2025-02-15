import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta, UTC
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis, SignalType, TechnicalAnalysisConfig
from config import API_KEY, API_SECRET
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
    "UNI-USDC",
    "SHIB-USDC"
]

class ScalpingAnalyzer:
    def __init__(self, product_id: str = 'BTC-USDC', candle_interval: str = 'ONE_MINUTE'):
        self.product_id = product_id
        self.candle_interval = candle_interval
        self.coinbase_service = CoinbaseService(API_KEY, API_SECRET)
        self.client = RESTClient(API_KEY, API_SECRET)
        self.historical_data = HistoricalData(self.client)
        
        # Initialize technical analysis with scalping-specific configuration
        self.ta_config = TechnicalAnalysisConfig(
            rsi_period=9,  # Shorter period for scalping
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            bollinger_window=20,
            bollinger_std=2.0,
            volatility_threshold=0.01,  # More sensitive to volatility
            risk_per_trade=0.01,  # Smaller risk per trade for scalping
            atr_multiplier=1.5  # Tighter stops for scalping
        )
        
        self.technical_analysis = TechnicalAnalysis(
            self.coinbase_service,
            config=self.ta_config,
            candle_interval=candle_interval,
            product_id=product_id
        )
        
        self.logger = logging.getLogger(__name__)

    def get_scalping_opportunities(self) -> Dict:
        """Analyze the market for scalping opportunities."""
        try:
            # Get recent candles (last hour)
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(hours=1)
            
            candles = self.historical_data.get_historical_data(
                self.product_id,
                start_time,
                end_time,
                self.candle_interval
            )

            if not candles:
                raise ValueError("No candle data available")

            formatted_candles = self._format_candles(candles)
            current_price = float(formatted_candles[-1]['close'])
            atr = self.technical_analysis.compute_atr(formatted_candles)
            
            # Get key indicators
            rsi = self.technical_analysis.compute_rsi(self.product_id, formatted_candles)
            macd, signal, histogram = self.technical_analysis.compute_macd(self.product_id, formatted_candles)
            upper_band, middle_band, lower_band = self.technical_analysis.compute_bollinger_bands(formatted_candles)
            consolidation_info = self.technical_analysis.detect_consolidation(formatted_candles)
            volume_info = self.technical_analysis.analyze_volume_confirmation(formatted_candles)
            
            scalping_setups = {
                'timestamp': datetime.now(UTC).isoformat(),
                'current_price': current_price,
                'opportunities': []
            }

            # Check for support/resistance rejections
            if consolidation_info.get('rejection_event'):
                rejection = consolidation_info['rejection_event']
                if rejection['volume_confirmation'] and rejection['confirming_candles'] >= 1:
                    setup = {
                        'type': f"{rejection['type'].title()} Rejection Scalp",
                        'direction': 'SHORT' if rejection['type'] == 'resistance' else 'LONG',
                        'entry': rejection['price_level'],
                        'stop_loss': rejection['price_level'] + (atr * 0.5) if rejection['type'] == 'resistance'
                                   else rejection['price_level'] - (atr * 0.5),
                        'target': rejection['price_level'] - (atr * 0.75) if rejection['type'] == 'resistance'
                                 else rejection['price_level'] + (atr * 0.75),
                        'strength': 'Strong' if rejection['volume_ratio'] > 1.5 else 'Moderate',
                        'confirmation': {
                            'volume': rejection['volume_confirmation'],
                            'candles': rejection['confirming_candles'],
                            'volume_ratio': rejection['volume_ratio']
                        }
                    }
                    scalping_setups['opportunities'].append(setup)

            # Check for oversold/overbought bounces
            if rsi <= 30 and volume_info['volume_trend'] == "Increasing":
                setup = {
                    'type': 'Oversold Bounce Scalp',
                    'direction': 'LONG',
                    'entry': current_price,
                    'stop_loss': current_price - (atr * 0.5),
                    'target': current_price + (atr * 0.75),
                    'strength': 'Strong' if rsi < 20 else 'Moderate',
                    'confirmation': {
                        'rsi': rsi,
                        'volume_increasing': True,
                        'near_support': current_price < lower_band
                    }
                }
                scalping_setups['opportunities'].append(setup)
            elif rsi >= 70 and volume_info['volume_trend'] == "Increasing":
                setup = {
                    'type': 'Overbought Reversal Scalp',
                    'direction': 'SHORT',
                    'entry': current_price,
                    'stop_loss': current_price + (atr * 0.5),
                    'target': current_price - (atr * 0.75),
                    'strength': 'Strong' if rsi > 80 else 'Moderate',
                    'confirmation': {
                        'rsi': rsi,
                        'volume_increasing': True,
                        'near_resistance': current_price > upper_band
                    }
                }
                scalping_setups['opportunities'].append(setup)

            # Add risk metrics
            scalping_setups['risk_metrics'] = {
                'atr': atr,
                'avg_volume': volume_info['average_volume'],
                'required_volume': volume_info['average_volume'] * 1.2,
                'max_spread': atr * 0.1,
                'min_risk_reward': 1.5
            }

            return scalping_setups

        except Exception as e:
            self.logger.error(f"Error finding scalping opportunities: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now(UTC).isoformat(),
                'opportunities': []
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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crypto Scalping Analyzer')
    
    parser.add_argument(
        '--product_id',
        type=str,
        choices=VALID_PRODUCTS,
        default='BTC-USDC',
        help='Product ID to analyze (e.g., BTC-USDC, ETH-USDC)'
    )
    
    parser.add_argument(
        '--granularity',
        type=str,
        choices=VALID_GRANULARITIES,
        default='ONE_MINUTE',  # Default to one minute for scalping
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse command line arguments
    args = parse_arguments()
    
    # If user requested to list options, show them and exit
    if args.list_products or args.list_granularities:
        list_options()
        return

    # Create analyzer instance with command line arguments
    analyzer = ScalpingAnalyzer(
        product_id=args.product_id,
        candle_interval=args.granularity
    )
    
    try:
        scalping_setups = analyzer.get_scalping_opportunities()
        
        print("\n=== Scalping Opportunities Report ===")
        print(f"Timestamp: {scalping_setups['timestamp']}")
        print(f"Product: {args.product_id}")
        print(f"Timeframe: {args.granularity}")
        print(f"Current Price: ${scalping_setups['current_price']:.4f}")
        
        if scalping_setups['opportunities']:
            print("\nActive Setups:")
            for setup in scalping_setups['opportunities']:
                print(f"\n{setup['type']} ({setup['direction']}):")
                print(f"• Entry: ${setup['entry']:.4f}")
                print(f"• Stop Loss: ${setup['stop_loss']:.4f}")
                print(f"• Target: ${setup['target']:.4f}")
                print(f"• Strength: {setup['strength']}")
                print("\nConfirmation Factors:")
                for factor, value in setup['confirmation'].items():
                    if isinstance(value, float):
                        print(f"  - {factor}: {value:.4f}")
                    else:
                        print(f"  - {factor}: {value}")
                
                # Calculate risk metrics
                risk = abs(setup['entry'] - setup['stop_loss'])
                reward = abs(setup['target'] - setup['entry'])
                risk_reward = reward / risk if risk > 0 else 0
                
                print(f"\nTrade Metrics:")
                print(f"• Risk: ${risk:.4f}")
                print(f"• Reward: ${reward:.4f}")
                print(f"• R:R Ratio: {risk_reward:.2f}")
        else:
            print("\nNo scalping opportunities found at the moment")
        
        if 'risk_metrics' in scalping_setups:
            print("\nScalping Risk Parameters:")
            print(f"• ATR: ${scalping_setups['risk_metrics']['atr']:.4f}")
            print(f"• Required Volume: {scalping_setups['risk_metrics']['required_volume']:.2f}")
            print(f"• Maximum Spread: ${scalping_setups['risk_metrics']['max_spread']:.4f}")
            print(f"• Minimum R:R: {scalping_setups['risk_metrics']['min_risk_reward']:.1f}")

    except Exception as e:
        print(f"\nError analyzing scalping opportunities: {str(e)}")

if __name__ == "__main__":
    main() 