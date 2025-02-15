import logging
from datetime import datetime, timedelta, UTC
import time
from market_analyzer import MarketAnalyzer
from typing import Dict, List
import json
import os
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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Continuous Market Monitor')
    
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
        default='ONE_MINUTE',
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

class ContinuousMarketMonitor:
    def __init__(self, product_id: str = 'BTC-USDC', candle_interval: str = 'ONE_MINUTE'):
        self.product_id = product_id
        self.candle_interval = candle_interval
        self.analyzer = MarketAnalyzer(product_id, candle_interval)
        self.logger = logging.getLogger(__name__)
        self.signal_history: List[Dict] = []
        self.last_signal = None
        self.last_signal_time = None
        
        # Create signals directory if it doesn't exist
        self.signals_dir = 'signals'
        if not os.path.exists(self.signals_dir):
            os.makedirs(self.signals_dir)

    def run(self, interval_seconds: int = 60):
        print(f"\nStarting Continuous Market Monitor for {self.product_id}")
        print(f"Checking every {interval_seconds} seconds")
        print("Press Ctrl+C to stop\n")
        
        while True:
            try:
                # Get current market analysis
                analysis = self.analyzer.get_market_signal()
                
                # Process and log the analysis
                self._process_analysis(analysis)
                
                # Save signal history periodically
                self._save_signal_history()
                
                # Wait for next check
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                print("\nStopping Market Monitor...")
                self._save_signal_history()
                break
            except Exception as e:
                self.logger.error(f"Error in market monitor: {str(e)}")
                time.sleep(interval_seconds)

    def _process_analysis(self, analysis: Dict):
        current_time = datetime.now(UTC)
        
        # Check if this is a new signal
        is_new_signal = self._is_new_signal(analysis)
        
        if is_new_signal:
            # Record the signal
            signal_record = {
                'timestamp': current_time.isoformat(),
                'product_id': self.product_id,
                'price': analysis['current_price'],
                'signal': analysis['signal'],
                'position': analysis['position'],
                'confidence': analysis['confidence'],
                'market_condition': analysis['market_condition']
            }
            
            self.signal_history.append(signal_record)
            
            # Print the signal with scalping opportunities
            print(f"\n=== Market Analysis & Scalping Opportunities ===")
            print(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Price: ${analysis['current_price']:.4f}")
            print(f"Signal: {analysis['signal']}")
            print(f"Position: {analysis['position']}")
            print(f"Confidence: {analysis['confidence']*100:.1f}%")
            print(f"Market Condition: {analysis['market_condition']}")
            
            # Add scalping-specific information
            if 'rejection_event' in analysis and analysis['rejection_event']:
                rejection = analysis['rejection_event']
                print(f"\nScalping Setup ({rejection['type'].upper()}):")
                print(f"• Level: ${rejection['price_level']:.4f}")
                print(f"• Stop Loss: ${(rejection['price_level'] + (analysis['risk_metrics']['atr'] * 0.5)):.4f}")
                print(f"• Target: ${(rejection['price_level'] + (analysis['risk_metrics']['atr'] * 0.75)):.4f}")
                print(f"• Volume Confirmation: {'Yes' if rejection['volume_confirmation'] else 'No'}")
                print(f"• Confirming Candles: {rejection['confirming_candles']}")
            
            if 'indicators' in analysis:
                print("\nKey Indicators:")
                for name, value in analysis['indicators'].items():
                    if isinstance(value, float):
                        print(f"  {name}: {value:.4f}")
                    else:
                        print(f"  {name}: {value}")
            
            if 'risk_metrics' in analysis:
                print("\nScalping Parameters:")
                print(f"• ATR: ${analysis['risk_metrics']['atr']:.4f}")
                print(f"• Suggested Stop: ${(analysis['current_price'] - analysis['risk_metrics']['atr'] * 0.5):.4f}")
                print(f"• Suggested Target: ${(analysis['current_price'] + analysis['risk_metrics']['atr'] * 0.75):.4f}")
                print(f"• Risk/Reward: 1.5")
            
            # Add potential scalping opportunities based on conditions
            print("\nScalping Opportunities:")
            
            # RSI-based opportunities
            if 'indicators' in analysis and 'rsi' in analysis['indicators']:
                rsi = analysis['indicators']['rsi']
                if rsi <= 30:
                    print("• Oversold Bounce Opportunity (LONG):")
                    print(f"  - Entry: ${analysis['current_price']:.4f}")
                    print(f"  - Stop: ${(analysis['current_price'] - analysis['risk_metrics']['atr'] * 0.5):.4f}")
                    print(f"  - Target: ${(analysis['current_price'] + analysis['risk_metrics']['atr'] * 0.75):.4f}")
                elif rsi >= 70:
                    print("• Overbought Reversal Opportunity (SHORT):")
                    print(f"  - Entry: ${analysis['current_price']:.4f}")
                    print(f"  - Stop: ${(analysis['current_price'] + analysis['risk_metrics']['atr'] * 0.5):.4f}")
                    print(f"  - Target: ${(analysis['current_price'] - analysis['risk_metrics']['atr'] * 0.75):.4f}")
            
            # Bollinger Bands opportunities
            if all(k in analysis['indicators'] for k in ['bollinger_upper', 'bollinger_lower']):
                if analysis['current_price'] <= analysis['indicators']['bollinger_lower']:
                    print("• Bollinger Band Bounce (LONG):")
                    print(f"  - Entry: ${analysis['current_price']:.4f}")
                    print(f"  - Stop: ${(analysis['current_price'] - analysis['risk_metrics']['atr'] * 0.5):.4f}")
                    print(f"  - Target: ${(analysis['current_price'] + analysis['risk_metrics']['atr']):.4f}")
                elif analysis['current_price'] >= analysis['indicators']['bollinger_upper']:
                    print("• Bollinger Band Reversal (SHORT):")
                    print(f"  - Entry: ${analysis['current_price']:.4f}")
                    print(f"  - Stop: ${(analysis['current_price'] + analysis['risk_metrics']['atr'] * 0.5):.4f}")
                    print(f"  - Target: ${(analysis['current_price'] - analysis['risk_metrics']['atr']):.4f}")
            
            print("\nRisk Management:")
            print(f"• Maximum Position Size: 1% of account")
            print(f"• Required Volume > {analysis['risk_metrics'].get('required_volume', 0):.2f}")
            print(f"• Maximum Hold Time: 15 minutes")
            print("• Use Market Orders for Entry/Exit")
            
            print("=" * 50)
            
            # Update last signal
            self.last_signal = analysis
            self.last_signal_time = current_time

    def _is_new_signal(self, analysis: Dict) -> bool:
        """Determine if this is a new market signal."""
        if self.last_signal is None or self.last_signal_time is None:
            return True
            
        # Check if enough time has passed since last signal
        time_passed = datetime.now(UTC) - self.last_signal_time
        if time_passed < timedelta(minutes=1):  # Minimum 1 minute between signals
            return False
            
        # Check if signal has changed
        if (self.last_signal['signal'] != analysis['signal'] or
            self.last_signal['position'] != analysis['position'] or
            abs(self.last_signal['confidence'] - analysis['confidence']) > 0.1):
            return True
            
        return False

    def _save_signal_history(self):
        filename = os.path.join(
            self.signals_dir, 
            f"signals_{self.product_id}_{datetime.now(UTC).strftime('%Y%m%d')}.json"
        )
        
        with open(filename, 'w') as f:
            json.dump(self.signal_history, f, indent=2)

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments (reuse from market_analyzer.py)
    args = parse_arguments()
    
    if args.list_products or args.list_granularities:
        list_options()
        return
    
    # Create and run the monitor
    monitor = ContinuousMarketMonitor(
        product_id=args.product_id,
        candle_interval=args.granularity
    )
    
    # Run with appropriate interval
    interval_seconds = 60  # Default to 1 minute
    if args.granularity != 'ONE_MINUTE':
        granularity_map = {
            'FIVE_MINUTE': 300,
            'FIFTEEN_MINUTE': 900,
            'THIRTY_MINUTE': 1800,
            'ONE_HOUR': 3600,
            'TWO_HOUR': 7200,
            'SIX_HOUR': 21600,
            'ONE_DAY': 86400
        }
        interval_seconds = granularity_map.get(args.granularity, 60)
    
    monitor.run(interval_seconds=interval_seconds)

if __name__ == "__main__":
    main() 