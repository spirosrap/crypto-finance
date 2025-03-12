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
            
            # Get bias category and strength from analysis if available
            bias_category = ""
            bias_strength = 0.0
            if 'market_bias' in analysis:
                bias_parts = analysis['market_bias'].split()
                if len(bias_parts) >= 2:
                    bias_category = bias_parts[0]
                    # Extract strength value from format like "Strength: 0.75"
                    if 'Strength:' in analysis['market_bias']:
                        try:
                            strength_str = analysis['market_bias'].split('Strength:')[1].strip()
                            if '(' in strength_str and ')' in strength_str:
                                strength_str = strength_str.strip('()')
                            bias_strength = float(strength_str)
                        except (ValueError, IndexError):
                            bias_strength = 0.0

            # Enhanced market visualization with rich formatting
            print("\n" + "="*70)
            print(f"📊 MARKET ANALYSIS: {self.product_id} @ {self.candle_interval} 📊")
            print("="*70)
            
            # Market Overview Section with timestamp and price information
            print(f"\n🕒 {current_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"💲 ${analysis['current_price']:.4f}")
            
            # Signal Strength Visualization
            signal_icon = "🔴" if analysis['signal'] in ["SELL", "STRONG_SELL"] else "🟢" if analysis['signal'] in ["BUY", "STRONG_BUY"] else "⚪"
            confidence_bars = "█" * int(analysis['confidence']*10) + "░" * (10 - int(analysis['confidence']*10))
            
            print(f"\n📡 SIGNAL: {signal_icon} {analysis['signal']} ({analysis['position']})")
            print(f"🎯 CONFIDENCE: {confidence_bars} {analysis['confidence']*100:.1f}%")
            
            # Market Conditions
            market_emoji = "🌊" if "Ranging" in analysis['market_condition'] else "📈" if "Uptrend" in analysis['market_condition'] else "📉" if "Downtrend" in analysis['market_condition'] else "🔄"
            print(f"🔍 MARKET STATE: {market_emoji} {analysis['market_condition']}")
            
            # Market Bias
            if 'market_bias' in analysis:
                bias_emoji = "📈" if "Bullish" in analysis['market_bias'] else "📉" if "Bearish" in analysis['market_bias'] else "↔️"
                print(f"⚖️  BIAS: {bias_emoji} {analysis['market_bias']}")
            
            # Key Levels Section
            print("\n" + "-"*70)
            print("🏔️  KEY PRICE LEVELS")
            print("-"*70)
            
            # Get key levels from indicators if available
            if 'indicators' in analysis:
                indicators = analysis['indicators']
                current_price = analysis['current_price']
                
                # Structure to hold all price levels for sorting
                price_levels = []
                
                # Add Bollinger Bands
                if all(k in indicators for k in ['bollinger_upper', 'bollinger_lower', 'bollinger_middle']):
                    price_levels.append(("BBAND UPPER  ", indicators['bollinger_upper'], current_price < indicators['bollinger_upper']))
                    price_levels.append(("BBAND MIDDLE ", indicators['bollinger_middle'], True))
                    price_levels.append(("BBAND LOWER  ", indicators['bollinger_lower'], current_price > indicators['bollinger_lower']))
                
                # Add support/resistance
                if 'key_levels' in analysis:
                    if 'resistance' in analysis['key_levels']:
                        price_levels.append(("RESISTANCE   ", analysis['key_levels']['resistance'], current_price < analysis['key_levels']['resistance']))
                    if 'support' in analysis['key_levels']:
                        price_levels.append(("SUPPORT      ", analysis['key_levels']['support'], current_price > analysis['key_levels']['support']))
                
                # Add moving averages
                if 'ema_20' in indicators:
                    price_levels.append(("EMA 20       ", indicators['ema_20'], True))
                if 'ema_50' in indicators:
                    price_levels.append(("EMA 50       ", indicators['ema_50'], True))
                if 'ema_200' in indicators:
                    price_levels.append(("EMA 200      ", indicators['ema_200'], True))
                
                # Sort price levels from highest to lowest
                price_levels.sort(key=lambda x: x[1], reverse=True)
                
                # Find where current price fits in the sorted list
                current_price_inserted = False
                for i, (name, price, _) in enumerate(price_levels):
                    if not current_price_inserted and current_price > price:
                        # Insert current price marker
                        print(f"           --> CURRENT PRICE: ${current_price:.4f} <--")
                        current_price_inserted = True
                    
                    # Display price level with distance percentage
                    distance_pct = abs(price - current_price) / current_price * 100
                    trend_arrow = "🔼" if price > current_price else "🔽"
                    print(f"{trend_arrow} {name}: ${price:.4f} ({distance_pct:.2f}% away)")
                
                # If price is below all levels, add it at the end
                if not current_price_inserted:
                    print(f"           --> CURRENT PRICE: ${current_price:.4f} <--")
            
            # Technical Indicators Section
            print("\n" + "-"*70)
            print("📊 TECHNICAL INDICATORS")
            print("-"*70)
            
            if 'indicators' in analysis:
                # Organize indicators in columns
                col_width = 25
                
                # RSI with colored status
                rsi_value = analysis['indicators'].get('rsi', 0)
                if rsi_value > 70:
                    rsi_status = "OVERBOUGHT 🔴"
                elif rsi_value < 30:
                    rsi_status = "OVERSOLD 🟢"
                else:
                    rsi_status = "NEUTRAL ⚪"
                
                # MACD status
                macd = analysis['indicators'].get('macd', 0)
                macd_signal = analysis['indicators'].get('macd_signal', 0)
                if macd > macd_signal:
                    macd_status = "BULLISH 🟢"
                else:
                    macd_status = "BEARISH 🔴"
                
                # Stochastic status
                stoch_k = analysis['indicators'].get('stoch_k', 50)
                stoch_d = analysis['indicators'].get('stoch_d', 50)
                if stoch_k > stoch_d:
                    stoch_status = "BULLISH 🟢"
                else:
                    stoch_status = "BEARISH 🔴"
                
                # Format indicators in columns
                print(f"RSI:      {rsi_value:.2f} ({rsi_status})".ljust(col_width) + 
                      f"ADX:      {analysis['indicators'].get('adx', 0):.2f}")
                print(f"MACD:     {macd:.4f} ({macd_status})".ljust(col_width) + 
                      f"MACD Signal: {macd_signal:.4f}")
                print(f"Stoch K:  {stoch_k:.2f}".ljust(col_width) + 
                      f"Stoch D:    {stoch_d:.2f} ({stoch_status})")
                
                # Volume analysis
                vol_change = analysis.get('volume_analysis', {}).get('volume_change', 0)
                vol_trend = analysis.get('volume_analysis', {}).get('volume_trend', 'Neutral')
                vol_emoji = "🟢" if vol_trend == "Increasing" else "🔴" if vol_trend == "Decreasing" else "⚪"
                
                print(f"Volume:   {vol_trend} {vol_emoji} ({vol_change:.2f}%)".ljust(col_width) + 
                      f"ATR:        {analysis['risk_metrics'].get('atr', 0):.4f}")
            
            # Add scalping-specific information
            if 'rejection_event' in analysis and analysis['rejection_event']:
                rejection = analysis['rejection_event']
                
                print("\n" + "-"*70)
                print(f"🎯 SCALPING SETUP: {rejection['type'].upper()}")
                print("-"*70)
                
                # Format this as a table
                print(f"Level:      ${rejection['price_level']:.4f}")
                print(f"Stop Loss:  ${(rejection['price_level'] + (analysis['risk_metrics']['atr'] * 0.5)):.4f}")
                print(f"Target:     ${(rejection['price_level'] + (analysis['risk_metrics']['atr'] * 0.75)):.4f}")
                print(f"Volume:     {'✅ Confirmed' if rejection['volume_confirmation'] else '❌ Not Confirmed'}")
                print(f"Candles:    {rejection['confirming_candles']} confirming")
            
            # Trading Opportunities Section
            print("\n" + "-"*70)
            print("💰 TRADING OPPORTUNITIES")
            print("-"*70)
            
            opportunities_found = False
            
            # RSI-based opportunities
            if 'indicators' in analysis and 'rsi' in analysis['indicators']:
                rsi = analysis['indicators']['rsi']
                if rsi <= 30:
                    opportunities_found = True
                    print("🟢 Oversold Bounce (LONG):")
                    print(f"  Entry:  ${analysis['current_price']:.4f}")
                    print(f"  Stop:   ${(analysis['current_price'] - analysis['risk_metrics']['atr'] * 0.5):.4f}")
                    print(f"  Target: ${(analysis['current_price'] + analysis['risk_metrics']['atr'] * 0.75):.4f}")
                    print(f"  R/R:    1.5")
                elif rsi >= 70:
                    opportunities_found = True
                    print("🔴 Overbought Reversal (SHORT):")
                    print(f"  Entry:  ${analysis['current_price']:.4f}")
                    print(f"  Stop:   ${(analysis['current_price'] + analysis['risk_metrics']['atr'] * 0.5):.4f}")
                    print(f"  Target: ${(analysis['current_price'] - analysis['risk_metrics']['atr'] * 0.75):.4f}")
                    print(f"  R/R:    1.5")
            
            # Bollinger Bands opportunities
            if all(k in analysis.get('indicators', {}) for k in ['bollinger_upper', 'bollinger_lower']):
                if analysis['current_price'] <= analysis['indicators']['bollinger_lower']:
                    if opportunities_found:
                        print("")  # Add space between opportunities
                    opportunities_found = True
                    print("🟢 Bollinger Band Bounce (LONG):")
                    print(f"  Entry:  ${analysis['current_price']:.4f}")
                    print(f"  Stop:   ${(analysis['current_price'] - analysis['risk_metrics']['atr'] * 0.5):.4f}")
                    print(f"  Target: ${(analysis['current_price'] + analysis['risk_metrics']['atr']):.4f}")
                    print(f"  R/R:    2.0")
                elif analysis['current_price'] >= analysis['indicators']['bollinger_upper']:
                    if opportunities_found:
                        print("")  # Add space between opportunities
                    opportunities_found = True
                    print("🔴 Bollinger Band Reversal (SHORT):")
                    print(f"  Entry:  ${analysis['current_price']:.4f}")
                    print(f"  Stop:   ${(analysis['current_price'] + analysis['risk_metrics']['atr'] * 0.5):.4f}")
                    print(f"  Target: ${(analysis['current_price'] - analysis['risk_metrics']['atr']):.4f}")
                    print(f"  R/R:    2.0")
            
            if not opportunities_found:
                print("No high-probability trading opportunities currently detected.")
            
            # Risk Management Section
            print("\n" + "-"*70)
            print("⚠️  RISK MANAGEMENT")
            print("-"*70)
            
            print(f"Max Position:   1% of account")
            print(f"Min Volume:     >{analysis['risk_metrics'].get('required_volume', 0):.2f}")
            print(f"Max Hold Time:  15 minutes")
            print(f"Order Type:     Market Orders")
            
            # Pattern Analysis if available
            if 'patterns' in analysis and analysis['patterns'].get('type') != 'None':
                print("\n" + "-"*70)
                print("📐 CHART PATTERN ANALYSIS")
                print("-"*70)
                
                pattern_type = analysis['patterns'].get('type', 'None')
                pattern_confidence = analysis['patterns'].get('confidence', 0)
                pattern_target = analysis['patterns'].get('target', 0)
                
                print(f"Pattern:   {pattern_type}")
                print(f"Confidence: {pattern_confidence:.2f}")
                if pattern_target > 0:
                    print(f"Target:    ${pattern_target:.4f}")
            
            # Bottom border
            print("\n" + "="*70)
            
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