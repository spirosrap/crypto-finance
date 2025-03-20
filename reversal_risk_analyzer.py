import logging
import argparse
import numpy as np
import talib
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, UTC
from enum import Enum
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from config import API_KEY, API_SECRET
import time

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)
# Set higher level for our module to reduce RSI/MACD warning logs
logger.setLevel(logging.ERROR)


class RiskLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    EXTREME = "Extreme"


class ReversalRiskAnalyzer:
    """Class to analyze the risk of a market reversal."""

    def __init__(self, product_id: str = 'BTC-USDC', candle_interval: str = 'ONE_HOUR'):
        """
        Initialize the reversal risk analyzer.
        
        Args:
            product_id (str): The product ID to analyze.
            candle_interval (str): The interval for candles.
        """
        self.product_id = product_id
        self.candle_interval = candle_interval
        
        # Initialize services
        self.coinbase_service = CoinbaseService(API_KEY, API_SECRET)
        self.technical_analysis = TechnicalAnalysis(
            self.coinbase_service, 
            candle_interval=candle_interval, 
            product_id=product_id
        )
        
        # Set thresholds for different risk factors
        self.volume_spike_threshold = 2.0  # Volume spike is 2x average
        self.overextended_threshold = 0.15  # 15% away from moving average
        self.divergence_threshold = 0.2  # 20% difference for divergence
        
    def compute_rsi(self, prices: List[float], period: int = 14) -> float:
        """
        Compute RSI directly using talib.
        
        Args:
            prices (List[float]): List of prices.
            period (int): RSI period.
            
        Returns:
            float: RSI value.
        """
        if len(prices) < period + 1:
            return 50  # Return neutral value if not enough data
            
        try:
            prices_array = np.array(prices, dtype=float)
            rsi = talib.RSI(prices_array, timeperiod=period)
            return rsi[-1]
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50
            
    def compute_macd(self, prices: List[float], fast_period=12, slow_period=26, signal_period=9) -> Tuple[float, float, float]:
        """
        Compute MACD directly using talib.
        
        Args:
            prices (List[float]): List of prices.
            fast_period (int): Fast EMA period.
            slow_period (int): Slow EMA period.
            signal_period (int): Signal line period.
            
        Returns:
            Tuple[float, float, float]: (MACD line, signal line, histogram).
        """
        if len(prices) < slow_period + signal_period:
            return 0, 0, 0
            
        try:
            prices_array = np.array(prices, dtype=float)
            macd_line, signal_line, histogram = talib.MACD(
                prices_array,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )
            return macd_line[-1], signal_line[-1], histogram[-1]
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return 0, 0, 0
            
    def check_price_momentum_divergence(self, candles: List[Dict]) -> Tuple[bool, float, str]:
        """
        Check for divergence between price and momentum indicators.
        
        Args:
            candles (List[Dict]): List of candle data.
            
        Returns:
            Tuple[bool, float, str]: (has_divergence, divergence_strength, description)
        """
        # Make sure we have enough data
        if len(candles) < 50:
            return False, 0.0, "Insufficient data for divergence analysis"
            
        # Get price data - use at least 50 candles for analysis
        prices = [float(c['close']) for c in candles[-50:]]
        
        # Calculate RSI for the last 30 periods
        rsi_values = []
        # We need at least 14 periods of prices before our first RSI value
        for i in range(len(prices) - 14):
            if i + 14 <= len(prices):
                price_window = prices[i:i+14]
                if len(price_window) == 14:  # Ensure we have exactly 14 periods
                    rsi = self.compute_rsi(price_window)
                    rsi_values.append(rsi)
        
        # Calculate MACD for the last 30 periods
        macd_line = []
        signal_line = []
        # We need at least 26+9=35 periods for MACD calculation
        for i in range(len(prices) - 35):
            if i + 35 <= len(prices):
                price_window = prices[i:i+35]
                if len(price_window) == 35:
                    macd, signal, _ = self.compute_macd(price_window)
                    macd_line.append(macd)
                    signal_line.append(signal)
        
        # Check for divergence only if we have enough data points
        if len(rsi_values) < 10 or len(macd_line) < 10:
            return False, 0.0, "Insufficient indicator history for divergence analysis"
        
        # Check for price making higher highs but RSI making lower highs (bearish divergence)
        bearish_rsi_divergence = False
        price_peaks = self._find_peaks(prices[-min(len(prices), 30):])
        rsi_peaks = self._find_peaks(rsi_values[-min(len(rsi_values), 30):])
        
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            # If price is making higher highs but RSI is making lower highs
            if (prices[price_peaks[-1]] > prices[price_peaks[-2]] and 
                rsi_values[rsi_peaks[-1]] < rsi_values[rsi_peaks[-2]]):
                bearish_rsi_divergence = True
        
        # Check for price making lower lows but RSI making higher lows (bullish divergence)
        bullish_rsi_divergence = False
        price_troughs = self._find_troughs(prices[-min(len(prices), 30):])
        rsi_troughs = self._find_troughs(rsi_values[-min(len(rsi_values), 30):])
        
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            # If price is making lower lows but RSI is making higher lows
            if (prices[price_troughs[-1]] < prices[price_troughs[-2]] and 
                rsi_values[rsi_troughs[-1]] > rsi_values[rsi_troughs[-2]]):
                bullish_rsi_divergence = True
        
        # Check for divergence between price and MACD
        bearish_macd_divergence = False
        bullish_macd_divergence = False
        
        if len(macd_line) >= 10:
            macd_peaks = self._find_peaks(macd_line[-10:])
            macd_troughs = self._find_troughs(macd_line[-10:])
            
            if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
                # If price is making higher highs but MACD is making lower highs
                if (prices[price_peaks[-1]] > prices[price_peaks[-2]] and 
                    macd_line[macd_peaks[-1]] < macd_line[macd_peaks[-2]]):
                    bearish_macd_divergence = True
            
            if len(price_troughs) >= 2 and len(macd_troughs) >= 2:
                # If price is making lower lows but MACD is making higher lows
                if (prices[price_troughs[-1]] < prices[price_troughs[-2]] and 
                    macd_line[macd_troughs[-1]] > macd_line[macd_troughs[-2]]):
                    bullish_macd_divergence = True
        
        # Determine if there's a divergence and its strength
        has_divergence = bearish_rsi_divergence or bullish_rsi_divergence or bearish_macd_divergence or bullish_macd_divergence
        
        divergence_strength = 0.0
        direction = ""
        description = ""
        
        if bearish_rsi_divergence:
            divergence_strength += 0.3
            direction = "bearish"
            description = "Bearish RSI divergence detected: price making higher highs but RSI making lower highs"
        
        if bearish_macd_divergence:
            divergence_strength += 0.4
            direction = "bearish"
            if description:
                description += " and bearish MACD divergence"
            else:
                description = "Bearish MACD divergence detected: price making higher highs but MACD making lower highs"
        
        if bullish_rsi_divergence:
            divergence_strength += 0.3
            direction = "bullish"
            description = "Bullish RSI divergence detected: price making lower lows but RSI making higher lows"
        
        if bullish_macd_divergence:
            divergence_strength += 0.4
            direction = "bullish"
            if description:
                description += " and bullish MACD divergence"
            else:
                description = "Bullish MACD divergence detected: price making lower lows but MACD making higher lows"
        
        return has_divergence, divergence_strength, f"{direction}: {description}" if has_divergence else "No divergence detected"

    def _find_peaks(self, data: List[float], window: int = 2) -> List[int]:
        """Find peaks in a data series."""
        if len(data) < window * 2 + 1:
            return []
            
        peaks = []
        for i in range(window, len(data) - window):
            if all(data[i] > data[i - j] for j in range(1, window + 1)) and \
               all(data[i] > data[i + j] for j in range(1, window + 1)):
                peaks.append(i)
        return peaks

    def _find_troughs(self, data: List[float], window: int = 2) -> List[int]:
        """Find troughs in a data series."""
        if len(data) < window * 2 + 1:
            return []
            
        troughs = []
        for i in range(window, len(data) - window):
            if all(data[i] < data[i - j] for j in range(1, window + 1)) and \
               all(data[i] < data[i + j] for j in range(1, window + 1)):
                troughs.append(i)
        return troughs
        
    def check_volume_spike(self, candles: List[Dict]) -> Tuple[bool, float, str]:
        """
        Check for a spike in volume at key price levels.
        
        Args:
            candles (List[Dict]): List of candle data.
            
        Returns:
            Tuple[bool, float, str]: (has_volume_spike, spike_strength, description)
        """
        if len(candles) < 20:
            return False, 0.0, "Insufficient data for volume analysis"
        
        # Extract volume data
        volumes = [float(c['volume']) for c in candles[-20:]]
        prices = [float(c['close']) for c in candles[-20:]]
        
        # Calculate average volume
        avg_volume = np.mean(volumes[:-1])  # Average excluding the latest candle
        latest_volume = volumes[-1]
        
        # Check if latest volume is a spike
        volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0
        is_spike = volume_ratio > self.volume_spike_threshold
        
        # Identify if this is at a key level
        # Key levels: support/resistance, moving averages, etc.
        
        # Check for price near moving averages
        ma_20 = np.mean(prices[:-1])
        ma_50 = np.mean([float(c['close']) for c in candles[-min(50, len(candles)):]])
        
        # Check if price is near a moving average (within 2%)
        near_ma_20 = abs(prices[-1] - ma_20) / ma_20 < 0.02
        near_ma_50 = abs(prices[-1] - ma_50) / ma_50 < 0.02
        at_key_level = near_ma_20 or near_ma_50
        
        # Check for price near local highs/lows
        local_high = max(prices[:-1])
        local_low = min(prices[:-1])
        near_high = abs(prices[-1] - local_high) / local_high < 0.02
        near_low = abs(prices[-1] - local_low) / local_low < 0.02
        at_key_level = at_key_level or near_high or near_low
        
        # Calculate the strength of the signal
        spike_strength = 0.0
        if is_spike:
            spike_strength = min((volume_ratio - self.volume_spike_threshold) / 3, 1.0)
            if at_key_level:
                spike_strength *= 1.5  # Increase significance if at key level
        
        # Generate description
        description = ""
        if is_spike:
            description = f"Volume spike detected ({volume_ratio:.1f}x average)"
            if near_ma_20:
                description += " near 20-period MA"
            elif near_ma_50:
                description += " near 50-period MA"
            elif near_high:
                description += " near local high"
            elif near_low:
                description += " near local low"
        else:
            description = "No significant volume spike detected"
        
        return is_spike, spike_strength, description

    def check_market_overextension(self, candles: List[Dict]) -> Tuple[bool, float, str]:
        """
        Check if the market is overextended.
        
        Args:
            candles (List[Dict]): List of candle data.
            
        Returns:
            Tuple[bool, float, str]: (is_overextended, overextension_strength, description)
        """
        if len(candles) < 50:
            return False, 0.0, "Insufficient data for overextension analysis"
        
        # Extract price data
        prices = [float(c['close']) for c in candles]
        current_price = prices[-1]
        
        # Calculate moving averages
        ma_20 = np.mean(prices[-20:])
        ma_50 = np.mean(prices[-50:])
        
        # Calculate RSI to check if market is overbought/oversold
        # Ensure we have at least 14 periods for RSI calculation
        if len(prices) >= 14:
            rsi = self.compute_rsi(prices[-14:])
        else:
            rsi = 50  # Neutral value if not enough data
        
        # Calculate deviation from moving averages
        deviation_20 = (current_price - ma_20) / ma_20
        deviation_50 = (current_price - ma_50) / ma_50
        
        # Determine if market is overextended
        is_overextended_up = deviation_20 > self.overextended_threshold and deviation_50 > self.overextended_threshold and rsi > 70
        is_overextended_down = deviation_20 < -self.overextended_threshold and deviation_50 < -self.overextended_threshold and rsi < 30
        is_overextended = is_overextended_up or is_overextended_down
        
        # Calculate strength of overextension
        overextension_strength = 0.0
        description = ""
        
        if is_overextended_up:
            # Calculate overextension strength based on how far above threshold
            overextension_strength = min(max(deviation_20, deviation_50) / (self.overextended_threshold * 2), 1.0)
            description = f"Market appears overextended upward: {deviation_20*100:.1f}% above 20MA, {deviation_50*100:.1f}% above 50MA, RSI: {rsi:.1f}"
        elif is_overextended_down:
            # Calculate overextension strength based on how far below threshold
            overextension_strength = min(max(abs(deviation_20), abs(deviation_50)) / (self.overextended_threshold * 2), 1.0)
            description = f"Market appears overextended downward: {-deviation_20*100:.1f}% below 20MA, {-deviation_50*100:.1f}% below 50MA, RSI: {rsi:.1f}"
        else:
            description = "Market is not overextended"
        
        return is_overextended, overextension_strength, description

    def analyze_reversal_risk(self) -> Dict:
        """
        Analyze the risk of a market reversal.
        
        Returns:
            Dict: Analysis results including risk level and factors.
        """
        try:
            # Get historical data
            historical_data = self.coinbase_service.historical_data
            
            # Calculate duration based on candle interval - fetch more historical data
            if self.candle_interval == 'ONE_MINUTE':
                duration = timedelta(hours=24)  # 24 hours
            elif self.candle_interval == 'FIVE_MINUTE':
                duration = timedelta(days=5)  # 5 days
            elif self.candle_interval == 'FIFTEEN_MINUTE':
                duration = timedelta(days=10)  # 10 days
            elif self.candle_interval == 'ONE_HOUR':
                duration = timedelta(days=21)  # 3 weeks
            elif self.candle_interval == 'SIX_HOUR':
                duration = timedelta(days=90)  # 3 months
            else:  # Default to ONE_DAY
                duration = timedelta(days=180)  # 6 months
            
            end_time = datetime.now(UTC)
            start_time = end_time - duration
            
            # Get candles using get_historical_data instead of get_product_candles
            candles = historical_data.get_historical_data(
                self.product_id,
                start_date=start_time,
                end_date=end_time,
                granularity=self.candle_interval
            )
            
            if len(candles) < 50:
                logger.warning(f"Got only {len(candles)} candles, need at least 50 for analysis")
                return {
                    "error": f"Insufficient data for analysis (got {len(candles)} candles, need at least 50)",
                    "reversal_risk": RiskLevel.LOW.value,
                    "score": 0.0
                }
            
            # Check for various reversal indicators
            has_divergence, divergence_strength, divergence_desc = self.check_price_momentum_divergence(candles)
            has_volume_spike, volume_spike_strength, volume_desc = self.check_volume_spike(candles)
            is_overextended, overextension_strength, overextension_desc = self.check_market_overextension(candles)
            
            # Calculate overall reversal risk score
            # Weights for different factors
            divergence_weight = 0.4
            volume_spike_weight = 0.3
            overextension_weight = 0.3
            
            # Calculate weighted risk score
            risk_score = (
                divergence_strength * divergence_weight +
                volume_spike_strength * volume_spike_weight +
                overextension_strength * overextension_weight
            )
            
            # Determine risk level
            risk_level = RiskLevel.LOW
            if risk_score > 0.7:
                risk_level = RiskLevel.EXTREME
            elif risk_score > 0.5:
                risk_level = RiskLevel.HIGH
            elif risk_score > 0.3:
                risk_level = RiskLevel.MEDIUM
            
            # Get market direction
            ma_20 = np.mean([float(c['close']) for c in candles[-20:]])
            ma_50 = np.mean([float(c['close']) for c in candles[-50:]])
            current_price = float(candles[-1]['close'])
            
            if current_price > ma_20 and ma_20 > ma_50:
                market_direction = "uptrend"
            elif current_price < ma_20 and ma_20 < ma_50:
                market_direction = "downtrend"
            else:
                market_direction = "sideways"
            
            # Determine if this is a potential reversal from current trend
            potential_reversal_direction = "none"
            if market_direction == "uptrend" and (has_divergence and "bearish" in divergence_desc):
                potential_reversal_direction = "bearish reversal from uptrend"
            elif market_direction == "downtrend" and (has_divergence and "bullish" in divergence_desc):
                potential_reversal_direction = "bullish reversal from downtrend"
            
            return {
                "reversal_risk": risk_level.value,
                "risk_score": risk_score,
                "factors": {
                    "divergence": {
                        "detected": has_divergence,
                        "strength": divergence_strength,
                        "description": divergence_desc
                    },
                    "volume_spike": {
                        "detected": has_volume_spike,
                        "strength": volume_spike_strength,
                        "description": volume_desc
                    },
                    "overextension": {
                        "detected": is_overextended,
                        "strength": overextension_strength,
                        "description": overextension_desc
                    }
                },
                "market_direction": market_direction,
                "potential_reversal": potential_reversal_direction,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        except Exception as e:
            logger.error(f"Error analyzing reversal risk: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": f"Analysis error: {str(e)}",
                "reversal_risk": RiskLevel.LOW.value,
                "score": 0.0
            }


def main():
    parser = argparse.ArgumentParser(description='Analyze market reversal risk')
    
    parser.add_argument('--product_id', type=str, default='BTC-USDC',
                        choices=['BTC-USDC', 'ETH-USDC', 'SOL-USDC', 'DOGE-USDC', 'XRP-USDC'],
                        help='Trading product (default: BTC-USDC)')
    
    parser.add_argument('--granularity', type=str, default='ONE_HOUR',
                        choices=['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'ONE_HOUR', 'SIX_HOUR', 'ONE_DAY'],
                        help='Candle interval (default: ONE_HOUR)')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output with more detailed logging')
    
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.INFO)
    
    # Create reversal risk analyzer
    analyzer = ReversalRiskAnalyzer(
        product_id=args.product_id,
        candle_interval=args.granularity
    )
    
    # Analyze reversal risk
    results = analyzer.analyze_reversal_risk()
    
    # Display results
    print(f"\n=== Reversal Risk Analysis for {args.product_id} ({args.granularity}) ===")
    print(f"Timestamp: {results.get('timestamp', 'N/A')}")
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
        
    print(f"Overall Risk Level: {results.get('reversal_risk', 'Unknown')}")
    print(f"Risk Score: {results.get('risk_score', 0.0):.2f}")
    print(f"Market Direction: {results.get('market_direction', 'Unknown')}")
    
    if results.get('potential_reversal') and results.get('potential_reversal') != 'none':
        print(f"Potential Reversal: {results.get('potential_reversal', 'None')}")
    
    print("\nRisk Factors:")
    factors = results.get('factors', {})
    
    # Divergence
    divergence = factors.get('divergence', {})
    print(f"\n1. Divergence:")
    print(f"   Detected: {divergence.get('detected', False)}")
    print(f"   Strength: {divergence.get('strength', 0.0):.2f}")
    print(f"   Description: {divergence.get('description', 'N/A')}")
    
    # Volume Spike
    volume = factors.get('volume_spike', {})
    print(f"\n2. Volume Spike:")
    print(f"   Detected: {volume.get('detected', False)}")
    print(f"   Strength: {volume.get('strength', 0.0):.2f}")
    print(f"   Description: {volume.get('description', 'N/A')}")
    
    # Overextension
    overextension = factors.get('overextension', {})
    print(f"\n3. Market Overextension:")
    print(f"   Detected: {overextension.get('detected', False)}")
    print(f"   Strength: {overextension.get('strength', 0.0):.2f}")
    print(f"   Description: {overextension.get('description', 'N/A')}")
    
    # Provide recommendation
    print("\nRecommendation:")
    risk_level = results.get('reversal_risk', 'Low')
    market_direction = results.get('market_direction', 'Unknown')
    potential_reversal = results.get('potential_reversal', 'none')
    
    if risk_level == 'High' or risk_level == 'Extreme':
        if "bearish" in str(potential_reversal):
            print("⚠️ High risk of bearish reversal. Consider taking profits or reducing position size.")
        elif "bullish" in str(potential_reversal):
            print("⚠️ High risk of bullish reversal. Consider closing short positions or preparing for long entry.")
        else:
            print("⚠️ High reversal risk detected. Exercise caution when entering new positions.")
    elif risk_level == 'Medium':
        print("⚠️ Moderate reversal risk. Be aware of potential trend change and adjust stop losses accordingly.")
    else:
        print("✅ Low reversal risk. Current trend likely to continue, but always use proper risk management.")


if __name__ == "__main__":
    main() 