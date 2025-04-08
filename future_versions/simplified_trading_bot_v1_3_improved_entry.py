# Simplified Trading Bot v1.3
# Single coin (BTC-USDC), single timeframe (5-min), improved entry logic
# Enhanced entry selectivity under fakeout-prone or trending regimes

from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np
from config import API_KEY_PERPS, API_SECRET_PERPS
import logging
import argparse
import subprocess
import unittest

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress logs from other modules
logging.getLogger('technicalanalysis').setLevel(logging.WARNING)
logging.getLogger('historicaldata').setLevel(logging.WARNING)
logging.getLogger('bitcoinpredictionmodel').setLevel(logging.WARNING)
logging.getLogger('ml_model').setLevel(logging.WARNING)

# Parameters
GRANULARITY = "FIVE_MINUTE"
RSI_THRESHOLD = 30
RSI_CONFIRMATION_THRESHOLD = 35  # v1.3: New threshold for confirmation bar
VOLUME_LOOKBACK = 20
TP_PERCENT = 0.015
SL_PERCENT = 0.007
LEVERAGE = 5  # Conservative leverage
POSITION_SIZE_USD = 100  # Position size in USD

# Regime classification thresholds (v1.3: Updated for better regime filtering)
STRONG_TREND_ATR_THRESHOLD = 1.0
MODERATE_ATR_THRESHOLD = 0.5
CHOP_ZONE_ATR_THRESHOLD = 0.5  # v1.3: New threshold for chop zone

# v1.3: New parameters for entry filters
REDUCE_POSITION_BELOW_EMA = True  # Set to False to skip trades entirely when price < EMA(50)
USE_RSI_CONFIRMATION = True  # Set to False to skip RSI confirmation for experimentation

def parse_args():
    parser = argparse.ArgumentParser(description='Simplified Trading Bot v1.3')
    parser.add_argument('--product_id', type=str, default='BTC-USDC',
                      help='Product ID to trade (e.g., BTC-USDC)')
    parser.add_argument('--margin', type=float, default=100,
                      help='Position size in USD')
    parser.add_argument('--leverage', type=int, default=5,
                      help='Trading leverage')
    parser.add_argument('--test', action='store_true',
                      help='Run unit tests')
    parser.add_argument('--no-rsi-confirmation', action='store_true',
                      help='Disable RSI confirmation for experimentation')
    return parser.parse_args()

def get_perp_product(product_id):
    """Convert spot product ID to perpetual futures product ID"""
    perp_map = {
        'BTC-USDC': 'BTC-PERP-INTX',
        'ETH-USDC': 'ETH-PERP-INTX',
        'DOGE-USDC': 'DOGE-PERP-INTX',
        'SOL-USDC': 'SOL-PERP-INTX',
        'SHIB-USDC': '1000SHIB-PERP-INTX'
    }
    return perp_map.get(product_id, 'BTC-PERP-INTX')

def get_price_precision(product_id):
    """Get price precision for a product"""
    precision_map = {
        'BTC-PERP-INTX': 1,      # $1 precision for BTC
        'ETH-PERP-INTX': 0.1,    # $0.1 precision for ETH
        'DOGE-PERP-INTX': 0.0001, # $0.0001 precision for DOGE
        'SOL-PERP-INTX': 0.01,   # $0.01 precision for SOL
        '1000SHIB-PERP-INTX': 0.000001  # $0.000001 precision for SHIB
    }
    return precision_map.get(product_id, 1)

def fetch_candles(cb, product_id):
    # Default to last 8000 5-minute candles
    now = datetime.now(UTC)
    start = now - timedelta(minutes=5 * 8000)
    end = now
    
    raw_data = cb.historical_data.get_historical_data(product_id, start, end, GRANULARITY)
    df = pd.DataFrame(raw_data)
    
    # Convert string columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle timestamp - convert Unix timestamp to datetime
    if 'start' in df.columns:
        df['start'] = pd.to_datetime(pd.to_numeric(df['start']), unit='s', utc=True)        
        df.set_index('start', inplace=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df.set_index('timestamp', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
    
    # v1.3: Add additional columns for entry filters
    df['body_size'] = abs(df['close'] - df['open'])
    df['body_size_sma'] = df['body_size'].rolling(window=VOLUME_LOOKBACK).mean()
    df['volume_sma'] = df['volume'].rolling(window=VOLUME_LOOKBACK).mean()
    
    return df

def classify_market_regime(atr, entry_price):
    """
    Classify the market regime based on ATR percentage.
    Returns 'trending', 'moderate', or 'sideways'
    """
    atr_percentage = (atr / entry_price) * 100
    
    # v1.3: Updated regime classification thresholds
    if atr_percentage < 0.15:  # Lowered from 0.3
        return "sideways"
    elif atr_percentage < 0.4:  # Lowered from 0.7
        return "moderate"
    else:
        return "trending"

def check_entry_conditions(df: pd.DataFrame, rsi_values, ema_value, atr, product_id: str):
    """
    v1.3: New function to check all entry conditions
    Returns (signal, entry_price, position_size_multiplier)
    """
    try:
        # Get current and previous values
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Get current RSI and previous RSI
        if len(rsi_values) != 2:
            logger.error("Expected exactly 2 RSI values")
            return False, None, 1.0
            
        previous_rsi, current_rsi = rsi_values
        
        # v1.3: Check if we're in a chop zone (low volatility)
        if atr < CHOP_ZONE_ATR_THRESHOLD:
            logger.info(f"[SKIPPED] Trade signal detected but skipped due to chop zone (ATR: {atr:.4f})")
            return False, None, 1.0
        
        # v1.3: Check if RSI crossed below threshold in previous bar
        rsi_crossed_down = previous_rsi >= RSI_THRESHOLD and current_rsi < RSI_THRESHOLD
        
        # v1.3: Check if RSI is still below confirmation threshold
        # Only check if USE_RSI_CONFIRMATION is True
        rsi_confirmed = True  # Default to True if confirmation is disabled
        if USE_RSI_CONFIRMATION:
            rsi_confirmed = current_rsi < RSI_CONFIRMATION_THRESHOLD
        
        # v1.3: Check if price is above previous low (no new low)
        price_above_prev_low = current["close"] > previous["low"]
        
        # v1.3: Check if volume is above SMA - now conditional on regime
        # If ATR < 0.5: skip volume check, else: keep volume > SMA(volume, 20)
        volume_above_sma = True  # Default to True
        if atr >= 0.5:  # Only check volume if not in low volatility regime
            volume_above_sma = current["volume"] > current["volume_sma"]
        
        # v1.3: Check if body size is below 1.5 * SMA (loosened filter)
        # Changed from: body_below_sma = current["body_size"] < current["body_size_sma"]
        body_below_sma = current["body_size"] < 1.5 * current["body_size_sma"]
        
        # v1.3: Check EMA context - use the provided EMA value
        price_above_ema = current["close"] > ema_value
        
        # v1.3: Determine position size multiplier based on EMA context
        position_size_multiplier = 0.5 if not price_above_ema else 1.0
        
        # v1.3: Log entry conditions for debugging
        logger.info(f"Entry conditions: RSI={current_rsi:.2f} (prev={previous_rsi:.2f}), "
                    f"RSI crossed down={rsi_crossed_down}, "
                    f"RSI confirmed={rsi_confirmed} (confirmation {'enabled' if USE_RSI_CONFIRMATION else 'disabled'}), "
                    f"Price above prev low={price_above_prev_low}, "
                    f"Volume above SMA={volume_above_sma}, "
                    f"Body below 1.5*SMA={body_below_sma}, "
                    f"Price above EMA={price_above_ema} "
                    f"(Price: {current['close']:.2f}, EMA: {ema_value:.2f})")
        
        # v1.3: Check if we should skip trade entirely when price < EMA(50)
        if not price_above_ema and not REDUCE_POSITION_BELOW_EMA:
            logger.info("[SKIPPED] Trade signal detected but skipped due to price below EMA(50)")
            return False, None, 1.0
        
        # v1.3: New entry logic with detailed rejection logging
        if not rsi_crossed_down:
            logger.info(f"[REJECTED] RSI did not cross down below threshold (prev={previous_rsi:.2f}, curr={current_rsi:.2f})")
            return False, None, 1.0
            
        if not rsi_confirmed and USE_RSI_CONFIRMATION:
            logger.info(f"[REJECTED] RSI not confirmed below threshold (curr={current_rsi:.2f})")
            return False, None, 1.0
            
        if not price_above_prev_low:
            logger.info(f"[REJECTED] Price not above previous low (curr={current['close']:.2f}, prev_low={previous['low']:.2f})")
            return False, None, 1.0
            
        if not volume_above_sma and atr >= 0.5:
            logger.info(f"[REJECTED] Volume not above SMA (curr={current['volume']:.2f}, sma={current['volume_sma']:.2f})")
            return False, None, 1.0
            
        if not body_below_sma:
            logger.info(f"[REJECTED] Body size not below 1.5*SMA (curr={current['body_size']:.2f}, 1.5*sma={1.5*current['body_size_sma']:.2f})")
            return False, None, 1.0
        
        # All conditions passed
        logger.info(f"[ACCEPTED] All entry conditions met. RSI at entry: {current_rsi:.2f}")
        return True, current["close"], position_size_multiplier
        
    except Exception as e:
        logger.error(f"Error in check_entry_conditions: {str(e)}")
        return False, None, 1.0

def analyze(df: pd.DataFrame, ta: TechnicalAnalysis, product_id: str):
    # Convert DataFrame to list of dictionaries for the technical analysis methods
    candles = df.to_dict('records')
    
    try:
        # Calculate RSI for current candles
        current_rsi = ta.compute_rsi(product_id, candles, period=14)
        
        # Calculate RSI for previous candles (excluding the last one)
        if len(candles) > 1:
            previous_candles = candles[:-1]
            previous_rsi = ta.compute_rsi(product_id, previous_candles, period=14)
            rsi_values = [previous_rsi, current_rsi]
        else:
            logger.error("Not enough candles to calculate previous RSI")
            return False, None, 1.0
        
        # Calculate EMA - this returns a single float value
        ema_value = ta.get_moving_average(candles, period=50, ma_type='ema')
        
        # Calculate ATR for regime classification
        atr = ta.compute_atr(candles)
        if atr is None:
            logger.error("ATR computation failed")
            return False, None, 1.0
            
        regime = classify_market_regime(atr, df.iloc[-1]["close"])
        
        # Log regime information
        logger.info(f"Current market regime: {regime}")
        
        # v1.3: Use new entry condition checker with proper values
        signal, entry_price, position_size_multiplier = check_entry_conditions(
            df,
            rsi_values,  # Pass the RSI values array
            ema_value,   # Pass the single EMA value
            atr,
            product_id
        )
        
        if signal:
            logger.info(f"[SIGNAL] BUY {product_id} at {entry_price:.2f} with position size multiplier {position_size_multiplier}")
            return True, entry_price, position_size_multiplier
        
        return False, None, 1.0
        
    except Exception as e:
        logger.error(f"Error in analyze function: {str(e)}")
        return False, None, 1.0

def determine_tp_mode(entry_price: float, atr: float, price_precision: float = None) -> tuple[str, float]:
    """
    Determine take profit mode and price based on ATR volatility.
    Returns a tuple of (tp_mode, tp_price)
    """
    atr_percent = (atr / entry_price) * 100
    if atr_percent > 0.7:
        # High volatility → Use adaptive TP (2.5x ATR handles volatility better)
        tp_mode = "ADAPTIVE"
        tp_price = entry_price + (2.5 * atr)  # 2.5×ATR adaptive TP
    else:
        # Low volatility → Use fixed TP (market less likely to run, so %-based makes sense)
        tp_mode = "FIXED"
        tp_price = entry_price * (1 + TP_PERCENT)  # 1.5% fixed TP
    
    # Round the price if precision is provided
    if price_precision is not None:
        tp_price = round(tp_price, price_precision)
    
    return tp_mode, tp_price

def execute_trade(cb, entry_price: float, product_id: str, margin: float, leverage: int, position_size_multiplier: float = 1.0):
    """Execute the trade using trade_btc_perp.py functions"""
    try:
        # Convert to perpetual futures product ID
        perp_product = get_perp_product(product_id)
        price_precision = get_price_precision(perp_product)
        
        # Calculate ATR for volatility check
        candles = cb.historical_data.get_historical_data(product_id, datetime.now(UTC) - timedelta(minutes=5 * 100), datetime.now(UTC), GRANULARITY)
        ta = TechnicalAnalysis(cb)
        atr = ta.compute_atr(candles)
        
        # Determine TP mode and price using centralized function
        tp_mode, tp_price = determine_tp_mode(entry_price, atr, price_precision)
        
        # Fixed stop loss
        sl_price = round(entry_price * (1 - SL_PERCENT), price_precision)              
        
        # v1.3: Apply position size multiplier
        size_usd = margin * leverage * position_size_multiplier
        
        # Determine trading session based on current UTC time
        current_hour = datetime.now(UTC).hour
        if 0 <= current_hour < 9:
            session = "Asia"
        elif 9 <= current_hour < 17:
            session = "EU"
        else:
            session = "US"            
        
        # Calculate ATR percentage
        atr_percent = (atr / entry_price) * 100
        
        # Determine setup type based on entry conditions
        setup_type = "RSI Dip v1.3"  # Updated to indicate v1.3
        
        # Determine market regime
        regime = classify_market_regime(atr, entry_price)
        
        # Log trade setup information
        logger.info(f"TP Mode: {tp_mode}")
        logger.info(f"Take Profit: ${tp_price:.2f}")
        logger.info(f"Stop Loss: ${sl_price:.2f}")
        logger.info(f"Size: ${size_usd:.2f} (multiplier: {position_size_multiplier})")
        logger.info(f"Session: {session}")
        logger.info(f"ATR %: {atr_percent:.2f}")
        logger.info(f"Setup Type: {setup_type}")
        logger.info(f"Market Regime: {regime}")

        # Prepare command for trade_btc_perp.py
        cmd = [
            'python', 'trade_btc_perp.py',
            '--product', perp_product,
            '--side', 'BUY',
            '--size', str(size_usd),
            '--leverage', str(leverage),
            '--tp', str(tp_price),
            '--sl', str(sl_price),
            '--no-confirm'
        ]
        
        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error placing order: {result.stderr}")
            return False
            
        logger.info("Order placed successfully!")
        logger.info(f"Command output: {result.stdout}")

        # Log trade to automated_trades.csv
        try:
            # Read existing trades to get the next trade number
            trades_df = pd.read_csv('automated_trades.csv')
            next_trade_no = len(trades_df) + 1
            
            # Calculate R/R ratio
            rr_ratio = (tp_price - entry_price) / (entry_price - sl_price)
            
            # Create new trade entry with additional metrics
            new_trade = pd.DataFrame([{
                'No.': next_trade_no,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'SIDE': 'LONG',
                'ENTRY': entry_price,
                'Take Profit': tp_price,
                'Stop Loss': sl_price,
                'R/R Ratio': round(rr_ratio, 2),
                'Volatility Level': regime,  # Use regime instead of volatility level
                'Outcome': 'PENDING',
                'Outcome %': 0.0,
                'Leverage': f"{leverage}x",
                'Margin': margin,
                'Position Multiplier': position_size_multiplier,  # v1.3: Added position multiplier
                'Session': session,
                'TP Mode': tp_mode,
                'ATR %': round(atr_percent, 2),
                'Setup Type': setup_type,
                'MAE': 0.0,
                'MFE': 0.0,
                'Trend Regime': regime,  # Use the actual regime
                'Version': 'v1.3'  # v1.3: Added version tracking
            }])
            
            # Append new trade to CSV
            new_trade.to_csv('automated_trades.csv', mode='a', header=False, index=False)
            logger.info("Trade logged to automated_trades.csv")
            
        except Exception as e:
            logger.error(f"Error logging trade to automated_trades.csv: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False

def main():
    args = parse_args()
    
    # Set global parameters based on command line arguments
    global USE_RSI_CONFIRMATION
    if args.no_rsi_confirmation:
        USE_RSI_CONFIRMATION = False
        logger.info("RSI confirmation disabled for experimentation")
    
    # Initialize services
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    ta = TechnicalAnalysis(cb)
    
    try:
        # Fetch historical data
        df = fetch_candles(cb, args.product_id)
        
        # Live trading mode
        signal, entry, position_size_multiplier = analyze(df, ta, args.product_id)
        
        if signal:
            logger.info(f"[SIGNAL] BUY {args.product_id} at {entry:.2f}")
            if execute_trade(cb, entry, args.product_id, args.margin, args.leverage, position_size_multiplier):
                logger.info("Trade executed successfully!")
            else:
                logger.error("Failed to execute trade")
        else:
            logger.info("[NO SIGNAL] Conditions not met.")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

# v1.3: Unit tests for entry filters
class TestEntryFilters(unittest.TestCase):
    def setUp(self):
        # Create a mock DataFrame for testing
        self.df = pd.DataFrame({
            'open': [100, 100, 100, 100, 100],
            'high': [105, 105, 105, 105, 105],
            'low': [95, 95, 95, 95, 95],
            'close': [98, 97, 96, 95, 94],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'body_size': [2, 3, 4, 5, 6],
            'body_size_sma': [3, 3, 3, 3, 3],
            'volume_sma': [1000, 1000, 1000, 1000, 1000]
        })
        
        # Create mock RSI values
        self.rsi_values = [40, 35, 32, 28, 25]
        
        # Create mock EMA values
        self.ema_50 = [100, 100, 100, 100, 100]
        
        # Create mock ATR value
        self.atr = 1.0
        
        # Create mock product ID
        self.product_id = 'BTC-USDC'
    
    def test_rsi_cross_down(self):
        """Test RSI crossing down below threshold"""
        # Test when RSI crosses down
        signal, _, _ = check_entry_conditions(self.df, self.rsi_values, self.ema_50[-1], self.atr, self.product_id)
        self.assertTrue(signal)
        
        # Test when RSI doesn't cross down
        rsi_values_no_cross = [25, 24, 23, 22, 21]
        signal, _, _ = check_entry_conditions(self.df, rsi_values_no_cross, self.ema_50[-1], self.atr, self.product_id)
        self.assertFalse(signal)
    
    def test_price_above_prev_low(self):
        """Test price above previous low condition"""
        # Test when price is above previous low
        df = self.df.copy()
        df.loc[4, 'close'] = 96  # Current close above previous low (95)
        signal, _, _ = check_entry_conditions(df, self.rsi_values, self.ema_50[-1], self.atr, self.product_id)
        self.assertTrue(signal)
        
        # Test when price is below previous low
        df.loc[4, 'close'] = 94  # Current close below previous low (95)
        signal, _, _ = check_entry_conditions(df, self.rsi_values, self.ema_50[-1], self.atr, self.product_id)
        self.assertFalse(signal)
    
    def test_volume_above_sma(self):
        """Test volume above SMA condition"""
        # Test when volume is above SMA
        df = self.df.copy()
        df.loc[4, 'volume'] = 1500  # Volume above SMA (1000)
        signal, _, _ = check_entry_conditions(df, self.rsi_values, self.ema_50[-1], self.atr, self.product_id)
        self.assertTrue(signal)
        
        # Test when volume is below SMA
        df.loc[4, 'volume'] = 500  # Volume below SMA (1000)
        signal, _, _ = check_entry_conditions(df, self.rsi_values, self.ema_50[-1], self.atr, self.product_id)
        self.assertFalse(signal)
    
    def test_body_below_sma(self):
        """Test body size below SMA condition"""
        # Test when body size is below SMA
        df = self.df.copy()
        df.loc[4, 'body_size'] = 2  # Body size below SMA (3)
        signal, _, _ = check_entry_conditions(df, self.rsi_values, self.ema_50[-1], self.atr, self.product_id)
        self.assertTrue(signal)
        
        # Test when body size is above SMA
        df.loc[4, 'body_size'] = 4  # Body size above SMA (3)
        signal, _, _ = check_entry_conditions(df, self.rsi_values, self.ema_50[-1], self.atr, self.product_id)
        self.assertFalse(signal)
    
    def test_price_above_ema(self):
        """Test price above EMA condition"""
        # Test when price is above EMA
        df = self.df.copy()
        df.loc[4, 'close'] = 105  # Price above EMA (100)
        _, _, multiplier = check_entry_conditions(df, self.rsi_values, self.ema_50[-1], self.atr, self.product_id)
        self.assertEqual(multiplier, 1.0)
        
        # Test when price is below EMA
        df.loc[4, 'close'] = 95  # Price below EMA (100)
        _, _, multiplier = check_entry_conditions(df, self.rsi_values, self.ema_50[-1], self.atr, self.product_id)
        self.assertEqual(multiplier, 0.5)
    
    def test_chop_zone(self):
        """Test chop zone condition"""
        # Test when ATR is below chop zone threshold
        signal, _, _ = check_entry_conditions(self.df, self.rsi_values, self.ema_50[-1], 0.4, self.product_id)
        self.assertFalse(signal)
        
        # Test when ATR is above chop zone threshold
        signal, _, _ = check_entry_conditions(self.df, self.rsi_values, self.ema_50[-1], 0.6, self.product_id)
        self.assertTrue(signal)

if __name__ == "__main__":
    args = parse_args()
    
    if args.test:
        # Run unit tests
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    else:
        # Run the trading bot
        main() 