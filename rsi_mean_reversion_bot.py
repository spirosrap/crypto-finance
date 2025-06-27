import time
from datetime import datetime, timedelta, UTC
import logging
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
import pandas as pd
import pandas_ta as ta
import subprocess
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PRODUCT_ID = "BTC-PERP-INTX"
GRANULARITY = "FIFTEEN_MINUTE"
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
POLL_INTERVAL_SECONDS = 60 # Poll every 60 seconds

# Trade parameters
MARGIN = 100  # USD
LEVERAGE = 10  # 10x leverage
RISK_REWARD_RATIO = 1.5

# --- State Management ---
# Simple file-based state to track active trades and cooldowns
STATE_FILE = "rsi_bot_state.json"

def get_state():
    if not os.path.exists(STATE_FILE):
        return {"active_trade_side": None, "cooldown_until": None}
    try:
        import json
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading state file: {e}")
        return {"active_trade_side": None, "cooldown_until": None}

def save_state(state):
    try:
        import json
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Error saving state file: {e}")

def setup_coinbase():
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise ValueError("API credentials not found in config.py")
    return CoinbaseService(api_key, api_secret)

def execute_trade(cb_service, side, entry_price, stop_loss, take_profit):
    """
    Executes a trade using the trade_btc_perp.py script.
    """
    try:
        position_size_usd = MARGIN * LEVERAGE
        trade_type = f"RSI Mean Reversion ({side})"
        
        cmd = [
            sys.executable, 'trade_btc_perp.py',
            '--product', PRODUCT_ID,
            '--side', side,
            '--size', str(position_size_usd),
            '--leverage', str(LEVERAGE),
            '--tp', str(take_profit),
            '--sl', str(stop_loss),
            '--no-confirm'
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info(f"Trade executed successfully: {trade_type}")
            logger.info(f"Trade output: {result.stdout}")
            return True
        else:
            logger.error(f"Trade execution failed: {trade_type}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Trade execution timed out")
        return False
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False

def check_for_rsi_signal(cb_service):
    """
    Checks for RSI overbought/oversold signals and executes trades.
    """
    state = get_state()
    
    # Check if there's an active trade
    if state.get("active_trade_side"):
        # In a real scenario, we would check if the trade is still open via API.
        # For this version, we assume the trade is active until manually cleared
        # or a more sophisticated state management is added.
        # We will add a check for open positions here later.
        logger.info(f"Skipping check: A trade is already active ({state['active_trade_side']}).")
        # A simple way to reset is to check if positions are zero.
        try:
            # Simple position check using the client directly
            portfolio_response = cb_service.client.get_portfolios()
            portfolios = portfolio_response.portfolios if hasattr(portfolio_response, 'portfolios') else portfolio_response.get('portfolios', [])
            
            has_open_positions = False
            for portfolio in portfolios:
                if portfolio.get('type') == 'INTX':
                    # Check if there are any open positions in the INTX portfolio
                    breakdown_response = cb_service.client.get_portfolio_breakdown(portfolio['uuid'])
                    breakdown = breakdown_response.breakdown if hasattr(breakdown_response, 'breakdown') else breakdown_response.get('breakdown', {})
                    
                    positions = breakdown.get('portfolio_balances', {}).get('positions', [])
                    for position in positions:
                        if abs(float(position.get('size', 0))) > 0.001:  # Check if position size is significant
                            has_open_positions = True
                            break
                    break
            
            if not has_open_positions:
                logger.info("No open positions found. Resetting active trade state.")
                state["active_trade_side"] = None
                save_state(state)
            else:
                logger.info("Positions are still open")
                return # Don't check for new signals if a position is open
        except Exception as e:
            logger.error(f"Could not verify open positions: {e}")
            # Continue with the signal check since we can't verify position status
            pass


    # Check for cooldown
    if state.get("cooldown_until"):
        if datetime.now(UTC).timestamp() < state["cooldown_until"]:
            logger.info("In cooldown period after last trade.")
            return
        else:
            state["cooldown_until"] = None
            save_state(state)

    try:
        # Fetch candles
        periods_needed = RSI_PERIOD + 2
        now = datetime.now(UTC)
        start = now - timedelta(minutes=periods_needed * 15)
        end = now
        
        response = cb_service.client.get_public_candles(
            product_id=PRODUCT_ID,
            start=int(start.timestamp()),
            end=int(end.timestamp()),
            granularity=GRANULARITY
        )
        
        if hasattr(response, 'candles'):
            candles = response.candles
        else:
            candles = response.get('candles', [])
        if not candles or len(candles) < RSI_PERIOD:
            logger.warning("Not enough candle data for RSI calculation.")
            return

        # Sort candles by timestamp (oldest first for RSI calculation)
        candles_sorted = sorted(candles, key=lambda x: int(x['start']))
        
        # Extract close prices for RSI calculation
        close_prices = [float(candle['close']) for candle in candles_sorted]
        
        # Calculate RSI using pandas_ta
        close_series = pd.Series(close_prices)
        rsi_series = ta.rsi(close_series, length=RSI_PERIOD)
        
        # Get the most recent candle and RSI value
        last_candle = candles_sorted[-1]
        rsi_value = rsi_series.iloc[-1]
        current_price = float(last_candle['close'])
        signal_candle_high = float(last_candle['high'])
        signal_candle_low = float(last_candle['low'])

        candle_time = datetime.fromtimestamp(int(last_candle['start']), UTC)
        logger.info(f"Checking RSI for {PRODUCT_ID} | Time: {candle_time} | Price: ${current_price:,.2f} | RSI({RSI_PERIOD}): {rsi_value:.2f}")

        # --- TRADING LOGIC ---
        side = None
        stop_loss = None
        take_profit = None

        # Oversold signal (BUY)
        if rsi_value < RSI_OVERSOLD:
            side = "BUY"
            risk_amount = current_price - signal_candle_low
            stop_loss = current_price - risk_amount * 1.1 # Place SL just below the low
            take_profit = current_price + risk_amount * RISK_REWARD_RATIO
            logger.info(f"RSI OVERSOLD SIGNAL ({rsi_value:.2f} < {RSI_OVERSOLD})")

        # Overbought signal (SELL)
        elif rsi_value > RSI_OVERBOUGHT:
            side = "SELL"
            risk_amount = signal_candle_high - current_price
            stop_loss = current_price + risk_amount * 1.1 # Place SL just above the high
            take_profit = current_price - risk_amount * RISK_REWARD_RATIO
            logger.info(f"RSI OVERBOUGHT SIGNAL ({rsi_value:.2f} > {RSI_OVERBOUGHT})")

        if side:
            logger.info(f"ACTION: Execute {side} trade.")
            logger.info(f"  - Entry Price (approx): ${current_price:,.2f}")
            logger.info(f"  - Stop Loss: ${stop_loss:,.2f}")
            logger.info(f"  - Take Profit: ${take_profit:,.2f}")
            
            trade_successful = execute_trade(cb_service, side, current_price, round(stop_loss, 2), round(take_profit, 2))
            
            if trade_successful:
                # Set active trade and cooldown
                state["active_trade_side"] = side
                # Cooldown for 4 hours to let the trade play out
                state["cooldown_until"] = (datetime.now(UTC) + timedelta(hours=4)).timestamp()
                save_state(state)

    except Exception as e:
        logger.error(f"An error occurred in the signal check loop: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    logger.info("Starting RSI Mean Reversion Bot...")
    try:
        cb_service = setup_coinbase()
        logger.info("Coinbase service initialized successfully.")
    except ValueError as e:
        logger.error(e)
        return

    while True:
        try:
            check_for_rsi_signal(cb_service)
            logger.info(f"Waiting {POLL_INTERVAL_SECONDS} seconds until next poll...")
            time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
