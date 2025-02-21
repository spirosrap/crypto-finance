import logging
import time
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS

def setup_logging():
    """Set up basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize CoinbaseService with your API credentials
        coinbase = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
        
        # First cancel all open orders
        logger.info("Cancelling all open orders first...")
        coinbase.cancel_all_orders()
        # Close all positions
        logger.info("Now closing all positions...")
        closed_positions = coinbase.close_all_positions()
        
        # Get position details before closing
        logger.info("Getting position details...")
        
        # Get the INTX portfolio UUID
        ports = coinbase.client.get_portfolios()
        portfolio_uuid = None
        for p in ports['portfolios']:
            if p['type'] == "INTX":
                portfolio_uuid = p['uuid']
                break
        
        if not portfolio_uuid:
            logger.error("Could not find INTX portfolio")
            return
            
        # Get portfolio positions
        portfolio = coinbase.client.get_portfolio_breakdown(portfolio_uuid=portfolio_uuid)
        
        # Access the perp_positions from the response object
        positions = []
        if hasattr(portfolio, 'breakdown'):
            if hasattr(portfolio.breakdown, 'perp_positions'):
                positions = portfolio.breakdown.perp_positions
               
        # Print details about closed positions
        logger.info("Position closing complete. Position details:")
        total_pnl = 0
 
        for position in positions:
            try:
                # Access the nested dictionary structure
                symbol = position['symbol']
                size = float(position['net_size'])
                mark_price = float(position['mark_price']['rawCurrency']['value'])
                unrealized_pnl = float(position['unrealized_pnl']['rawCurrency']['value'])
                position_side = position['position_side']
                leverage = position['leverage']
                
                # Log raw P&L value for debugging
                logger.debug(f"Raw unrealized P&L value: {unrealized_pnl}")
                
                total_pnl += unrealized_pnl
                
                logger.info(f"""
                Symbol: {symbol}
                Position Side: {position_side}
                Position Size: {size}
                Mark Price: ${mark_price:.2f}
                Leverage: {leverage}x
                Unrealized P&L: ${unrealized_pnl:.2f}
                ------------------------""")
                
            except Exception as e:
                logger.error(f"Error processing position: {str(e)}")
                logger.error(f"Position data: {position}")
                continue
            
        logger.info(f"Total P&L: ${total_pnl:.2f}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    main() 