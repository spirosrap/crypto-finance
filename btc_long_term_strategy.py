import os
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from coinbaseservice import CoinbaseService
from config import API_KEY, API_SECRET
import argparse

# Set up logging with file output (following user preference)
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Add file handler for persistent logging test test
log_file = 'btc_long_term_strategy.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
logger.addHandler(file_handler)

@dataclass
class DCASchedule:
    """DCA schedule configuration"""
    frequency_days: int = 7  # Weekly DCA
    amount_usd: float = 100.0  # $100 per DCA
    enabled: bool = True
    max_dca_per_month: int = 4  # Maximum 4 DCA purchases per month

@dataclass
class PortfolioStats:
    """Portfolio statistics tracking"""
    total_invested_usd: float = 0.0
    total_btc_owned: float = 0.0
    average_price_usd: float = 0.0
    current_value_usd: float = 0.0
    total_return_usd: float = 0.0
    total_return_percent: float = 0.0
    last_dca_date: Optional[str] = None
    dca_count: int = 0

@dataclass
class TradeRecord:
    """Record of individual trades"""
    timestamp: str
    trade_type: str  # "INITIAL_BUY", "DCA_BUY", "MANUAL_BUY"
    btc_amount: float
    usd_amount: float
    price_per_btc: float
    portfolio_value_after: float

class BTCLongTermStrategy:
    """
    Long-term Bitcoin buy and hold strategy with DCA capabilities.
    
    Strategy:
    - Initial lump sum investment
    - Regular DCA purchases (weekly/monthly)
    - Hold forever - no selling
    - Track portfolio performance
    - Use Coinbase spot trading (not perpetuals)
    """
    
    def __init__(self, 
                 initial_investment_usd: float = 1000.0,
                 dca_schedule: DCASchedule = None,
                 portfolio_file: str = "btc_portfolio.json",
                 trades_file: str = "btc_trades.json"):
        
        self.initial_investment_usd = initial_investment_usd
        self.dca_schedule = dca_schedule or DCASchedule()
        self.portfolio_file = portfolio_file
        self.trades_file = trades_file
        
        # Initialize Coinbase service
        self.cb_service = self._setup_coinbase()
        
        # Load existing portfolio data
        self.portfolio_stats = self._load_portfolio_stats()
        self.trade_history = self._load_trade_history()
        
        logger.info(f"BTC Long Term Strategy initialized")
        logger.info(f"Initial investment: ${initial_investment_usd}")
        logger.info(f"DCA enabled: {self.dca_schedule.enabled}")
        if self.dca_schedule.enabled:
            logger.info(f"DCA amount: ${self.dca_schedule.amount_usd} every {self.dca_schedule.frequency_days} days")
    
    def _setup_coinbase(self) -> CoinbaseService:
        """Initialize CoinbaseService with API credentials."""
        api_key = API_KEY
        api_secret = API_SECRET
        
        if not api_key or not api_secret:
            raise ValueError("API credentials not found in config.py")
        
        return CoinbaseService(api_key, api_secret)
    
    def _load_portfolio_stats(self) -> PortfolioStats:
        """Load portfolio statistics from file."""
        try:
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    data = json.load(f)
                    return PortfolioStats(**data)
        except Exception as e:
            logger.error(f"Error loading portfolio stats: {e}")
        
        return PortfolioStats()
    
    def _save_portfolio_stats(self):
        """Save portfolio statistics to file."""
        try:
            with open(self.portfolio_file, 'w') as f:
                json.dump(asdict(self.portfolio_stats), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving portfolio stats: {e}")
    
    def _load_trade_history(self) -> List[TradeRecord]:
        """Load trade history from file."""
        try:
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)
                    return [TradeRecord(**trade) for trade in data]
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
        
        return []
    
    def _save_trade_history(self):
        """Save trade history to file."""
        try:
            with open(self.trades_file, 'w') as f:
                json.dump([asdict(trade) for trade in self.trade_history], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
    
    def get_current_btc_price(self) -> float:
        """Get current BTC price from Coinbase."""
        try:
            prices = self.cb_service.get_btc_prices()
            if "BTC-USDC" in prices:
                # Use ask price for buying
                return prices["BTC-USDC"]["ask"]
            # Fallback to BTC-USD if BTC-USDC not available
            if "BTC-USD" in prices:
                return prices["BTC-USD"]["ask"]
            else:
                logger.error("BTC-USDC price not available")
                return 0.0
        except Exception as e:
            logger.error(f"Error getting BTC price: {e}")
            return 0.0
    
    def get_portfolio_balance(self) -> Tuple[float, float]:
        """Get current portfolio balance (USD and BTC)."""
        try:
            fiat_balance, btc_balance = self.cb_service.get_portfolio_info(portfolio_type="DEFAULT")
            return fiat_balance, btc_balance
        except Exception as e:
            logger.error(f"Error getting portfolio balance: {e}")
            return 0.0, 0.0
    
    def get_usdc_balance(self) -> float:
        """Get current USDC balance in the account."""
        try:
            ports = self.cb_service.client.get_portfolios()["portfolios"]
            
            for p in ports:
                if p["type"] == "DEFAULT":
                    uuid = p["uuid"]
                    breakdown = self.cb_service.client.get_portfolio_breakdown(portfolio_uuid=uuid)
                    spot_positions = breakdown["breakdown"]["spot_positions"]
                    
                    for position in spot_positions:
                        if position["asset"] == "USDC":
                            usdc_balance = float(position["total_balance_crypto"])
                            logger.info(f"USDC Balance: {usdc_balance}")
                            return usdc_balance
                    
                    # If no USDC position found, return 0
                    logger.info("No USDC position found, balance: 0")
                    return 0.0
            
            logger.warning("DEFAULT portfolio not found")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting USDC balance: {e}")
            return 0.0
    
    def calculate_portfolio_value(self) -> float:
        """Calculate current total portfolio value in USD."""
        try:
            fiat_balance, btc_balance = self.get_portfolio_balance()
            btc_price = self.get_current_btc_price()
            total_value = fiat_balance + (btc_balance * btc_price)
            
            # Update portfolio stats
            self.portfolio_stats.current_value_usd = total_value
            self.portfolio_stats.total_btc_owned = btc_balance
            self.portfolio_stats.total_return_usd = total_value - self.portfolio_stats.total_invested_usd
            # Maintain average price based on current holdings
            if btc_balance > 0 and self.portfolio_stats.total_invested_usd > 0:
                self.portfolio_stats.average_price_usd = round(
                    self.portfolio_stats.total_invested_usd / btc_balance, 2
                )
            else:
                self.portfolio_stats.average_price_usd = 0.0
            
            if self.portfolio_stats.total_invested_usd > 0:
                self.portfolio_stats.total_return_percent = (
                    (self.portfolio_stats.total_return_usd / self.portfolio_stats.total_invested_usd) * 100
                )
            
            return total_value
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
    
    def place_btc_buy_order(self, usd_amount: float, trade_type: str = "MANUAL_BUY") -> bool:
        """
        Place a BTC buy order using USD.
        
        Args:
            usd_amount: Amount in USD to spend
            trade_type: Type of trade for record keeping
        
        Returns:
            bool: True if order was successful
        """
        try:
            # Check USDC balance first
            usdc_balance = self.get_usdc_balance()
            if usdc_balance < usd_amount:
                logger.error(f"Insufficient USDC balance: ${usdc_balance:.2f} available, ${usd_amount:.2f} needed")
                return False
            
            btc_price = self.get_current_btc_price()
            if btc_price <= 0:
                logger.error("Invalid BTC price")
                return False
            
            # Ensure currency precision: 2 decimals for USD amounts
            usd_amount = round(usd_amount, 2)

            logger.info(f"Placing {trade_type} order:")
            logger.info(f"  USD Amount: ${usd_amount:.2f}")
            logger.info(f"  USDC Balance: ${usdc_balance:.2f}")
            logger.info(f"  Price per BTC: ${btc_price:.2f}")
            
            # For spot trading, we need to use the correct order configuration
            # Use quote_size for BUY orders (amount in USDC)
            client_order_id = f"order_{int(time.time())}_{trade_type.lower()}"
            
            order_params = {
                "client_order_id": client_order_id,
                "product_id": "BTC-USDC",
                "side": "BUY",
                "order_configuration": {
                    "market_market_ioc": {
                        # Pass quote size with at most two decimals per USD precision
                        "quote_size": f"{usd_amount:.2f}"
                    }
                }
            }
            
            logger.info(f"Order parameters: {order_params}")
            
            # Place the order directly using the client
            order_result = self.cb_service.client.create_order(**order_params)
            
            logger.info(f"Order result: {order_result}")
            
            # Check if the order was successful
            btc_amount = 0.0
            order_successful = False
            
            if isinstance(order_result, dict):
                if order_result.get('success') == False:
                    error_msg = order_result.get('error_response', {}).get('message', 'Unknown error')
                    logger.error(f"Order failed: {error_msg}")
                    return False
                elif order_result.get('success') == True:
                    logger.info("Order successful!")
                    order_successful = True
                    # Calculate actual BTC amount from the order result
                    # For now, use the expected amount based on current price
                    btc_amount = usd_amount / btc_price
                    btc_amount = round(btc_amount, 6)
                else:
                    logger.warning("Order result format unclear")
                    return False
            else:
                # Handle case where order_result is an object
                if hasattr(order_result, 'order_id') and order_result.order_id:
                    logger.info(f"Order ID: {order_result.order_id}")
                    order_successful = True
                    btc_amount = usd_amount / btc_price
                    btc_amount = round(btc_amount, 6)
                else:
                    logger.warning("Order returned success but no order_id - may not have executed")
                    return False
            
            if order_successful:
                # Record the trade
                trade_record = TradeRecord(
                    timestamp=datetime.now().isoformat(),
                    trade_type=trade_type,
                    btc_amount=btc_amount,
                    usd_amount=usd_amount,
                    price_per_btc=round(btc_price, 2),
                    portfolio_value_after=self.calculate_portfolio_value()
                )
                
                self.trade_history.append(trade_record)
                self._save_trade_history()
                
                # Update portfolio stats
                self.portfolio_stats.total_invested_usd += usd_amount
                # Update DCA counters only for DCA buys
                if trade_type == "DCA_BUY":
                    self.portfolio_stats.dca_count += 1
                    self.portfolio_stats.last_dca_date = datetime.now().isoformat()
                self._save_portfolio_stats()
                
                logger.info(f"✅ {trade_type} order successful!")
                logger.info(f"  BTC Amount: {btc_amount:.6f}")
                return True
            else:
                logger.error(f"❌ {trade_type} order failed")
                return False
                
        except Exception as e:
            logger.error(f"Error placing {trade_type} order: {e}")
            return False
    
    def execute_initial_investment(self) -> bool:
        """Execute the initial lump sum investment."""
        if self.portfolio_stats.total_invested_usd > 0:
            logger.info("Initial investment already made, skipping...")
            return True
        
        # Check USDC balance before attempting investment
        usdc_balance = self.get_usdc_balance()
        if usdc_balance < self.initial_investment_usd:
            logger.error(f"Insufficient USDC for initial investment: ${usdc_balance:.2f} available, ${self.initial_investment_usd:.2f} needed")
            logger.info("Please add more USDC to your account before running the strategy")
            return False
        
        logger.info(f"Executing initial investment of ${self.initial_investment_usd}")
        return self.place_btc_buy_order(self.initial_investment_usd, "INITIAL_BUY")
    
    def should_execute_dca(self) -> bool:
        """Check if it's time to execute a DCA purchase."""
        if not self.dca_schedule.enabled:
            return False
        
        if not self.portfolio_stats.last_dca_date:
            # First DCA if we have initial investment
            return self.portfolio_stats.total_invested_usd > 0
        
        # Check frequency
        last_dca = datetime.fromisoformat(self.portfolio_stats.last_dca_date)
        days_since_last_dca = (datetime.now() - last_dca).days
        
        if days_since_last_dca < self.dca_schedule.frequency_days:
            return False
        
        # Check monthly limit
        current_month = datetime.now().month
        current_year = datetime.now().year
        monthly_dca_count = sum(
            1 for trade in self.trade_history
            if trade.trade_type == "DCA_BUY" and
            datetime.fromisoformat(trade.timestamp).month == current_month and
            datetime.fromisoformat(trade.timestamp).year == current_year
        )
        
        if monthly_dca_count >= self.dca_schedule.max_dca_per_month:
            logger.info(f"Monthly DCA limit reached ({monthly_dca_count}/{self.dca_schedule.max_dca_per_month})")
            return False
        
        return True
    
    def execute_dca(self) -> bool:
        """Execute a DCA purchase if conditions are met."""
        if not self.should_execute_dca():
            return False
        
        # Check USDC balance before attempting DCA
        usdc_balance = self.get_usdc_balance()
        if usdc_balance < self.dca_schedule.amount_usd:
            logger.warning(f"Insufficient USDC for DCA: ${usdc_balance:.2f} available, ${self.dca_schedule.amount_usd:.2f} needed")
            logger.info("Skipping DCA purchase - please add more USDC to your account")
            return False
        
        logger.info(f"Executing DCA purchase of ${self.dca_schedule.amount_usd}")
        return self.place_btc_buy_order(self.dca_schedule.amount_usd, "DCA_BUY")
    
    def print_portfolio_summary(self):
        """Print a comprehensive portfolio summary."""
        current_value = self.calculate_portfolio_value()
        usdc_balance = self.get_usdc_balance()
        
        print("\n" + "="*60)
        print("🚀 BITCOIN LONG-TERM PORTFOLIO SUMMARY")
        print("="*60)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Total Invested: ${self.portfolio_stats.total_invested_usd:,.2f}")
        print(f"🪙 BTC Owned: {self.portfolio_stats.total_btc_owned:.6f}")
        print(f"💵 Current Value: ${current_value:,.2f}")
        print(f"📈 Total Return: ${self.portfolio_stats.total_return_usd:,.2f}")
        print(f"📊 Return %: {self.portfolio_stats.total_return_percent:+.2f}%")
        print(f"💎 USDC Balance: ${usdc_balance:,.2f}")
        
        if self.portfolio_stats.total_invested_usd > 0 and self.portfolio_stats.total_btc_owned > 0:
            avg_price = self.portfolio_stats.total_invested_usd / self.portfolio_stats.total_btc_owned
            current_price = self.get_current_btc_price()
            print(f"📉 Average Price: ${avg_price:,.2f}")
            print(f"📈 Current Price: ${current_price:,.2f}")
        elif self.portfolio_stats.total_invested_usd > 0:
            current_price = self.get_current_btc_price()
            print(f"📈 Current Price: ${current_price:,.2f}")
            print(f"⚠️  No BTC owned yet (order may be pending)")
        
        print(f"🔄 DCA Count: {self.portfolio_stats.dca_count}")
        if self.portfolio_stats.last_dca_date:
            print(f"📅 Last DCA: {self.portfolio_stats.last_dca_date}")
        
        print("="*60)
    
    def print_trade_history(self, limit: int = 10):
        """Print recent trade history."""
        print(f"\n📋 RECENT TRADES (Last {limit}):")
        print("-" * 80)
        print(f"{'Date':<20} {'Type':<15} {'BTC':<12} {'USD':<12} {'Price/BTC':<12}")
        print("-" * 80)
        
        for trade in self.trade_history[-limit:]:
            date = datetime.fromisoformat(trade.timestamp).strftime('%Y-%m-%d %H:%M')
            print(f"{date:<20} {trade.trade_type:<15} {trade.btc_amount:<12.6f} "
                  f"${trade.usd_amount:<11.2f} ${trade.price_per_btc:<11.2f}")
    
    def run_strategy(self, auto_dca: bool = True):
        """
        Run the long-term strategy.
        
        Args:
            auto_dca: Whether to automatically execute DCA if conditions are met
        """
        logger.info("Starting BTC Long Term Strategy")
        
        # Execute initial investment if not done
        if self.portfolio_stats.total_invested_usd == 0:
            self.execute_initial_investment()
        
        # Execute DCA if conditions are met
        if auto_dca:
            self.execute_dca()
        
        # Print summary
        self.print_portfolio_summary()
        self.print_trade_history()

def main():
    parser = argparse.ArgumentParser(description="Bitcoin Long Term Buy & Hold Strategy")
    parser.add_argument("--initial", type=float, default=1000.0, 
                       help="Initial investment amount in USD")
    parser.add_argument("--dca-amount", type=float, default=100.0,
                       help="DCA amount in USD")
    parser.add_argument("--dca-frequency", type=int, default=7,
                       help="DCA frequency in days")
    parser.add_argument("--disable-dca", action="store_true",
                       help="Disable DCA functionality")
    parser.add_argument("--summary", action="store_true",
                       help="Show portfolio summary only")
    parser.add_argument("--manual-buy", type=float,
                       help="Execute manual buy with specified USD amount")
    
    args = parser.parse_args()
    
    # Create DCA schedule
    dca_schedule = DCASchedule(
        frequency_days=args.dca_frequency,
        amount_usd=args.dca_amount,
        enabled=not args.disable_dca
    )
    
    # Initialize strategy
    strategy = BTCLongTermStrategy(
        initial_investment_usd=args.initial,
        dca_schedule=dca_schedule
    )
    
    if args.summary:
        strategy.print_portfolio_summary()
        strategy.print_trade_history()
    elif args.manual_buy:
        strategy.place_btc_buy_order(args.manual_buy, "MANUAL_BUY")
        strategy.print_portfolio_summary()
    else:
        strategy.run_strategy()

if __name__ == "__main__":
    main()
