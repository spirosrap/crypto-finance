#!/usr/bin/env python3
"""
ChatGPT Trades Analyzer

This program analyzes trades from chatgpt_trades.csv to determine if they were wins or losses
based on whether the price action (including wicks) reached the take profit or stop loss levels.

Key Features:
- Fetches historical candle data for each trade
- Analyzes wicks to determine if TP/SL was hit
- Calculates actual trade outcomes
- Provides detailed analysis and statistics
- Handles both long and short positions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import logging
import sys
import os
from pathlib import Path

# Define UTC timezone
UTC = timezone.utc

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatgpt_trades_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ChatGPTTradesAnalyzer:
    def __init__(self, csv_file_path: str = "chatgpt_trades.csv"):
        self.csv_file_path = csv_file_path
        self.trades_df = None
        self.analyzed_trades = []
        self.cb_service = None
        
    def load_trades(self) -> bool:
        """Load trades from CSV file."""
        try:
            if not os.path.exists(self.csv_file_path):
                logger.error(f"CSV file not found: {self.csv_file_path}")
                return False
                
            self.trades_df = pd.read_csv(self.csv_file_path)
            logger.info(f"Loaded {len(self.trades_df)} trades from {self.csv_file_path}")
            
            # Convert timestamp to datetime
            self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
            
            # Convert numeric columns
            numeric_columns = ['entry_price', 'stop_loss', 'take_profit', 'position_size_usd', 'margin', 'leverage']
            for col in numeric_columns:
                if col in self.trades_df.columns:
                    self.trades_df[col] = pd.to_numeric(self.trades_df[col], errors='coerce')
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            return False
    
    def initialize_coinbase_service(self):
        """Initialize Coinbase service for fetching historical data."""
        try:
            self.cb_service = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
            logger.info("Coinbase service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Coinbase service: {e}")
            raise
    
    def get_perp_product(self, symbol: str) -> str:
        """Convert symbol to perpetual futures product ID."""
        perp_map = {
            'BTC': 'BTC-PERP-INTX',
            'ETH': 'ETH-PERP-INTX',
            'SOL': 'SOL-PERP-INTX',
            'DOGE': 'DOGE-PERP-INTX',
            'XRP': 'XRP-PERP-INTX',
            'NEAR': 'NEAR-PERP-INTX',
            'SUI': 'SUI-PERP-INTX',
            'ATOM': 'ATOM-PERP-INTX'
        }
        
        # Extract base symbol (remove -USDC, -PERP-INTX, etc.)
        base_symbol = symbol.split('-')[0]
        return perp_map.get(base_symbol, f"{base_symbol}-PERP-INTX")
    
    def fetch_historical_data(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch historical candle data for analysis."""
        try:
            # Convert to perpetual product if needed
            if not symbol.endswith('-PERP-INTX'):
                perp_symbol = self.get_perp_product(symbol)
                logger.info(f"Converting {symbol} to {perp_symbol}")
                symbol = perp_symbol
            
            # Fetch 5-minute candles for granular analysis
            candles = self.cb_service.historical_data.get_historical_data(
                symbol, start_time, end_time, "FIVE_MINUTE"
            )
            
            if not candles:
                logger.warning(f"No candles returned for {symbol}")
                return pd.DataFrame()
            
            # Log the first candle to understand the structure
            logger.info(f"First candle structure: {candles[0] if candles else 'No candles'}")
            logger.info(f"First candle type: {type(candles[0]) if candles else 'No candles'}")
            if candles and hasattr(candles[0], '__dict__'):
                logger.info(f"First candle attributes: {candles[0].__dict__}")
            
            # Convert candles to a format that pandas can handle
            processed_candles = []
            for candle in candles:
                if isinstance(candle, dict):
                    # Already a dictionary
                    processed_candles.append(candle)
                else:
                    # Convert object to dictionary
                    try:
                        candle_dict = {
                            'start': candle.start,
                            'time': candle.start,
                            'low': candle.low,
                            'high': candle.high,
                            'open': candle.open,
                            'close': candle.close,
                            'volume': candle.volume
                        }
                        processed_candles.append(candle_dict)
                    except AttributeError as e:
                        logger.error(f"Error converting candle to dict: {e}")
                        logger.error(f"Candle object: {candle}")
                        continue
            
            if not processed_candles:
                logger.warning(f"No valid candles after processing for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(processed_candles)
            
            # Convert string columns to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle timestamp conversion - the candles have a 'start' field
            if 'start' in df.columns:
                # Convert the start timestamp to datetime (explicitly convert to numeric first)
                df['timestamp'] = pd.to_datetime(pd.to_numeric(df['start']), unit='s', utc=True)
                df.set_index('timestamp', inplace=True)
            else:
                logger.warning(f"No 'start' column found in candles. Available columns: {df.columns.tolist()}")
                return pd.DataFrame()
            
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            logger.error(f"Candle structure: {candles[0] if candles else 'No candles'}")
            return pd.DataFrame()
    
    def analyze_trade_outcome(self, trade: pd.Series) -> Dict:
        """Analyze a single trade to determine if TP/SL was hit based on wicks."""
        try:
            # Extract trade information
            entry_time = trade['timestamp']
            symbol = trade['symbol']
            side = trade['side']
            entry_price = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            position_size_usd = trade['position_size_usd']
            margin = trade['margin']
            leverage = trade['leverage']
            
            # Validate required fields
            if pd.isna(entry_price) or pd.isna(stop_loss) or pd.isna(take_profit):
                logger.warning(f"Missing required price data for trade: entry={entry_price}, SL={stop_loss}, TP={take_profit}")
                return self._create_error_result(symbol, side, entry_price, stop_loss, take_profit, entry_time, "MISSING_PRICE_DATA")
            
            logger.info(f"Analyzing trade: {symbol} {side} at {entry_price}, TP: {take_profit}, SL: {stop_loss}")
            
            # Define time window for analysis (24 hours after entry)
            start_time = entry_time
            end_time = entry_time + timedelta(hours=24)
            
            # Fetch historical data
            df = self.fetch_historical_data(symbol, start_time, end_time)
            
            if df.empty:
                logger.warning(f"No historical data available for {symbol}")
                return self._create_error_result(symbol, side, entry_price, stop_loss, take_profit, entry_time, "NO_DATA")
            
            # Filter candles after entry time
            df = df[df.index > entry_time]
            
            if df.empty:
                logger.warning(f"No candles after entry time for {symbol}")
                return self._create_error_result(symbol, side, entry_price, stop_loss, take_profit, entry_time, "NO_CANDLES_AFTER_ENTRY")
            
            # Initialize tracking variables
            tp_hit = False
            sl_hit = False
            exit_price = None
            exit_time = None
            exit_reason = None
            max_favorable_excursion = 0.0
            max_adverse_excursion = 0.0
            
            # Analyze each candle for wicks hitting TP/SL
            for timestamp, candle in df.iterrows():
                high = candle['high']
                low = candle['low']
                
                if side == 'BUY':
                    # For long positions, check if high wick hits TP or low wick hits SL
                    if high >= take_profit and not tp_hit:
                        tp_hit = True
                        exit_price = take_profit
                        exit_time = timestamp
                        exit_reason = 'TAKE_PROFIT_HIT'
                        logger.info(f"TP hit for {symbol} at {timestamp}: high={high}, TP={take_profit}")
                    
                    if low <= stop_loss and not sl_hit:
                        sl_hit = True
                        exit_price = stop_loss
                        exit_time = timestamp
                        exit_reason = 'STOP_LOSS_HIT'
                        logger.info(f"SL hit for {symbol} at {timestamp}: low={low}, SL={stop_loss}")
                    
                    # Track maximum favorable and adverse excursions
                    current_profit_pct = ((high - entry_price) / entry_price) * 100 * leverage
                    current_loss_pct = ((low - entry_price) / entry_price) * 100 * leverage
                    
                    max_favorable_excursion = max(max_favorable_excursion, current_profit_pct)
                    max_adverse_excursion = min(max_adverse_excursion, current_loss_pct)
                    
                else:  # SELL
                    # For short positions, check if low wick hits TP or high wick hits SL
                    if low <= take_profit and not tp_hit:
                        tp_hit = True
                        exit_price = take_profit
                        exit_time = timestamp
                        exit_reason = 'TAKE_PROFIT_HIT'
                        logger.info(f"TP hit for {symbol} at {timestamp}: low={low}, TP={take_profit}")
                    
                    if high >= stop_loss and not sl_hit:
                        sl_hit = True
                        exit_price = stop_loss
                        exit_time = timestamp
                        exit_reason = 'STOP_LOSS_HIT'
                        logger.info(f"SL hit for {symbol} at {timestamp}: high={high}, SL={stop_loss}")
                    
                    # Track maximum favorable and adverse excursions
                    current_profit_pct = ((entry_price - low) / entry_price) * 100 * leverage
                    current_loss_pct = ((entry_price - high) / entry_price) * 100 * leverage
                    
                    max_favorable_excursion = max(max_favorable_excursion, current_profit_pct)
                    max_adverse_excursion = min(max_adverse_excursion, current_loss_pct)
                
                # If either TP or SL is hit, stop analyzing
                if tp_hit or sl_hit:
                    break
            
            # Determine final outcome
            if tp_hit:
                outcome = 'WIN'
                if side == 'BUY':
                    profit_loss_pct = ((take_profit - entry_price) / entry_price) * 100 * leverage
                else:
                    profit_loss_pct = ((entry_price - take_profit) / entry_price) * 100 * leverage
            elif sl_hit:
                outcome = 'LOSS'
                if side == 'BUY':
                    profit_loss_pct = ((stop_loss - entry_price) / entry_price) * 100 * leverage
                else:
                    profit_loss_pct = ((entry_price - stop_loss) / entry_price) * 100 * leverage
            else:
                outcome = 'OPEN'
                exit_price = df.iloc[-1]['close']  # Use last close price
                exit_time = df.index[-1]
                exit_reason = 'NO_TP_SL_HIT'
                
                # Calculate unrealized P&L
                if side == 'BUY':
                    profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100 * leverage
                else:
                    profit_loss_pct = ((entry_price - exit_price) / entry_price) * 100 * leverage
            
            # Calculate dollar P&L
            profit_loss_usd = margin * (profit_loss_pct / 100)
            
            result = {
                'trade_id': len(self.analyzed_trades) + 1,
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': entry_time,
                'outcome': outcome,
                'exit_price': exit_price,
                'exit_time': exit_time,
                'exit_reason': exit_reason,
                'profit_loss_pct': round(profit_loss_pct, 2),
                'profit_loss_usd': round(profit_loss_usd, 2),
                'max_favorable_excursion': round(max_favorable_excursion, 2),
                'max_adverse_excursion': round(max_adverse_excursion, 2),
                'candles_analyzed': len(df)
            }
            
            logger.info(f"Trade analysis complete: {outcome} ({profit_loss_pct:.2f}%)")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing trade: {e}")
            return self._create_error_result(
                symbol if 'symbol' in locals() else 'UNKNOWN',
                side if 'side' in locals() else 'UNKNOWN',
                entry_price if 'entry_price' in locals() else 0,
                stop_loss if 'stop_loss' in locals() else 0,
                take_profit if 'take_profit' in locals() else 0,
                entry_time if 'entry_time' in locals() else None,
                f'ERROR: {str(e)}'
            )
    
    def _create_error_result(self, symbol: str, side: str, entry_price: float, stop_loss: float, 
                           take_profit: float, entry_time, error_reason: str) -> Dict:
        """Create a standardized error result."""
        return {
            'trade_id': len(self.analyzed_trades) + 1,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': entry_time,
            'outcome': 'ERROR',
            'exit_price': None,
            'exit_time': None,
            'exit_reason': error_reason,
            'profit_loss_pct': 0.0,
            'profit_loss_usd': 0.0,
            'max_favorable_excursion': 0.0,
            'max_adverse_excursion': 0.0,
            'candles_analyzed': 0
        }
    
    def analyze_all_trades(self) -> List[Dict]:
        """Analyze all trades in the CSV file."""
        if self.trades_df is None:
            logger.error("No trades loaded. Call load_trades() first.")
            return []
        
        logger.info(f"Starting analysis of {len(self.trades_df)} trades...")
        
        for index, trade in self.trades_df.iterrows():
            logger.info(f"Analyzing trade {index + 1}/{len(self.trades_df)}")
            result = self.analyze_trade_outcome(trade)
            self.analyzed_trades.append(result)
            
            # Add a small delay to avoid rate limiting
            import time
            time.sleep(0.5)
        
        logger.info(f"Analysis complete. Processed {len(self.analyzed_trades)} trades.")
        return self.analyzed_trades
    
    def calculate_statistics(self) -> Dict:
        """Calculate comprehensive trading statistics."""
        if not self.analyzed_trades:
            logger.error("No analyzed trades available. Run analyze_all_trades() first.")
            return {}
        
        df = pd.DataFrame(self.analyzed_trades)
        
        # Basic statistics
        total_trades = len(df)
        winning_trades = len(df[df['outcome'] == 'WIN'])
        losing_trades = len(df[df['outcome'] == 'LOSS'])
        open_trades = len(df[df['outcome'] == 'OPEN'])
        error_trades = len(df[df['outcome'] == 'ERROR'])
        
        win_rate = (winning_trades / (winning_trades + losing_trades)) * 100 if (winning_trades + losing_trades) > 0 else 0
        
        # P&L statistics
        total_profit_loss_usd = df['profit_loss_usd'].sum()
        total_profit_loss_pct = df['profit_loss_pct'].sum()
        
        winning_trades_df = df[df['outcome'] == 'WIN']
        losing_trades_df = df[df['outcome'] == 'LOSS']
        
        avg_win_usd = winning_trades_df['profit_loss_usd'].mean() if len(winning_trades_df) > 0 else 0
        avg_loss_usd = losing_trades_df['profit_loss_usd'].mean() if len(losing_trades_df) > 0 else 0
        avg_win_pct = winning_trades_df['profit_loss_pct'].mean() if len(winning_trades_df) > 0 else 0
        avg_loss_pct = losing_trades_df['profit_loss_pct'].mean() if len(losing_trades_df) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades_df['profit_loss_usd'].sum() if len(winning_trades_df) > 0 else 0
        gross_loss = abs(losing_trades_df['profit_loss_usd'].sum()) if len(losing_trades_df) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        max_favorable_excursion = df['max_favorable_excursion'].max()
        max_adverse_excursion = df['max_adverse_excursion'].min()
        avg_favorable_excursion = df['max_favorable_excursion'].mean()
        avg_adverse_excursion = df['max_adverse_excursion'].mean()
        
        # Exit reason breakdown
        exit_reasons = df['exit_reason'].value_counts()
        
        # Symbol breakdown
        symbol_stats = df.groupby('symbol').agg({
            'outcome': lambda x: (x == 'WIN').sum() / len(x) * 100,
            'profit_loss_usd': 'sum',
            'profit_loss_pct': 'sum'
        }).round(2)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'open_trades': open_trades,
            'error_trades': error_trades,
            'win_rate': round(win_rate, 2),
            'total_profit_loss_usd': round(total_profit_loss_usd, 2),
            'total_profit_loss_pct': round(total_profit_loss_pct, 2),
            'avg_win_usd': round(avg_win_usd, 2),
            'avg_loss_usd': round(avg_loss_usd, 2),
            'avg_win_pct': round(avg_win_pct, 2),
            'avg_loss_pct': round(avg_loss_pct, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'inf',
            'max_favorable_excursion': round(max_favorable_excursion, 2),
            'max_adverse_excursion': round(max_adverse_excursion, 2),
            'avg_favorable_excursion': round(avg_favorable_excursion, 2),
            'avg_adverse_excursion': round(avg_adverse_excursion, 2),
            'exit_reasons': exit_reasons.to_dict(),
            'symbol_statistics': symbol_stats.to_dict('index')
        }
    
    def save_results(self, output_file: str = "chatgpt_trades_analysis_results.csv"):
        """Save analysis results to CSV file."""
        if not self.analyzed_trades:
            logger.error("No analyzed trades to save.")
            return
        
        df = pd.DataFrame(self.analyzed_trades)
        df.to_csv(output_file, index=False)
        logger.info(f"Analysis results saved to {output_file}")
    
    def print_summary(self):
        """Print a comprehensive summary of the analysis."""
        stats = self.calculate_statistics()
        
        print("\n" + "="*80)
        print("CHATGPT TRADES ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nTRADE COUNT:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Winning Trades: {stats['winning_trades']}")
        print(f"  Losing Trades: {stats['losing_trades']}")
        print(f"  Open Trades: {stats['open_trades']}")
        print(f"  Error Trades: {stats['error_trades']}")
        
        print(f"\nPERFORMANCE:")
        print(f"  Win Rate: {stats['win_rate']}%")
        print(f"  Total P&L: ${stats['total_profit_loss_usd']} ({stats['total_profit_loss_pct']}%)")
        print(f"  Average Win: ${stats['avg_win_usd']} ({stats['avg_win_pct']}%)")
        print(f"  Average Loss: ${stats['avg_loss_usd']} ({stats['avg_loss_pct']}%)")
        print(f"  Profit Factor: {stats['profit_factor']}")
        
        print(f"\nRISK METRICS:")
        print(f"  Max Favorable Excursion: {stats['max_favorable_excursion']}%")
        print(f"  Max Adverse Excursion: {stats['max_adverse_excursion']}%")
        print(f"  Avg Favorable Excursion: {stats['avg_favorable_excursion']}%")
        print(f"  Avg Adverse Excursion: {stats['avg_adverse_excursion']}%")
        
        print(f"\nEXIT REASONS:")
        for reason, count in stats['exit_reasons'].items():
            print(f"  {reason}: {count}")
        
        print(f"\nSYMBOL BREAKDOWN:")
        for symbol, data in stats['symbol_statistics'].items():
            win_rate = data['outcome']
            total_pnl_usd = data['profit_loss_usd']
            total_pnl_pct = data['profit_loss_pct']
            print(f"  {symbol}: {win_rate}% win rate, ${total_pnl_usd} ({total_pnl_pct}%)")
        
        print("\n" + "="*80)

def main():
    """Main function to run the analysis."""
    analyzer = ChatGPTTradesAnalyzer()
    
    # Load trades
    if not analyzer.load_trades():
        logger.error("Failed to load trades. Exiting.")
        return
    
    # Initialize Coinbase service
    try:
        analyzer.initialize_coinbase_service()
    except Exception as e:
        logger.error(f"Failed to initialize Coinbase service: {e}")
        return
    
    # Analyze all trades
    results = analyzer.analyze_all_trades()
    
    if results:
        # Save results
        analyzer.save_results()
        
        # Print summary
        analyzer.print_summary()
        
        logger.info("Analysis completed successfully!")
    else:
        logger.error("No results generated from analysis.")

if __name__ == "__main__":
    main()
