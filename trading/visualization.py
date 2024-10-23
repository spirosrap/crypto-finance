import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from typing import List, Dict, Any
from trading.models import TradeRecord

def plot_trades(candles: List[Dict[str, Any]], trades: List[TradeRecord], 
                balance_history: List[float], btc_balance_history: List[float]) -> None:
    # Convert timestamps to datetime
    dates = [datetime.fromtimestamp(float(candle['start']), tz=timezone.utc) for candle in candles]
    prices = [float(candle['close']) for candle in candles]

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot price
    ax1.plot(dates, prices, label='Price')

    # Plot trades
    for trade in trades:
        date = datetime.fromtimestamp(int(trade.date), tz=timezone.utc)
        if trade.action in ['BUY', 'STRONG BUY']:
            ax1.plot(date, trade.price, '^', color='g', markersize=10)
        elif trade.action in ['SELL', 'STRONG SELL', 'STOP LOSS', 'TRAILING STOP']:
            ax1.plot(date, trade.price, 'v', color='r', markersize=10)

    # Plot balances
    ax2.plot(dates, balance_history, label='USD Balance')
    ax2.plot(dates, btc_balance_history, label='BTC Balance (in USD)')

    # Format the plot
    ax1.set_title('Price and Trades')
    ax1.legend()
    ax2.set_title('Account Balance')
    ax2.legend()

    plt.xlabel('Date')
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig('trades_and_balance.png')
    plt.close()
