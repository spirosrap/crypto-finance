from flask import Flask, request, jsonify
from base import CryptoTrader
from config import API_KEY, API_SECRET
from datetime import datetime, timedelta
import traceback

app = Flask(__name__)
trader = CryptoTrader(API_KEY, API_SECRET)

@app.route('/backtest', methods=['GET'])
def run_backtest():
    try:
        app.logger.info("Backtest request received")
        product_id = request.args.get('product_id', 'BTC-USD')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        initial_balance = float(request.args.get('initial_balance', 10000))
        risk_per_trade = float(request.args.get('risk_per_trade', 0.02))
        trailing_stop_percent = float(request.args.get('trailing_stop_percent', 0.08))

        app.logger.info(f"Parameters: product_id={product_id}, start_date={start_date}, end_date={end_date}, "
                        f"initial_balance={initial_balance}, risk_per_trade={risk_per_trade}, "
                        f"trailing_stop_percent={trailing_stop_percent}")

        # Handle special date cases
        if request.args.get('bearmarket'):
            start_date = "2021-11-01 00:00:00"
            end_date = "2022-11-01 23:59:59"
        elif request.args.get('bullmarket'):
            start_date = "2020-10-01 00:00:00"
            end_date = "2021-04-01 23:59:59"
        elif request.args.get('ytd'):
            start_date = "2024-01-01 00:00:00"
            end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d 00:00:00")
            end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        app.logger.info(f"Running backtest from {start_date} to {end_date}")
        final_value, trades = trader.run_backtest(product_id, start_date, end_date, initial_balance, risk_per_trade, trailing_stop_percent)
        app.logger.info("Backtest completed")

        result = {
            "initial_balance": initial_balance,
            "final_value": final_value,
            "total_return": (final_value - initial_balance) / initial_balance * 100,
            "number_of_trades": len(trades),
            "trades": [
                {
                    "date": datetime.utcfromtimestamp(int(trade['date'])).strftime('%Y-%m-%d %H:%M:%S'),
                    "action": trade['action'],
                    "price": trade['price'],
                    "amount": trade['amount']
                } for trade in trades
            ]
        }

        app.logger.info("Sending response")
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change the port to 5001