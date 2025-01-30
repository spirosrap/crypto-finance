from flask import Flask, request, jsonify
from base import CryptoTrader
from config import API_KEY, API_SECRET
from datetime import datetime, timedelta, timezone
import traceback

app = Flask(__name__)
trader = CryptoTrader(API_KEY, API_SECRET)


@app.route('/backtest', methods=['GET'])
def run_backtest():
    try:
        app.logger.info("Backtest request received")
        product_id = request.args.get('product_id', 'BTC-USDC')
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
        else:
            # If no special case and no dates provided, use last year as default
            if not start_date and not end_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d 00:00:00")
                end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # If only start_date is provided, set end_date to now
            elif start_date and not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # If only end_date is provided, set start_date to one year before
            elif not start_date and end_date:
                start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d 00:00:00")

        # Ensure start_date and end_date are in the correct format
        if start_date and len(start_date) == 10:  # If only date is provided
            start_date += " 00:00:00"
        if end_date and len(end_date) == 10:  # If only date is provided
            end_date += " 23:59:59"

        app.logger.info(f"Running backtest from {start_date} to {end_date}")
        final_value, trades = trader.run_backtest(product_id, start_date, end_date, initial_balance, risk_per_trade, trailing_stop_percent)
        app.logger.info("Backtest completed")

        # Get the latest available candle data
        try:
            current_date = datetime.now()
            latest_candles = trader.get_historical_data(product_id, current_date - timedelta(days=1), current_date)
            if latest_candles:
                latest_candle = latest_candles[-1]
                close_price = float(latest_candle['close'])
                
                # Calculate signals and market conditions
                rsi = trader.compute_rsi_for_backtest(latest_candles)
                macd, signal, histogram = trader.compute_macd_for_backtest(latest_candles)
                market_conditions = trader.technical_analysis.analyze_market_conditions(latest_candles)
                combined_signal = trader.generate_combined_signal(rsi, macd, signal, histogram, latest_candles, market_conditions=market_conditions)
                trend = trader.identify_trend(product_id, latest_candles)
                volume_signal = trader.technical_analysis.analyze_volume(latest_candles)

                current_info = {
                    "combined_signal": combined_signal,
                    "market_conditions": market_conditions,
                    "trend": trend,
                    "volume_signal": volume_signal,
                    "current_bitcoin_value": close_price,
                    "data_as_of": latest_candle['start']
                }
            else:
                current_info = {
                    "error": "No recent data available. Unable to calculate current market information."
                }
        except Exception as data_error:
            current_info = {
                "error": f"Failed to fetch latest market data: {str(data_error)}"
            }

        result = {
            "initial_balance": initial_balance,
            "final_value": final_value,
            "total_return": (final_value - initial_balance) / initial_balance * 100,
            "number_of_trades": len(trades),
            "trades": [
                {
                    "date": datetime.fromtimestamp(int(trade['date']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "action": trade['action'],
                    "price": trade['price'],
                    "amount": trade['amount']
                } for trade in trades
            ],
            "current_market_info": current_info
        }

        app.logger.info("Sending response")
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change the port to 5001