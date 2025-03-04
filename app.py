import os
import json
import time
import threading
import subprocess
import queue
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from coinbaseservice import CoinbaseService

app = Flask(__name__)

# Global variables
analysis_queue = queue.Queue()
price_update_thread = None
auto_trading_thread = None
current_process = None
auto_trading = False
price_update_active = False
trading_history = []

# Default configuration
config = {
    "product": "BTC-USDC",
    "model": "o1_mini",
    "granularity": "ONE_HOUR",
    "margin": "CROSS",
    "margin_amount": 60,
    "leverage": 10,
    "limit_order": False,
    "take_profit": 2.0,
    "stop_loss": 2.0,
    "risk_level": "medium"
}

# Model mapping for display names
model_display_names = {
    "o1_mini": "o1_mini",
    "o3_mini": "o3_mini",
    "deepseek": "deepseek",
    "grok": "grok",
    "gpt4o": "gpt4o"
}

# Available products
available_products = ["BTC-USDC", "ETH-USDC", "DOGE-USDC", "SOL-USDC", "SHIB-USDC"]

# Available granularities
available_granularities = ["ONE_HOUR", "THIRTY_MINUTE", "FIFTEEN_MINUTE", "FIVE_MINUTE", "ONE_MINUTE"]

# Initialize CoinbaseService
coinbase_service = None

def load_api_keys():
    """Load API keys from config.py or environment variables"""
    try:
        from config import API_KEY_PERPS, API_SECRET_PERPS
        return API_KEY_PERPS, API_SECRET_PERPS
    except ImportError:
        api_key = os.getenv('API_KEY_PERPS')
        api_secret = os.getenv('API_SECRET_PERPS')
        if not api_key or not api_secret:
            return None, None
        return api_key, api_secret

def initialize_coinbase_service():
    """Initialize the CoinbaseService with API keys"""
    global coinbase_service
    api_key, api_secret = load_api_keys()
    if api_key and api_secret:
        coinbase_service = CoinbaseService(api_key, api_secret)
        return True
    return False

def load_trade_history():
    """Load trade history from JSON file"""
    global trading_history
    try:
        if os.path.exists('trade_history.json'):
            with open('trade_history.json', 'r') as f:
                trading_history = json.load(f)
    except Exception as e:
        print(f"Error loading trade history: {str(e)}")

def save_trade_history():
    """Save trade history to JSON file"""
    global trading_history
    try:
        with open('trade_history.json', 'w') as f:
            json.dump(trading_history, f, indent=2)
    except Exception as e:
        print(f"Error saving trade history: {str(e)}")

def record_trade_result(result):
    """Record trade result in history"""
    global trading_history
    if result:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trading_history.append({
            "timestamp": timestamp,
            "result": result,
            "product": config["product"],
            "granularity": config["granularity"]
        })
        save_trade_history()

def detect_trade_result(output_text):
    """Detect trade result from output text"""
    if "Order executed successfully" in output_text:
        return "success"
    elif "Error executing order" in output_text:
        return "failure"
    return None

def start_price_updates():
    """Start price update thread"""
    global price_update_thread, price_update_active
    if price_update_thread is None or not price_update_thread.is_alive():
        price_update_active = True
        price_update_thread = threading.Thread(target=price_update_loop)
        price_update_thread.daemon = True
        price_update_thread.start()

def price_update_loop():
    """Background loop for price updates"""
    global price_update_active
    while price_update_active:
        try:
            if coinbase_service:
                # Get current price for the selected product
                product_id = config["product"]
                price_data = coinbase_service.get_btc_prices()
                
                # Debug log
                print(f"Price data received: {price_data}")
                
                if price_data and product_id in price_data:
                    # Extract the price (average of bid and ask)
                    bid_price = price_data[product_id].get("bid", 0)
                    ask_price = price_data[product_id].get("ask", 0)
                    price = (bid_price + ask_price) / 2 if bid_price and ask_price else 0
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Put price update in queue
                    analysis_queue.put({
                        "type": "price_update",
                        "price": price,
                        "timestamp": timestamp
                    })
                else:
                    # If the specific product is not found, try to get any BTC price
                    for key, value in price_data.items():
                        if "BTC" in key:
                            bid_price = value.get("bid", 0)
                            ask_price = value.get("ask", 0)
                            price = (bid_price + ask_price) / 2 if bid_price and ask_price else 0
                            
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            analysis_queue.put({
                                "type": "price_update",
                                "price": price,
                                "timestamp": timestamp,
                                "note": f"Using {key} price as fallback"
                            })
                            break
        except Exception as e:
            print(f"Price update error: {str(e)}")
            analysis_queue.put({
                "type": "error",
                "message": f"Price update error: {str(e)}"
            })
        
        # Sleep for 10 seconds before next update
        time.sleep(10)

def stop_price_updates():
    """Stop price update thread"""
    global price_update_active
    price_update_active = False
    if price_update_thread:
        price_update_thread.join(timeout=1)

def run_analysis(granularity):
    """Run market analysis with specified granularity"""
    global current_process
    
    # Create analysis thread
    analysis_thread = threading.Thread(
        target=_run_analysis_thread,
        args=(granularity,)
    )
    analysis_thread.daemon = True
    analysis_thread.start()
    
    return {"status": "Analysis started"}

def _run_analysis_thread(granularity):
    """Background thread for running analysis"""
    global current_process
    
    try:
        # Notify that analysis is starting
        analysis_queue.put({
            "type": "message",
            "message": f"Running {granularity.lower().replace('_', ' ')} timeframe analysis..."
        })
        
        # Create command for analysis
        cmd = [
            "python",
            "prompt_market.py",
            "--product_id",
            config["product"],
            f"--use_{config['model']}",
            "--granularity",
            granularity
        ]
        
        # Add trading options if enabled
        if config.get("execute_trades", False):
            cmd.append("--execute_trades")
            cmd.extend(["--margin", str(config["margin_amount"])])
            cmd.extend(["--leverage", str(config["leverage"])])
            if config.get("limit_order", False):
                cmd.append("--limit_order")
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        current_process = process
        
        # Read output line by line
        for line in iter(process.stdout.readline, ''):
            # Strip ANSI codes
            clean_line = _strip_ansi_codes(line)
            if clean_line.strip():
                analysis_queue.put({
                    "type": "message",
                    "message": clean_line
                })
        
        # Wait for process to complete
        process.wait()
        
        # Check if process completed successfully
        if process.returncode == 0:
            analysis_queue.put({
                "type": "status",
                "status": "Analysis completed"
            })
        else:
            analysis_queue.put({
                "type": "status",
                "status": f"Analysis failed with code {process.returncode}"
            })
    
    except Exception as e:
        analysis_queue.put({
            "type": "error",
            "message": f"Error in analysis: {str(e)}"
        })
    
    finally:
        current_process = None

def place_market_order(side):
    """Place a market order"""
    global current_process
    
    # Create order thread
    order_thread = threading.Thread(
        target=_run_market_order_thread,
        args=(side,)
    )
    order_thread.daemon = True
    order_thread.start()
    
    return {"status": f"{side.capitalize()} order initiated"}

def _run_market_order_thread(side):
    """Background thread for placing market order"""
    global current_process
    
    try:
        # Notify that order is being placed
        analysis_queue.put({
            "type": "message",
            "message": f"Placing {side.lower()} market order for {config['product']}..."
        })
        
        # Create command for order
        cmd = [
            "python",
            "prompt_market.py",
            "--product_id",
            config["product"],
            f"--use_{config['model']}",
            "--granularity",
            config["granularity"],
            "--execute_trades",
            "--margin",
            str(config["margin_amount"]),
            "--leverage",
            str(config["leverage"]),
            f"--{side.lower()}"
        ]
        
        if config["limit_order"]:
            cmd.append("--limit_order")
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        current_process = process
        
        # Read output line by line
        output_text = ""
        for line in iter(process.stdout.readline, ''):
            # Strip ANSI codes
            clean_line = _strip_ansi_codes(line)
            if clean_line.strip():
                output_text += clean_line
                analysis_queue.put({
                    "type": "message",
                    "message": clean_line
                })
        
        # Wait for process to complete
        process.wait()
        
        # Check if process completed successfully
        if process.returncode == 0:
            analysis_queue.put({
                "type": "status",
                "status": "Order completed"
            })
            
            # Detect and record trade result
            result = detect_trade_result(output_text)
            if result:
                record_trade_result(result)
        else:
            analysis_queue.put({
                "type": "status",
                "status": f"Order failed with code {process.returncode}"
            })
    
    except Exception as e:
        analysis_queue.put({
            "type": "error",
            "message": f"Error placing order: {str(e)}"
        })
    
    finally:
        current_process = None

def close_positions():
    """Close all open positions"""
    global current_process
    
    # Create close positions thread
    close_thread = threading.Thread(target=_close_positions_thread)
    close_thread.daemon = True
    close_thread.start()
    
    return {"status": "Closing positions initiated"}

def _close_positions_thread():
    """Background thread for closing positions"""
    global current_process
    
    try:
        # Notify that positions are being closed
        analysis_queue.put({
            "type": "message",
            "message": "Closing all open positions..."
        })
        
        if coinbase_service:
            # Close positions using CoinbaseService
            result = coinbase_service.close_all_positions()
            
            if result:
                analysis_queue.put({
                    "type": "message",
                    "message": "All positions closed successfully."
                })
            else:
                analysis_queue.put({
                    "type": "message",
                    "message": "No positions to close or error occurred."
                })
        else:
            analysis_queue.put({
                "type": "error",
                "message": "CoinbaseService not initialized. Cannot close positions."
            })
    
    except Exception as e:
        analysis_queue.put({
            "type": "error",
            "message": f"Error closing positions: {str(e)}"
        })

def check_open_orders_and_positions():
    """Check for open orders and positions"""
    if not coinbase_service:
        return False, False
    
    try:
        # Get portfolio info for PERPETUALS
        _, position_size = coinbase_service.get_portfolio_info(portfolio_type="PERPETUALS")
        has_positions = abs(position_size) > 0
        
        # Get open orders
        open_orders = coinbase_service.get_truly_open_orders(coinbase_service)
        has_open_orders = len(open_orders) > 0
        
        return has_open_orders, has_positions
    
    except Exception as e:
        print(f"Error checking orders and positions: {str(e)}")
        return False, False

def toggle_auto_trading():
    """Toggle auto-trading on/off"""
    global auto_trading, auto_trading_thread
    
    if not auto_trading:
        # Start auto-trading
        auto_trading = True
        
        # Start the auto-trading thread
        auto_trading_thread = threading.Thread(target=_auto_trading_loop)
        auto_trading_thread.daemon = True
        auto_trading_thread.start()
        
        return {"status": "Auto-trading started", "auto_trading": True}
    else:
        # Stop auto-trading
        auto_trading = False
        
        if auto_trading_thread:
            auto_trading_thread.join(timeout=1)
            auto_trading_thread = None
        
        return {"status": "Auto-trading stopped", "auto_trading": False}

def _auto_trading_loop():
    """Background loop for auto-trading"""
    global auto_trading
    
    while auto_trading:
        try:
            # Check if trading is allowed based on time
            current_time = datetime.now()
            current_hour = current_time.hour
            current_minute = current_time.minute
            current_time_float = current_hour + current_minute / 60.0
            
            # Trading is allowed from 17:00 (5 PM) to 7:20 AM
            if current_time_float >= 7.35 and current_time_float < 17.0:  # 7:21 AM to 5:00 PM
                analysis_queue.put({
                    "type": "message",
                    "message": "Trading paused: Current time is outside trading hours (5:00 PM - 7:20 AM)."
                })
                time.sleep(60)  # Check every minute
                continue
            
            # Check for open orders or positions
            has_open_orders, has_positions = check_open_orders_and_positions()
            
            if has_open_orders or has_positions:
                # Log that we're waiting for orders to close
                if has_open_orders and has_positions:
                    analysis_queue.put({
                        "type": "message",
                        "message": "Found open orders and positions. Waiting for them to close before continuing..."
                    })
                elif has_open_orders:
                    analysis_queue.put({
                        "type": "message",
                        "message": "Found open orders. Waiting for them to close before continuing..."
                    })
                else:
                    analysis_queue.put({
                        "type": "message",
                        "message": "Found open positions. Waiting for them to close before continuing..."
                    })
                
                # Wait for a shorter interval before checking again
                time.sleep(60)  # Check every minute
                continue
            
            # Get current granularity
            granularity = config["granularity"]
            
            # Run analysis with current granularity
            analysis_queue.put({
                "type": "message",
                "message": f"Running automated {granularity.lower().replace('_', ' ')} timeframe analysis..."
            })
            
            # Create and run the analysis process
            cmd = [
                "python",
                "prompt_market.py",
                "--product_id",
                config["product"],
                f"--use_{config['model']}",
                "--granularity",
                granularity,
                "--execute_trades",
                "--margin",
                str(config["margin_amount"]),
                "--leverage",
                str(config["leverage"])
            ]
            
            if config["limit_order"]:
                cmd.append("--limit_order")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output line by line
            output_text = ""
            for line in iter(process.stdout.readline, ''):
                # Strip ANSI codes
                clean_line = _strip_ansi_codes(line)
                if clean_line.strip():
                    output_text += clean_line
                    analysis_queue.put({
                        "type": "message",
                        "message": clean_line
                    })
            
            # Wait for process to complete
            process.wait()
            
            # Detect and record trade result
            result = detect_trade_result(output_text)
            if result:
                record_trade_result(result)
            
            # Wait time based on granularity
            wait_minutes = {
                'ONE_MINUTE': 0.3,  # Check every 20 seconds
                'FIVE_MINUTE': 2,   # Check every 2 minutes
                'FIFTEEN_MINUTE': 2, # Check every 2 minutes
                'THIRTY_MINUTE': 10, # Check every 10 minutes
                'ONE_HOUR': 20      # Check every 20 minutes
            }.get(granularity, 20)  # Default to 20 minutes
            
            # Sleep for the specified time
            time.sleep(wait_minutes * 60)
        
        except Exception as e:
            analysis_queue.put({
                "type": "error",
                "message": f"Error in auto-trading: {str(e)}"
            })
            time.sleep(60)  # Wait a minute before retrying

def _strip_ansi_codes(text):
    """Remove ANSI color codes from text"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def cancel_operation():
    """Cancel current operation"""
    global current_process
    
    if current_process:
        try:
            current_process.terminate()
            analysis_queue.put({
                "type": "message",
                "message": "Operation cancelled."
            })
            return {"status": "Operation cancelled"}
        except Exception as e:
            analysis_queue.put({
                "type": "error",
                "message": f"Error cancelling operation: {str(e)}"
            })
            return {"status": "Error cancelling operation", "error": str(e)}
    else:
        return {"status": "No operation to cancel"}

# Routes
@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', 
                          products=available_products,
                          granularities=available_granularities,
                          models=model_display_names,
                          config=config)

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update configuration"""
    global config
    
    if request.method == 'POST':
        # Update config with form data
        data = request.json
        for key in data:
            if key in config:
                # Convert leverage to int if provided
                if key == 'leverage' and data[key]:
                    config[key] = int(data[key])
                else:
                    config[key] = data[key]
        return jsonify({"status": "Configuration updated", "config": config})
    else:
        # Return current config
        return jsonify(config)

@app.route('/api/analysis', methods=['POST'])
def handle_analysis():
    """Run market analysis"""
    data = request.json
    granularity = data.get('granularity', config['granularity'])
    return jsonify(run_analysis(granularity))

@app.route('/api/order', methods=['POST'])
def handle_order():
    """Place market order"""
    data = request.json
    side = data.get('side')
    if side not in ['LONG', 'SHORT']:
        return jsonify({"status": "Error", "message": "Invalid side parameter"}), 400
    return jsonify(place_market_order(side))

@app.route('/api/close-positions', methods=['POST'])
def handle_close_positions():
    """Close all positions"""
    return jsonify(close_positions())

@app.route('/api/auto-trading', methods=['POST'])
def handle_auto_trading():
    """Toggle auto-trading"""
    return jsonify(toggle_auto_trading())

@app.route('/api/cancel', methods=['POST'])
def handle_cancel():
    """Cancel current operation"""
    return jsonify(cancel_operation())

@app.route('/api/events')
def events():
    """Server-sent events endpoint for real-time updates"""
    def generate():
        while True:
            try:
                # Get message from queue with timeout
                try:
                    message = analysis_queue.get(timeout=1)
                    yield f"data: {json.dumps(message)}\n\n"
                except queue.Empty:
                    # Send a ping to keep the connection alive
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
            except GeneratorExit:
                # Client disconnected
                break
            except Exception as e:
                print(f"Error in event stream: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                time.sleep(1)
    
    return Response(stream_with_context(generate()), 
                   mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive'})

@app.route('/api/trade-history')
def get_trade_history():
    """Get trade history"""
    return jsonify(trading_history)

@app.route('/api/check-orders', methods=['POST'])
def handle_check_orders():
    """Check for open orders and positions"""
    # Create check orders thread
    check_thread = threading.Thread(target=_check_orders_thread)
    check_thread.daemon = True
    check_thread.start()
    
    return jsonify({"status": "Checking orders initiated"})

def _check_orders_thread():
    """Background thread for checking orders"""
    try:
        # Notify that we're checking orders
        analysis_queue.put({
            "type": "message",
            "message": "Checking for open orders and positions..."
        })
        
        if coinbase_service:
            # Check for open orders
            has_open_orders, has_positions = check_open_orders_and_positions()
            
            if has_open_orders and has_positions:
                analysis_queue.put({
                    "type": "message",
                    "message": "Found open orders and positions."
                })
            elif has_open_orders:
                analysis_queue.put({
                    "type": "message",
                    "message": "Found open orders."
                })
            elif has_positions:
                analysis_queue.put({
                    "type": "message",
                    "message": "Found open positions."
                })
            else:
                analysis_queue.put({
                    "type": "message",
                    "message": "No open orders or positions found."
                })
        else:
            analysis_queue.put({
                "type": "error",
                "message": "CoinbaseService not initialized. Cannot check orders."
            })
    
    except Exception as e:
        analysis_queue.put({
            "type": "error",
            "message": f"Error checking orders: {str(e)}"
        })

@app.route('/api/current-price')
def get_current_price():
    """Get the current price for the selected product"""
    if not coinbase_service:
        return jsonify({"error": "CoinbaseService not initialized"}), 500
    
    try:
        # Get current price for the selected product
        product_id = config["product"]
        price_data = coinbase_service.get_btc_prices()
        
        if price_data and product_id in price_data:
            # Extract the price (average of bid and ask)
            bid_price = price_data[product_id].get("bid", 0)
            ask_price = price_data[product_id].get("ask", 0)
            price = (bid_price + ask_price) / 2 if bid_price and ask_price else 0
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return jsonify({
                "price": price,
                "timestamp": timestamp,
                "product": product_id
            })
        else:
            # If the specific product is not found, try to get any BTC price
            for key, value in price_data.items():
                if "BTC" in key:
                    bid_price = value.get("bid", 0)
                    ask_price = value.get("ask", 0)
                    price = (bid_price + ask_price) / 2 if bid_price and ask_price else 0
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    return jsonify({
                        "price": price,
                        "timestamp": timestamp,
                        "product": key,
                        "note": f"Using {key} price as fallback"
                    })
            
            return jsonify({"error": "No BTC price available"}), 404
    
    except Exception as e:
        print(f"Error getting current price: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize services
    initialize_coinbase_service()
    
    # Load trade history
    load_trade_history()
    
    # Start price updates
    start_price_updates()
    
    # Run the app
    app.run(debug=True, threaded=True) 