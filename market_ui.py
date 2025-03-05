import customtkinter as ctk
import subprocess
import threading
import queue
from datetime import datetime
import json
import requests
import time
import re
import os
import traceback
from coinbaseservice import CoinbaseService
import select
import tkinter.messagebox as messagebox

class MarketAnalyzerUI:
    def __init__(self):
        # Set theme and color
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Crypto Market Analyzer")
        self.root.geometry("1460x1115")  # Larger default size (width x height)
        self.root.minsize(800, 600)    # Larger minimum window size
        
        # Initialize granularity variable first
        self.granularity_var = ctk.StringVar(value="ONE_HOUR")
        
        # Queue for communication between threads
        self.queue = queue.Queue()
        
        # Model retraining tracking
        self.model_retrain_interval = 10 * 24 * 60 * 60  # 10 days in seconds
        self.last_train_time = self.get_last_train_time()
        self.retrain_timer_active = True
        
        # Add training timer variables
        self.training_start_time = None
        self.training_timer_active = False
        self.estimated_training_times = {
            'ONE_MINUTE': 60,  # 1 minute
            'FIVE_MINUTE': 180,  # 3 minutes
            'FIFTEEN_MINUTE': 300,  # 5 minutes
            'THIRTY_MINUTE': 600,  # 10 minutes
            'ONE_HOUR': 1200,  # 20 minutes
        }
        
        # Initialize price tracking
        self.current_price = "-.--"
        self.last_price_update = None
        self.price_update_thread = None
        self.stop_price_updates = False
        self.price_update_errors = 0  # Track consecutive errors
        self.max_price_update_errors = 3  # Maximum consecutive errors before restarting
        
        # Add current process tracking
        self.current_process = None
        
        # Create a session for connection pooling
        self.session = requests.Session()
        self.session.mount('https://', requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        ))
        
        # ANSI escape sequence pattern
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        
        # Add auto-trading variables
        self.auto_trading = False
        self.auto_trading_thread = None
        self.last_trade_time = None
        
        # Add trade output tracking
        self.current_trade_output = ""
        self.trade_in_progress = False
        
        # Add trade history variable
        self.trade_history = []  # List of trade results (win/loss)
        
        # Create main container with padding
        self.create_gui()
        
        # Start queue processing
        self.process_queue()
        
        # Start price updates
        self.start_price_updates()
        
        # Monitor threads
        self.monitor_threads()
        
        # Load trade history if available
        self.load_trade_history()

    def create_gui(self):
        # Create main container that fills the window
        main_container = ctk.CTkFrame(self.root)
        main_container.pack(fill="both", expand=True)
        
        # Create sidebar container
        sidebar_container = ctk.CTkFrame(
            main_container,
            width=200,  # Reduced width
        )
        sidebar_container.pack(side="left", fill="y", padx=5, pady=5)
        sidebar_container.pack_propagate(False)  # Prevent the frame from shrinking
        
        # Title in sidebar
        title = ctk.CTkLabel(
            sidebar_container,
            text="Market Analysis",
            font=ctk.CTkFont(size=18, weight="bold")  # Slightly smaller font
        )
        title.pack(pady=(10,15))  # Reduced padding
        
        # Product selection
        ctk.CTkLabel(sidebar_container, text="Select Product:").pack(pady=(0,2))
        self.product_var = ctk.StringVar(value="BTC-USDC")
        products = ["BTC-USDC", "ETH-USDC", "DOGE-USDC", "SOL-USDC", "SHIB-USDC"]
        product_menu = ctk.CTkOptionMenu(sidebar_container, values=products, variable=self.product_var)
        product_menu.pack(pady=(0,10))  # Reduced padding
        
        # Price display frame
        self.price_frame = ctk.CTkFrame(sidebar_container)
        self.price_frame.pack(pady=(0,10), padx=5, fill="x")  # Reduced padding
        
        ctk.CTkLabel(self.price_frame, text="Current Price:", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(5,0))
        
        self.price_label = ctk.CTkLabel(
            self.price_frame,
            text="-.--",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.price_label.pack(pady=(0,5))
        
        self.price_time_label = ctk.CTkLabel(
            self.price_frame,
            text="Last update: Never",
            font=ctk.CTkFont(size=10)
        )
        self.price_time_label.pack(pady=(0,5))
        
        # Model selection with reduced padding
        ctk.CTkLabel(sidebar_container, text="Select Model:").pack(pady=(0,2))
        self.model_var = ctk.StringVar(value="o1_mini")
        
        # Create model mapping for display names
        model_display_names = {
            "o1_mini": "o1_mini",
            "o3_mini": "o3_mini",
            "deepseek": "deepseek",
            "grok": "grok",
            "gpt4o": "gpt4o"
        }
        
        models = list(model_display_names.keys())
        
        # Create model dropdown menu
        model_menu = ctk.CTkOptionMenu(
            sidebar_container,
            values=[model_display_names[m] for m in models],
            variable=self.model_var,
            command=lambda x: self.model_var.set(next(k for k, v in model_display_names.items() if v == x))
        )
        model_menu.set(model_display_names[self.model_var.get()])
        model_menu.pack(pady=(0,10))
        
        # Granularity section
        ctk.CTkLabel(sidebar_container, text="Time Frame:", font=ctk.CTkFont(weight="bold")).pack(pady=(20,10))
        
        # Add granularity dropdown
        granularities = ["ONE_HOUR", "THIRTY_MINUTE", "FIFTEEN_MINUTE", "FIVE_MINUTE", "ONE_MINUTE"]
        granularity_menu = ctk.CTkOptionMenu(
            sidebar_container,
            values=granularities,
            variable=self.granularity_var,
            command=self.update_training_time
        )
        granularity_menu.pack(pady=(0,5), padx=10, fill="x")
        
        # Add training time frame
        self.training_frame = ctk.CTkFrame(sidebar_container)
        self.training_frame.pack(pady=(0,10), padx=5, fill="x")
        
        # Create a single line for retrain countdown
        self.retrain_time_label = ctk.CTkLabel(
            self.training_frame,
            text="Retrain in: --:--:--",
            font=ctk.CTkFont(size=12)
        )
        self.retrain_time_label.pack(pady=5)
        
        # Progress bar for retraining countdown
        self.retrain_progress = ctk.CTkProgressBar(
            self.training_frame,
            mode="determinate",
            height=6
        )
        self.retrain_progress.pack(pady=(0,5), padx=5, fill="x")
        self.retrain_progress.set(0)
        
        # Start retrain countdown timer
        self.update_retrain_countdown()
        
        # Single analysis button
        self.analyze_btn = ctk.CTkButton(
            sidebar_container, 
            text="Run Analysis", 
            command=lambda: self.run_analysis(self.granularity_var.get()),
            height=32  # Reduced height
        )
        self.analyze_btn.pack(pady=3, padx=10, fill="x")
        
        # Close Positions button
        self.close_positions_btn = ctk.CTkButton(
            sidebar_container,
            text="Close All Positions",
            command=self.close_positions,
            fg_color="#b22222",
            hover_color="#8b0000",
            height=32  # Reduced height
        )
        self.close_positions_btn.pack(pady=(10,5), padx=10, fill="x")
        
        # Check Orders button
        self.check_orders_btn = ctk.CTkButton(
            sidebar_container,
            text="Check Open Orders",
            command=self.check_open_orders_and_positions_ui,
            fg_color="#4682B4",  # Steel Blue
            hover_color="#36648B",  # Dark Steel Blue
            height=32  # Reduced height
        )
        self.check_orders_btn.pack(pady=(5,10), padx=10, fill="x")
        
        # Control buttons at bottom
        button_frame = ctk.CTkFrame(sidebar_container)
        button_frame.pack(fill="x", pady=5, padx=5)
        
        # Clear and Cancel buttons side by side
        self.clear_btn = ctk.CTkButton(
            button_frame, 
            text="Clear", 
            command=self.clear_output,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "#DCE4EE"),
            width=90,
            height=32
        )
        self.clear_btn.pack(side="left", padx=2)
        
        self.cancel_btn = ctk.CTkButton(
            button_frame, 
            text="Cancel", 
            command=self.cancel_operation,
            fg_color="#FF4B4B",
            hover_color="#CC3C3C",
            state="disabled",
            width=90,
            height=32
        )
        self.cancel_btn.pack(side="right", padx=2)
        
        # Status indicator
        self.status_var = ctk.StringVar(value="Ready")
        self.status_label = ctk.CTkLabel(
            sidebar_container, 
            textvariable=self.status_var,
            font=ctk.CTkFont(size=11)
        )
        self.status_label.pack(pady=(5,0))
        
        # Create center content area with output text
        center_content = ctk.CTkFrame(main_container)
        center_content.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Output text area
        self.output_text = ctk.CTkTextbox(
            center_content,
            wrap="word",
            font=ctk.CTkFont(family="Courier", size=14)  # Slightly smaller font
        )
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create right pane for risk management and trading options
        right_pane = ctk.CTkFrame(main_container, width=200)
        right_pane.pack(side="right", fill="y", padx=5, pady=5)
        right_pane.pack_propagate(False)  # Prevent the frame from shrinking
        
        # Title in right pane
        right_title = ctk.CTkLabel(
            right_pane,
            text="Trading Controls",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        right_title.pack(pady=(10,15))
        
        # Trading Options Section in right pane
        trading_frame = ctk.CTkFrame(right_pane)
        trading_frame.pack(pady=10, padx=5, fill="x")
        
        ctk.CTkLabel(trading_frame, text="Trading Options", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
        # Add Auto-Trading Button
        self.auto_trade_btn = ctk.CTkButton(
            trading_frame,
            text="Start Auto-Trading",
            command=self.toggle_auto_trading,
            fg_color="#4B0082",  # Indigo
            hover_color="#301934",  # Dark Purple
            height=32
        )
        self.auto_trade_btn.pack(pady=(0, 5), padx=10, fill="x")
        
        # Quick Market Order Buttons
        market_buttons_frame = ctk.CTkFrame(trading_frame)
        market_buttons_frame.pack(fill="x", pady=5)
        
        self.long_btn = ctk.CTkButton(
            market_buttons_frame,
            text="LONG",
            command=lambda: self.place_quick_market_order("BUY"),
            fg_color="#228B22",  # Forest Green
            hover_color="#006400",  # Dark Green
            width=90,
            height=32
        )
        self.long_btn.pack(side="left", padx=2, expand=True)
        
        self.short_btn = ctk.CTkButton(
            market_buttons_frame,
            text="SHORT",
            command=lambda: self.place_quick_market_order("SELL"),
            fg_color="#B22222",  # Fire Brick
            hover_color="#8B0000",  # Dark Red
            width=90,
            height=32
        )
        self.short_btn.pack(side="right", padx=2, expand=True)
        
        # TP/SL Percentage slider
        tp_sl_frame = ctk.CTkFrame(trading_frame)
        tp_sl_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(tp_sl_frame, text="TP/SL %:").pack(side="left", padx=5)
        self.tp_sl_var = ctk.DoubleVar(value=0.2)
        self.tp_sl_label = ctk.CTkLabel(tp_sl_frame, text="0.2%")
        self.tp_sl_label.pack(side="right", padx=5)
        
        self.tp_sl_slider = ctk.CTkSlider(
            trading_frame,
            from_=0.1,
            to=1.0,
            number_of_steps=18,  # 0.05% increments
            variable=self.tp_sl_var,
            command=self.update_tp_sl_label
        )
        self.tp_sl_slider.pack(pady=(0,5), padx=10, fill="x")
        
        # Execute trades checkbox
        self.execute_trades_var = ctk.BooleanVar(value=False)
        self.execute_trades_cb = ctk.CTkCheckBox(
            trading_frame,
            text="Execute Trades",
            variable=self.execute_trades_var,
            command=self.toggle_trading_options
        )
        self.execute_trades_cb.pack(pady=5)
        
        # Margin input
        margin_frame = ctk.CTkFrame(trading_frame)
        margin_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(margin_frame, text="Margin ($):").pack(side="left", padx=5)
        self.margin_var = ctk.StringVar(value="60")
        self.margin_entry = ctk.CTkEntry(
            margin_frame, 
            width=80,
            textvariable=self.margin_var
        )
        self.margin_entry.pack(side="right", padx=5)
        
        # Leverage slider
        ctk.CTkLabel(trading_frame, text=f"Leverage: 10x").pack(pady=(5,0))
        self.leverage_var = ctk.IntVar(value=10)
        self.leverage_slider = ctk.CTkSlider(
            trading_frame,
            from_=1,
            to=20,
            number_of_steps=19,
            variable=self.leverage_var,
            command=self.update_leverage_label
        )
        self.leverage_slider.pack(pady=(0,5), padx=10, fill="x")
        self.leverage_label = ctk.CTkLabel(trading_frame, text="10x")
        self.leverage_label.pack()
        
        # Limit order checkbox
        self.limit_order_var = ctk.BooleanVar(value=True)
        self.limit_order_cb = ctk.CTkCheckBox(
            trading_frame,
            text="Use Limit Orders",
            variable=self.limit_order_var
        )
        self.limit_order_cb.pack(pady=5)
        
        # Initially disable trading options
        self.toggle_trading_options()

    def update_leverage_label(self, value):
        """Update the leverage label when slider moves"""
        self.leverage_label.configure(text=f"{int(value)}x")

    def update_tp_sl_label(self, value):
        """Update the TP/SL percentage label when slider moves"""
        self.tp_sl_label.configure(text=f"{value:.1f}%")

    def toggle_trading_options(self):
        """Enable/disable trading options based on execute trades checkbox"""
        state = "normal" if self.execute_trades_var.get() else "disabled"
        self.margin_entry.configure(state=state)
        self.leverage_slider.configure(state=state)
        self.limit_order_cb.configure(state=state)
        self.long_btn.configure(state=state)
        self.short_btn.configure(state=state)
        self.tp_sl_slider.configure(state=state)

    def clear_output(self):
        """Clear the output text area"""
        self.output_text.delete("1.0", "end")
        self.status_var.set("Ready")

    def run_analysis(self, granularity):
        """Run the market analysis with selected parameters"""
        # Check if model needs retraining
        current_time = time.time()
        if current_time - self.last_train_time >= self.model_retrain_interval:
            self.queue.put(("append", "\nModel requires retraining before analysis.\n"))
            self.queue.put(("status", "Model needs retraining"))
            return
        
        # Check for open orders or positions
        has_open_orders, has_positions = self.check_for_open_orders_and_positions()
        
        if has_open_orders or has_positions:
            # Log that we can't run analysis with open orders/positions
            if has_open_orders and has_positions:
                self.queue.put(("append", "\nCannot run analysis: Found open orders and positions. Please close them first.\n"))
            elif has_open_orders:
                self.queue.put(("append", "\nCannot run analysis: Found open orders. Please close them first.\n"))
            else:
                self.queue.put(("append", "\nCannot run analysis: Found open positions. Please close them first.\n"))
            
            self.queue.put(("status", "Cannot run with open orders/positions"))
            return
            
        # Disable button during analysis
        self.analyze_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        
        self.status_var.set("Running analysis...")
        
        # Create and start thread
        thread = threading.Thread(
            target=self._run_analysis_thread,
            args=(granularity,)
        )
        thread.daemon = True
        thread.start()

    def _run_analysis_thread(self, granularity):
        """Thread function to run the analysis"""
        try:
            # Store current price update thread state
            was_updating = self.price_update_thread is not None and self.price_update_thread.is_alive()
            
            # Construct command
            model_flag = f"--use_{self.model_var.get()}"
            cmd = [
                "python",
                "prompt_market.py",
                "--product_id",
                self.product_var.get(),
                model_flag,
                "--granularity",
                granularity
            ]
            
            # Add trading options if enabled
            if self.execute_trades_var.get():
                cmd.append("--execute_trades")
                cmd.extend(["--margin", self.margin_var.get()])
                cmd.extend(["--leverage", str(self.leverage_var.get())])
                if self.limit_order_var.get():
                    cmd.append("--limit_order")
            
            # Run command and capture output
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"\n{'='*80}\n{timestamp} - Running analysis for {self.product_var.get()} ({granularity})\n"
            if self.execute_trades_var.get():
                header += f"Trading enabled - Margin: ${self.margin_var.get()}, Leverage: {self.leverage_var.get()}x, "
                header += f"{'Limit' if self.limit_order_var.get() else 'Market'} orders\n"
            header += f"{'='*80}\n"
            self.queue.put(("append", header))
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store the process
            self.current_process = process
            
            # Enable cancel button
            self.queue.put(("enable_cancel", None))
            
            # Variables to track trade output
            trade_output_buffer = ""
            capturing_trade_output = False
            order_placed = False
            
            # Read output in real-time with non-blocking I/O
            while True:
                reads = [process.stdout.fileno(), process.stderr.fileno()]
                ret = select.select(reads, [], [])
                for fd in ret[0]:
                    if fd == process.stdout.fileno():
                        output = process.stdout.readline()
                        if output:
                            self.queue.put(("append", output))
                            
                            # Start capturing trade output when we see a JSON recommendation
                            if "{\"BUY AT\":" in output or "{\"SELL AT\":" in output:
                                capturing_trade_output = True
                                trade_output_buffer = output
                            # Continue capturing trade output
                            elif capturing_trade_output:
                                trade_output_buffer += output
                            
                            # Check if a trade was executed
                            if "Order placed successfully" in output:
                                order_placed = True
                                
                                # Save the trade output to file
                                if trade_output_buffer:
                                    self.save_trade_output(trade_output_buffer)
                                    trade_output_buffer = ""
                                    capturing_trade_output = False
                                    
                    if fd == process.stderr.fileno():
                        error = process.stderr.readline()
                        if error:
                            self.queue.put(("append", f"\nErrors:\n{error}"))
                if process.poll() is not None:
                    break
            
            # Get any remaining output
            stdout, stderr = process.communicate(timeout=10)
            if stdout:
                self.queue.put(("append", stdout))
            if stderr:
                self.queue.put(("append", f"\nErrors:\n{stderr}"))
            
            self.queue.put(("status", "Ready"))
            self.queue.put(("enable_buttons", None))
            self.queue.put(("disable_cancel", None))
            
            # Clear current process
            self.current_process = None
            
            # Restart price updates if they were running before
            if was_updating and (self.price_update_thread is None or not self.price_update_thread.is_alive()):
                self.start_price_updates()
            
        except Exception as e:
            self.queue.put(("append", f"\nError: {str(e)}\n"))
            self.queue.put(("status", "Error occurred"))
            self.queue.put(("enable_buttons", None))
            self.queue.put(("disable_cancel", None))
            
            # Clear current process
            self.current_process = None
            
            # Restart price updates in case of error too
            if was_updating and (self.price_update_thread is None or not self.price_update_thread.is_alive()):
                self.start_price_updates()

    def close_positions(self):
        """Close all open positions"""
        self.status_var.set("Closing positions...")
        self.close_positions_btn.configure(state="disabled")
        
        # Create and start thread
        thread = threading.Thread(target=self._close_positions_thread)
        thread.daemon = True
        thread.start()
        
    def _close_positions_thread(self):
        """Thread function to close positions"""
        try:
            # Run close_positions.py
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"\n{'='*80}\n{timestamp} - Closing all open positions\n{'='*80}\n"
            self.queue.put(("append", header))
            
            process = subprocess.Popen(
                ["python", "close_positions.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store the process
            self.current_process = process
            
            # Enable cancel button
            self.queue.put(("enable_cancel", None))
            
            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.queue.put(("append", output))
            
            # Get any remaining output
            stdout, stderr = process.communicate()
            if stdout:
                self.queue.put(("append", stdout))
            if stderr:
                self.queue.put(("append", f"\nErrors:\n{stderr}"))
            
            self.queue.put(("status", "Ready"))
            self.queue.put(("enable_close_button", None))
            self.queue.put(("disable_cancel", None))
            
            # Clear current process
            self.current_process = None
            
        except Exception as e:
            self.queue.put(("append", f"\nError: {str(e)}\n"))
            self.queue.put(("status", "Error occurred"))
            self.queue.put(("enable_close_button", None))
            self.queue.put(("disable_cancel", None))
            
            # Clear current process
            self.current_process = None

    def start_price_updates(self):
        """Start the price update thread"""
        self.stop_price_updates = False
        self.price_update_errors = 0  # Reset error counter
        self.price_update_thread = threading.Thread(target=self._price_update_loop)
        self.price_update_thread.daemon = True
        self.price_update_thread.start()

    def stop_price_update_thread(self):
        """Stop the price update thread"""
        self.stop_price_updates = True
        if self.price_update_thread:
            self.price_update_thread.join()

    def _price_update_loop(self):
        """Background loop to update price"""
        retry_delay = 1  # Initial retry delay in seconds
        max_retry_delay = 30  # Maximum retry delay
        
        while not self.stop_price_updates:
            try:
                # Get current product from dropdown
                product = self.product_var.get()
                
                # Fetch price from Coinbase API with connection pooling
                response = self.session.get(
                    f"https://api.coinbase.com/v2/prices/{product}/spot",
                    timeout=(5, 15)  # (connect timeout, read timeout)
                )
                response.raise_for_status()  # Raise exception for bad status codes
                data = response.json()
                
                if 'data' in data and 'amount' in data['data']:
                    price = float(data['data']['amount'])
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # Format price with appropriate decimals
                    if price < 1:
                        formatted_price = f"${price:.6f}"
                    elif price < 100:
                        formatted_price = f"${price:.4f}"
                    else:
                        formatted_price = f"${price:,.2f}"
                    
                    # Update via queue to ensure thread safety
                    self.queue.put(("update_price", {
                        'price': formatted_price,
                        'time': f"Last update: {timestamp}"
                    }))
                    
                    # Reset error counter and retry delay on successful update
                    self.price_update_errors = 0
                    retry_delay = 1
                    
                else:
                    self.price_update_errors += 1
                    self.queue.put(("append", f"\nWarning: Invalid price data received for {product}"))
                    retry_delay = min(retry_delay * 2, max_retry_delay)
                
            except requests.exceptions.RequestException as e:
                self.price_update_errors += 1
                self.queue.put(("append", f"\nWarning: Network error updating price: {str(e)}"))
                retry_delay = min(retry_delay * 2, max_retry_delay)
                self.queue.put(("append", f"\nRetrying in {retry_delay} seconds..."))
            except Exception as e:
                self.price_update_errors += 1
                self.queue.put(("append", f"\nWarning: Error updating price: {str(e)}"))
                retry_delay = min(retry_delay * 2, max_retry_delay)
            
            # Check if we need to restart due to too many errors
            if self.price_update_errors >= self.max_price_update_errors:
                self.queue.put(("append", "\nRestarting price updates due to multiple errors..."))
                # Schedule a restart of price updates in the main thread
                self.root.after(0, self.restart_price_updates)
                break
            
            # Wait before next update, using the current retry delay
            time.sleep(retry_delay)

    def restart_price_updates(self):
        """Safely restart the price update thread"""
        try:
            # Stop existing thread if running
            if self.price_update_thread and self.price_update_thread.is_alive():
                self.stop_price_updates = True
                self.price_update_thread.join(timeout=2)
            
            # Reset error counter and flags
            self.price_update_errors = 0
            self.stop_price_updates = False
            
            # Start new thread
            self.price_update_thread = threading.Thread(target=self._price_update_loop)
            self.price_update_thread.daemon = True
            self.price_update_thread.start()
            
            self.queue.put(("append", "\nPrice updates successfully restarted"))
        except Exception as e:
            self.queue.put(("append", f"\nError restarting price updates: {str(e)}"))

    def _strip_ansi_codes(self, text):
        """Remove ANSI escape sequences from text"""
        return self.ansi_escape.sub('', text)

    def process_queue(self):
        """Process messages from the queue"""
        try:
            while True:
                action, data = self.queue.get_nowait()
                
                if action == "append":
                    # Strip ANSI codes before displaying
                    clean_text = self._strip_ansi_codes(data)
                    self.output_text.insert("end", clean_text)
                    self.output_text.see("end")
                elif action == "status":
                    self.status_var.set(data)
                elif action == "enable_buttons":
                    self.analyze_btn.configure(state="normal")
                    self.clear_btn.configure(state="normal")
                elif action == "enable_close_button":
                    self.close_positions_btn.configure(state="normal")
                elif action == "update_price":
                    self.price_label.configure(text=data['price'])
                    self.price_time_label.configure(text=data['time'])
                elif action == "enable_cancel":
                    self.cancel_btn.configure(state="normal")
                elif action == "disable_cancel":
                    self.cancel_btn.configure(state="disabled")
                
        except queue.Empty:
            pass
        finally:
            # Schedule next queue check
            self.root.after(100, self.process_queue)

    def cancel_operation(self):
        """Cancel the current running operation"""
        if self.current_process and self.current_process.poll() is None:
            # Process is still running, terminate it
            self.current_process.terminate()
            self.queue.put(("append", "\nOperation cancelled by user.\n"))
            self.queue.put(("status", "Operation cancelled"))
            
            # Re-enable buttons
            self.queue.put(("enable_buttons", None))
            self.queue.put(("enable_close_button", None))
            self.cancel_btn.configure(state="disabled")

    def place_quick_market_order(self, side: str):
        """Place a quick market order with configurable TP/SL percentage"""
        try:
            # Check for open orders or positions
            has_open_orders, has_positions = self.check_for_open_orders_and_positions()
            
            if has_open_orders or has_positions:
                # Log that we can't place order with open orders/positions
                if has_open_orders and has_positions:
                    self.queue.put(("append", "\nCannot place order: Found open orders and positions. Please close them first.\n"))
                elif has_open_orders:
                    self.queue.put(("append", "\nCannot place order: Found open orders. Please close them first.\n"))
                else:
                    self.queue.put(("append", "\nCannot place order: Found open positions. Please close them first.\n"))
                
                self.queue.put(("status", "Cannot trade with open orders/positions"))
                return
            
            # Get current price
            product = self.product_var.get()
            perp_product_map = {
                'BTC-USDC': 'BTC-PERP-INTX',
                'ETH-USDC': 'ETH-PERP-INTX',
                'DOGE-USDC': 'DOGE-PERP-INTX',
                'SOL-USDC': 'SOL-PERP-INTX',
                'SHIB-USDC': '1000SHIB-PERP-INTX'
            }
            
            # Define price precision for each product
            price_precision_map = {
                'BTC-PERP-INTX': 1,      # $1 precision for BTC
                'ETH-PERP-INTX': 0.1,    # $0.1 precision for ETH
                'DOGE-PERP-INTX': 0.0001, # $0.0001 precision for DOGE
                'SOL-PERP-INTX': 0.01,   # $0.01 precision for SOL
                '1000SHIB-PERP-INTX': 0.000001  # $0.000001 precision for SHIB
            }
            
            perp_product = perp_product_map.get(product)
            if not perp_product:
                self.queue.put(("append", f"\nError: Unsupported product for perpetual trading: {product}"))
                return
                
            # Get price precision for the product
            price_precision = price_precision_map.get(perp_product)
            if not price_precision:
                self.queue.put(("append", f"\nError: Unknown price precision for product: {perp_product}"))
                return
                
            # Get margin and leverage from UI
            try:
                margin = float(self.margin_var.get())
                leverage = self.leverage_var.get()
                tp_sl_pct = self.tp_sl_var.get() / 100  # Convert percentage to decimal
            except ValueError:
                self.queue.put(("append", "\nError: Invalid margin value"))
                return
                
            # Calculate position size
            size_usd = margin * leverage
            
            # Get current price for TP/SL calculation
            response = self.session.get(
                f"https://api.coinbase.com/v2/prices/{product}/spot",
                timeout=(5, 15)
            )
            response.raise_for_status()
            current_price = float(response.json()['data']['amount'])
            
            # Helper function to round to precision
            def round_to_precision(value, precision):
                return round(value / precision) * precision
            
            # Calculate TP/SL prices using the configured percentage and round to appropriate precision
            if side == "BUY":
                tp_price = round_to_precision(current_price * (1 + tp_sl_pct), price_precision)
                sl_price = round_to_precision(current_price * (1 - tp_sl_pct), price_precision)
            else:  # SELL
                tp_price = round_to_precision(current_price * (1 - tp_sl_pct), price_precision)
                sl_price = round_to_precision(current_price * (1 + tp_sl_pct), price_precision)
            
            # Construct and run command
            cmd = [
                'python', 'trade_btc_perp.py',
                '--product', perp_product,
                '--side', side,
                '--size', str(size_usd),
                '--leverage', str(leverage),
                '--tp', str(tp_price),
                '--sl', str(sl_price),
                '--no-confirm'
            ]
            
            # Add limit price if limit orders are enabled
            if self.limit_order_var.get():
                # For limit orders, set entry slightly better than current price
                limit_offset = 0.001  # 0.1% better entry
                if side == "BUY":
                    limit_price = round_to_precision(current_price * (1 - limit_offset), price_precision)
                else:  # SELL
                    limit_price = round_to_precision(current_price * (1 + limit_offset), price_precision)
                cmd.extend(['--limit', str(limit_price)])
            
            # Log the order details
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"\n{'='*80}\n{timestamp} - Placing {side} {'Limit' if self.limit_order_var.get() else 'Market'} Order\n"
            header += f"Product: {perp_product}\n"
            header += f"Size: ${size_usd} (Margin: ${margin}, Leverage: {leverage}x)\n"
            header += f"Current Price: ${current_price:.2f}\n"
            if self.limit_order_var.get():
                header += f"Limit Entry: ${limit_price:.6f}\n"
            header += f"Take Profit: ${tp_price:.6f} ({'+' if side == 'BUY' else '-'}{tp_sl_pct*100:.1f}%)\n"
            header += f"Stop Loss: ${sl_price:.6f} ({'-' if side == 'BUY' else '+'}{tp_sl_pct*100:.1f}%)\n"
            header += f"{'='*80}\n"
            self.queue.put(("append", header))
            self.queue.put(("status", f"Executing {side.lower()} {perp_product} trade..."))
            
            # Disable buttons during execution
            self.long_btn.configure(state="disabled")
            self.short_btn.configure(state="disabled")
            
            # Create and start thread
            thread = threading.Thread(
                target=self._run_market_order_thread,
                args=(cmd,)
            )
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.queue.put(("append", f"\nError placing market order: {str(e)}"))
            self.long_btn.configure(state="normal")
            self.short_btn.configure(state="normal")

    def _run_market_order_thread(self, cmd):
        """Thread function to run the market order command"""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store the process
            self.current_process = process
            
            # Enable cancel button
            self.queue.put(("enable_cancel", None))
            
            # Variables to track trade output
            trade_output_buffer = ""
            capturing_trade_output = False
            order_placed = False
            trade_completed = False
            
            # Read output in real-time with non-blocking I/O
            while True:
                reads = [process.stdout.fileno(), process.stderr.fileno()]
                ret = select.select(reads, [], [])
                for fd in ret[0]:
                    if fd == process.stdout.fileno():
                        output = process.stdout.readline()
                        if output:
                            self.queue.put(("append", output))
                            
                            # Start capturing trade output when we see a JSON recommendation or order summary
                            if "{\"BUY AT\":" in output or "{\"SELL AT\":" in output or "=== Order Summary ===" in output:
                                capturing_trade_output = True
                                trade_output_buffer = output
                            # Continue capturing trade output
                            elif capturing_trade_output:
                                trade_output_buffer += output
                            
                            # Update status based on output
                            if "Order placed successfully" in output:
                                self.queue.put(("status", "Order placed successfully"))
                                order_placed = True
                                
                                # Save the trade output to file
                                if trade_output_buffer:
                                    self.save_trade_output(trade_output_buffer)
                        
                        # Check for trade completion indicators
                        if "Take profit hit" in output or "TP hit" in output:
                            self.queue.put(("status", "Trade completed - Take Profit hit"))
                            trade_completed = True
                            # Record as a win
                            self.record_trade_result("win")
                        elif "Stop loss hit" in output or "SL hit" in output:
                            self.queue.put(("status", "Trade completed - Stop Loss hit"))
                            trade_completed = True
                            # Record as a loss
                            self.record_trade_result("loss")
                        elif "Position closed" in output:
                            # Try to determine if it was a win or loss
                            result = self.detect_trade_result(trade_output_buffer)
                            if result:
                                self.record_trade_result(result)
                                trade_completed = True
                
                # Get any remaining output
                stdout, stderr = process.communicate()
                if stdout:
                    output += stdout
                    self.queue.put(("append", stdout))
                    
                    # Check for trade completion in final output if not already detected
                    if not trade_completed and order_placed:
                        result = self.detect_trade_result(output)
                        if result:
                            self.record_trade_result(result)
                            trade_completed = True
                            
                if stderr:
                    self.queue.put(("append", f"\nErrors:\n{stderr}"))
                
                # Clear current process
                self.current_process = None
                
                # Wait based on granularity before next analysis
                wait_minutes = {
                    'ONE_MINUTE': 0.3,  # Check every 20 seconds
                    'FIVE_MINUTE': 2,   # Check every 2 minutes
                    'FIFTEEN_MINUTE': 2, # Check every 2 minutes (changed from 5)
                    'THIRTY_MINUTE': 10, # Check every 10 minutes
                    'ONE_HOUR': 20      # Check every 20 minutes
                }.get(self.granularity_var.get(), 20)  # Default to 20 minutes
                
                # For sub-minute intervals, adjust the sleep time
                if wait_minutes < 1:
                    time.sleep(wait_minutes * 60)  # Convert to seconds
                else:
                    # Wait in 1-minute intervals to allow for clean stopping
                    for _ in range(int(wait_minutes)):
                        if not self.auto_trading:
                            return
                        time.sleep(60)  # 1 minute intervals
                    
                    # Handle any remaining partial minute
                    remaining_seconds = (wait_minutes - int(wait_minutes)) * 60
                    if remaining_seconds > 0:
                        time.sleep(remaining_seconds)
                
        except Exception as e:
            self.queue.put(("append", f"\nError executing order: {str(e)}\n"))
            self.queue.put(("status", "Error occurred"))
            self.long_btn.configure(state="normal")
            self.short_btn.configure(state="normal")
            self.queue.put(("disable_cancel", None))
            
            # Clear current process
            self.current_process = None

    def toggle_auto_trading(self):
        """Toggle auto-trading on/off"""
        if not self.auto_trading:
            # Start auto-trading
            self.auto_trading = True
            self.auto_trade_btn.configure(
                text="Stop Auto-Trading",
                fg_color="#B22222"  # Fire Brick Red
            )
            granularity = self.granularity_var.get()
            wait_minutes = {
                'ONE_MINUTE': 0.3,  # Check every 20 seconds
                'FIVE_MINUTE': 2,   # Check every 2 minutes
                'FIFTEEN_MINUTE': 2, # Check every 2 minutes (changed from 5)
                'THIRTY_MINUTE': 10, # Check every 10 minutes
                'ONE_HOUR': 20      # Check every 20 minutes
            }.get(granularity, 20)  # Default to 20 minutes
            
            self.root.title(f"Crypto Market Analyzer [Auto-Trading ON - {granularity}]")
            self.queue.put(("append", f"\nAuto-trading started. Analyzing {granularity.lower().replace('_', ' ')} timeframe every {wait_minutes} minutes...\n"))
            self.queue.put(("append", "When an order is placed, auto-trading will pause until the order/position is closed, then resume automatically.\n"))
            
            # Start the auto-trading thread
            self.auto_trading_thread = threading.Thread(target=self._auto_trading_loop)
            self.auto_trading_thread.daemon = True
            self.auto_trading_thread.start()
        else:
            # Stop auto-trading
            self.auto_trading = False
            self.auto_trade_btn.configure(
                text="Start Auto-Trading",
                fg_color="#4B0082"  # Back to Indigo
            )
            self.root.title("Crypto Market Analyzer")
            self.queue.put(("append", "\nAuto-trading stopped.\n"))
            
            if self.auto_trading_thread:
                self.auto_trading_thread.join(timeout=1)
                self.auto_trading_thread = None

    def is_trading_allowed(self):
        """Check if trading is allowed based on current time"""
        current_time = datetime.now()
        
        # Check if it's weekend (5 = Saturday, 6 = Sunday)
        if current_time.weekday() >= 5:
            return False
            
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_time_float = current_hour + current_minute / 60.0
        
        # Trading is allowed from 17:00 (5 PM) to 14:30 (2:30 PM)
        if current_time_float >= 14.5 and current_time_float < 17.0:  # 14:30 (2:30 PM) to 5:00 PM
            return False
        return True

    def _auto_trading_loop(self):
        """Background loop for auto-trading"""
        while self.auto_trading:
            try:
                # Check if trading is allowed based on time
                current_time = datetime.now()
                if not self.is_trading_allowed():
                    if not hasattr(self, '_trading_paused_logged'):
                        if current_time.weekday() >= 5:
                            self.queue.put(("append", "\nTrading paused: Weekend trading is not allowed. Will resume on Monday at 5:00 PM.\n"))
                        else:
                            self.queue.put(("append", "\nTrading paused: Current time is outside trading hours (5:00 PM - 14:30 PM). Will resume at 5:00 PM.\n"))
                        self._trading_paused_logged = True
                    time.sleep(60)  # Check every minute
                    continue
                else:
                    # Reset the logged flag when we're out of the pause period
                    if hasattr(self, '_trading_paused_logged'):
                        del self._trading_paused_logged
                        self.queue.put(("append", "\nTrading resumed: Current time is within trading hours (5:00 PM - 14:30 PM).\n"))

                # Check if model needs retraining
                current_time = time.time()
                if current_time - self.last_train_time >= self.model_retrain_interval:
                    self.queue.put(("append", "\nModel requires retraining. Auto-trading will stop.\n"))
                    self.root.after(0, self.toggle_auto_trading)  # Stop auto-trading
                    return
                
                # Check for open orders or positions
                has_open_orders, has_positions = self.check_for_open_orders_and_positions()
                
                if has_open_orders or has_positions:
                    # Log that we're waiting for orders to close
                    if has_open_orders and has_positions:
                        self.queue.put(("append", "\nFound open orders and positions. Waiting for them to close before continuing...\n"))
                    elif has_open_orders:
                        self.queue.put(("append", "\nFound open orders. Waiting for them to close before continuing...\n"))
                    else:
                        self.queue.put(("append", "\nFound open positions. Waiting for them to close before continuing...\n"))
                    
                    # Wait for a shorter interval before checking again
                    time.sleep(60)  # Check every minute
                    continue
                
                # Get current granularity
                granularity = self.granularity_var.get()
                
                # Run analysis with current granularity
                self.queue.put(("append", f"\nRunning automated {granularity.lower().replace('_', ' ')} timeframe analysis...\n"))
                
                # Create and run the analysis process
                cmd = [
                    "python",
                    "prompt_market.py",
                    "--product_id",
                    self.product_var.get(),
                    f"--use_{self.model_var.get()}",
                    "--granularity",
                    granularity,
                    "--execute_trades",
                    "--margin",
                    self.margin_var.get(),
                    "--leverage",
                    str(self.leverage_var.get())
                ]
                
                if self.limit_order_var.get():
                    cmd.append("--limit_order")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Store the process
                self.current_process = process
                
                # Read output and check for trade execution
                output = ""
                order_placed = False
                trade_output_buffer = ""
                capturing_trade_output = False
                trade_completed = False
                
                while True:
                    line = process.stdout.readline()
                    if line == '' and process.poll() is not None:
                        break
                    if line:
                        output += line
                        self.queue.put(("append", line))
                        
                        # Start capturing trade output when we see a JSON recommendation
                        if "{\"BUY AT\":" in line or "{\"SELL AT\":" in line:
                            capturing_trade_output = True
                            trade_output_buffer = line
                        # Continue capturing trade output
                        elif capturing_trade_output:
                            trade_output_buffer += line
                        
                        # Check if a trade was executed
                        if "Order placed successfully" in line:
                            self.queue.put(("append", "\nTrade executed! Waiting for order to close before continuing auto-trading...\n"))
                            order_placed = True
                            
                            # Save the trade output to file
                            if trade_output_buffer:
                                self.save_trade_output(trade_output_buffer)
                        
                        # Check for trade completion indicators
                        if "Take profit hit" in line or "TP hit" in line:
                            self.queue.put(("status", "Trade completed - Take Profit hit"))
                            trade_completed = True
                            # Record as a win
                            self.record_trade_result("win")
                        elif "Stop loss hit" in line or "SL hit" in line:
                            self.queue.put(("status", "Trade completed - Stop Loss hit"))
                            trade_completed = True
                            # Record as a loss
                            self.record_trade_result("loss")
                        elif "Position closed" in line:
                            # Try to determine if it was a win or loss
                            result = self.detect_trade_result(trade_output_buffer)
                            if result:
                                self.record_trade_result(result)
                                trade_completed = True
                
                # Get any remaining output
                stdout, stderr = process.communicate()
                if stdout:
                    output += stdout
                    self.queue.put(("append", stdout))
                    
                    # Check for trade completion in final output if not already detected
                    if not trade_completed and order_placed:
                        result = self.detect_trade_result(output)
                        if result:
                            self.record_trade_result(result)
                            trade_completed = True
                            
                if stderr:
                    self.queue.put(("append", f"\nErrors:\n{stderr}"))
                
                # Clear current process
                self.current_process = None
                
                # Wait based on granularity before next analysis
                wait_minutes = {
                    'ONE_MINUTE': 0.3,  # Check every 20 seconds
                    'FIVE_MINUTE': 2,   # Check every 2 minutes
                    'FIFTEEN_MINUTE': 2, # Check every 2 minutes (changed from 5)
                    'THIRTY_MINUTE': 10, # Check every 10 minutes
                    'ONE_HOUR': 20      # Check every 20 minutes
                }.get(granularity, 20)  # Default to 20 minutes
                
                # For sub-minute intervals, adjust the sleep time
                if wait_minutes < 1:
                    time.sleep(wait_minutes * 60)  # Convert to seconds
                else:
                    # Wait in 1-minute intervals to allow for clean stopping
                    for _ in range(int(wait_minutes)):
                        if not self.auto_trading:
                            return
                        time.sleep(60)  # 1 minute intervals
                    
                    # Handle any remaining partial minute
                    remaining_seconds = (wait_minutes - int(wait_minutes)) * 60
                    if remaining_seconds > 0:
                        time.sleep(remaining_seconds)
                
            except Exception as e:
                self.queue.put(("append", f"\nError in auto-trading loop: {str(e)}\nTraceback:\n{traceback.format_exc()}\n"))
                # Don't exit the thread on error, just wait a bit and continue
                time.sleep(60)  # Wait 1 minute before retrying on error
                continue

    def get_truly_open_orders(self, service):
        """
        Get only truly open orders, filtering out any that are filled or canceled.
        
        Args:
            service: CoinbaseService instance
            
        Returns:
            list: List of truly open orders
        """
        try:
            orders_response = service.client.list_orders(status='OPEN')
            
            # Extract orders from response
            orders = []
            
            if isinstance(orders_response, dict):
                if 'orders' in orders_response:
                    orders = orders_response['orders']
            else:
                # Try to handle if orders_response is an object
                if hasattr(orders_response, 'orders'):
                    orders = orders_response.orders
            
            # Convert to list if needed
            if orders and not isinstance(orders, list):
                orders = [orders]
            
            # Filter out orders that are not actually open
            active_orders = []
            for order in orders:
                # Extract status from order
                status = None
                if isinstance(order, dict):
                    status = order.get('status')
                elif hasattr(order, 'status'):
                    status = order.status
                
                # Only count orders with status "OPEN"
                if status and status.upper() == 'OPEN':
                    active_orders.append(order)
            
            return active_orders
        except Exception as e:
            self.queue.put(("append", f"\nError getting open orders: {str(e)}\n"))
            return []

    def check_for_open_orders_and_positions(self):
        """
        Check if there are any open orders or positions.
        
        Returns:
            tuple: (has_open_orders, has_positions)
        """
        try:
            # Load API keys
            api_key, api_secret = self.load_api_keys()
            if not api_key or not api_secret:
                self.queue.put(("append", "\nError: Failed to load API keys\n"))
                return False, False
            
            # Initialize Coinbase service
            service = CoinbaseService(api_key, api_secret)
            
            # Check for open orders using the more robust method
            active_orders = self.get_truly_open_orders(service)
            has_open_orders = len(active_orders) > 0
            
            if has_open_orders:
                self.queue.put(("append", f"\nFound {len(active_orders)} truly open orders\n"))
            
            # Check for open positions
            has_positions = False
            try:
                # Get portfolio info for INTX (perpetuals)
                usd_balance, perp_position_size = service.get_portfolio_info(portfolio_type="INTX")
                
                has_positions = abs(perp_position_size) > 0
                if has_positions:
                    self.queue.put(("append", f"\nFound open position with size: {perp_position_size}\n"))
            except Exception as e:
                self.queue.put(("append", f"\nError checking for open positions: {str(e)}\n"))
            
            return has_open_orders, has_positions
            
        except Exception as e:
            self.queue.put(("append", f"\nError checking for open orders and positions: {str(e)}\n"))
            return False, False
    
    def load_api_keys(self):
        """Load API keys from environment variables or config file."""
        try:
            # First try to import from config.py
            from config import API_KEY_PERPS, API_SECRET_PERPS
            return API_KEY_PERPS, API_SECRET_PERPS
        except ImportError:
            # If config.py doesn't exist, try environment variables
            api_key = os.getenv('API_KEY_PERPS')
            api_secret = os.getenv('API_SECRET_PERPS')
            
            if not (api_key and api_secret):
                self.queue.put(("append", "\nAPI keys not found. Please set API_KEY_PERPS and API_SECRET_PERPS in config.py or as environment variables.\n"))
                return None, None
                
            return api_key, api_secret

    def update_training_time(self, _=None):
        """Update the training time label based on selected granularity"""
        # Update last training time for new granularity
        self.last_train_time = self.get_last_train_time()
        
        # Training days mapping (from ml_model.py)
        training_days = {
            'ONE_MINUTE': 10,
            'FIVE_MINUTE': 60,
            'FIFTEEN_MINUTE': 90,
            'THIRTY_MINUTE': 180,
            'ONE_HOUR': 365,
        }
        
        days = training_days.get(self.granularity_var.get(), 365)
        if days < 30:
            time_text = f"{days} days"
        elif days < 365:
            months = days // 30
            time_text = f"{months} months"
        else:
            years = days / 365
            time_text = f"{years:.1f} years"
            
        # Force an immediate update of the retrain countdown
        self.update_retrain_countdown()

    def find_model_file(self, granularity):
        """Find the model file path"""
        try:
            cwd = os.getcwd()
            granularity_formatted = granularity.lower().replace('_', ' ')
            
            model_paths = [
                f"models/ml_model_btc_usdc_{granularity_formatted.replace(' ', '_')}.joblib",
                f"models/btc_usdc_{granularity_formatted.replace(' ', '_')}_prediction_model.joblib",
                f"models/model_{granularity.lower()}.joblib",
                f"models/model_{granularity}.joblib",
                f"model_{granularity.lower()}.joblib",
                f"model_{granularity}.joblib"
            ]
            
            # Create models directory if it doesn't exist
            models_dir = os.path.join(cwd, "models")
            if not os.path.exists(models_dir):
                os.makedirs(models_dir, exist_ok=True)
            
            # Try each path
            for path in model_paths:
                full_path = os.path.join(cwd, path)
                if os.path.exists(full_path):
                    try:
                        # Test if file is readable
                        with open(full_path, 'rb') as f:
                            f.read(1)
                        return full_path
                    except (IOError, PermissionError):
                        continue
            
            return None
            
        except Exception:
            return None

    def get_last_train_time(self):
        """Get the last training time from a file or return current time if not found"""
        try:
            if not hasattr(self, 'granularity_var'):
                return time.time()
                
            model_path = self.find_model_file(self.granularity_var.get())
            if model_path:
                try:
                    return os.path.getmtime(model_path)
                except OSError:
                    pass
        except Exception:
            pass
        return time.time()  # Return current time if no model file found

    def format_time_ago(self, seconds):
        """Format time ago in a human readable format"""
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"

    def update_retrain_countdown(self):
        """Update the countdown display for model retraining"""
        if not self.retrain_timer_active:
            return
            
        current_time = time.time()
        elapsed = current_time - self.last_train_time
        remaining = max(0, self.model_retrain_interval - elapsed)
        
        # Update progress bar (inverted progress - starts full and empties)
        progress = max(0, min(1.0, remaining / self.model_retrain_interval))
        self.retrain_progress.set(progress)
        
        # Format remaining time
        days = int(remaining // (24 * 3600))
        hours = int((remaining % (24 * 3600)) // 3600)
        minutes = int((remaining % 3600) // 60)
        
        # Get model file info
        model_path = self.find_model_file(self.granularity_var.get())
        
        if not model_path:
            self.retrain_time_label.configure(text="Model not found", text_color="red")
        else:
            if days > 0:
                time_text = f"Retrain in: {days}d {hours:02d}h {minutes:02d}m"
            else:
                time_text = f"Retrain in: {hours:02d}h {minutes:02d}m"
                
            self.retrain_time_label.configure(text=time_text)
            
            # Change color based on time until retraining needed
            if remaining < 12 * 3600:  # Less than 12 hours
                self.retrain_time_label.configure(text_color="red")
            elif remaining < 24 * 3600:  # Less than 1 day
                self.retrain_time_label.configure(text_color="orange")
            else:
                self.retrain_time_label.configure(text_color=("gray10", "#DCE4EE"))
        
        # Schedule next update
        self.root.after(60000, self.update_retrain_countdown)  # Update every minute

    def check_open_orders_and_positions_ui(self):
        """Check for open orders and positions and display the results in the UI"""
        self.status_var.set("Checking for open orders and positions...")
        self.check_orders_btn.configure(state="disabled")
        
        # Create and start thread
        thread = threading.Thread(target=self._check_orders_thread)
        thread.daemon = True
        thread.start()
    
    def _check_orders_thread(self):
        """Thread function to check for open orders and positions"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"\n{'='*80}\n{timestamp} - Checking for open orders and positions\n{'='*80}\n"
            self.queue.put(("append", header))
            
            # Load API keys and initialize service
            api_key, api_secret = self.load_api_keys()
            if not api_key or not api_secret:
                self.queue.put(("append", "\nError: Failed to load API keys\n"))
                return
                
            service = CoinbaseService(api_key, api_secret)
            
            # Get detailed order information
            try:
                orders_response = service.client.list_orders(status='OPEN')
                
                # Extract orders from response
                orders = []
                if isinstance(orders_response, dict):
                    if 'orders' in orders_response:
                        orders = orders_response['orders']
                else:
                    # Try to handle if orders_response is an object
                    if hasattr(orders_response, 'orders'):
                        orders = orders_response.orders
                
                # Convert to list if needed
                if orders and not isinstance(orders, list):
                    orders = [orders]
                
                # Display detailed order information
                if orders:
                    self.queue.put(("append", f"\nFound {len(orders)} orders in response\n"))
                    self.queue.put(("append", "\nOrder details:\n"))
                    
                    # Count orders by status
                    status_counts = {}
                    
                    for i, order in enumerate(orders[:10]):  # Limit to first 10 orders to avoid flooding
                        # Extract order details
                        order_id = None
                        product_id = None
                        status = None
                        side = None
                        
                        if isinstance(order, dict):
                            order_id = order.get('order_id', 'Unknown')
                            product_id = order.get('product_id', 'Unknown')
                            status = order.get('status', 'Unknown')
                            side = order.get('side', 'Unknown')
                        else:
                            order_id = getattr(order, 'order_id', 'Unknown')
                            product_id = getattr(order, 'product_id', 'Unknown')
                            status = getattr(order, 'status', 'Unknown')
                            side = getattr(order, 'side', 'Unknown')
                        
                        # Count by status
                        if status not in status_counts:
                            status_counts[status] = 0
                        status_counts[status] += 1
                        
                        # Display order details
                        self.queue.put(("append", f"Order {i+1}: ID={order_id[:8] if order_id and len(order_id) > 8 else order_id}..., Product={product_id}, Status={status}, Side={side}\n"))
                    
                    # Display status summary
                    self.queue.put(("append", "\nOrder status summary:\n"))
                    for status, count in status_counts.items():
                        self.queue.put(("append", f"  {status}: {count} orders\n"))
                    
                    # Get truly open orders
                    active_orders = self.get_truly_open_orders(service)
                    self.queue.put(("append", f"\nTruly open orders: {len(active_orders)}\n"))
                    
                    # Display details of truly open orders
                    if active_orders:
                        self.queue.put(("append", "\nDetails of truly open orders:\n"))
                        for i, order in enumerate(active_orders):
                            # Extract order details
                            order_id = None
                            product_id = None
                            side = None
                            
                            if isinstance(order, dict):
                                order_id = order.get('order_id', 'Unknown')
                                product_id = order.get('product_id', 'Unknown')
                                side = order.get('side', 'Unknown')
                            else:
                                order_id = getattr(order, 'order_id', 'Unknown')
                                product_id = getattr(order, 'product_id', 'Unknown')
                                side = getattr(order, 'side', 'Unknown')
                            
                            # Display order details
                            self.queue.put(("append", f"Open Order {i+1}: ID={order_id[:8] if order_id and len(order_id) > 8 else order_id}..., Product={product_id}, Side={side}\n"))
                else:
                    self.queue.put(("append", "\nNo orders found in response\n"))
            except Exception as e:
                self.queue.put(("append", f"\nError getting detailed order information: {str(e)}\n"))
                self.queue.put(("append", f"Error details: {traceback.format_exc()}\n"))
            
            # Check for open orders and positions
            has_open_orders, has_positions = self.check_for_open_orders_and_positions()
            
            if not has_open_orders and not has_positions:
                self.queue.put(("append", "\nNo open orders or positions found. Ready to trade!\n"))
            
            # Re-enable button
            self.queue.put(("status", "Ready"))
            self.root.after(0, lambda: self.check_orders_btn.configure(state="normal"))
            
        except Exception as e:
            self.queue.put(("append", f"\nError checking for open orders and positions: {str(e)}\n"))
            self.queue.put(("append", f"Error details: {traceback.format_exc()}\n"))
            self.queue.put(("status", "Error occurred"))
            self.root.after(0, lambda: self.check_orders_btn.configure(state="normal"))

    def monitor_threads(self):
        """Monitor the health of threads and restart if necessary"""
        if self.price_update_thread and not self.price_update_thread.is_alive():
            self.queue.put(("append", "\nPrice update thread stopped unexpectedly. Restarting...\n"))
            self.start_price_updates()
        if self.auto_trading_thread and not self.auto_trading_thread.is_alive() and self.auto_trading:
            self.queue.put(("append", "\nAuto-trading thread stopped unexpectedly. Restarting...\n"))
            self.auto_trading_thread = threading.Thread(target=self._auto_trading_loop)
            self.auto_trading_thread.daemon = True
            self.auto_trading_thread.start()
        # Schedule next check
        self.root.after(60000, self.monitor_threads)  # Check every minute

    def save_trade_output(self, output_text):
        """Save trade output to a file"""
        try:
            # Create a timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Clean the output text (remove ANSI codes)
            clean_text = self._strip_ansi_codes(output_text)
            
            # Format the output with a header
            formatted_output = f"\n\n====== 🤖 AI Trading Recommendation ({timestamp}) ======\n{clean_text}"
            
            # Append to the trade_output.txt file
            with open("trade_output.txt", "a") as f:
                f.write(formatted_output)
                
            self.queue.put(("append", "\nTrade output saved to trade_output.txt\n"))
        except Exception as e:
            self.queue.put(("append", f"\nError saving trade output: {str(e)}\n"))

    def load_trade_history(self):
        """Load trade history from file"""
        try:
            history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_history.json")
            
            if not os.path.exists(history_file):
                self.queue.put(("append", "\nNo trade history file found. Starting fresh.\n"))
                return
                
            with open(history_file, "r") as f:
                data = json.load(f)
                
                # Load trade history
                self.trade_history = data.get("trades", [])
                
                # Log the loaded history
                self.queue.put(("append", f"\nLoaded trade history: {len(self.trade_history)} trades\n"))
                
        except Exception as e:
            self.queue.put(("append", f"\nError loading trade history: {str(e)}\n"))
            self.trade_history = []

    def save_trade_history(self):
        """Save trade history to file"""
        try:
            data = {
                "trades": self.trade_history,
                "last_updated": datetime.now().isoformat()
            }
            
            with open("trade_history.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.queue.put(("append", f"\nError saving trade history: {str(e)}\n"))

    def record_trade_result(self, result):
        """Record a trade result (win or loss)"""
        try:
            # Add result to history
            self.trade_history.append({
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "product": self.product_var.get()
            })
            
            # Save updated history
            self.save_trade_history()
            
            # Log the result
            self.queue.put(("append", f"\nTrade result recorded: {result.upper()}\n"))
            
        except Exception as e:
            self.queue.put(("append", f"\nError recording trade result: {str(e)}\n"))

    def detect_trade_result(self, output_text):
        """Detect if a trade was a win or loss from the output text"""
        # Clean the text
        clean_text = self._strip_ansi_codes(output_text.lower())
        
        # Check for win/loss indicators
        if "take profit hit" in clean_text or "tp hit" in clean_text or "profit:" in clean_text and "+" in clean_text:
            return "win"
        elif "stop loss hit" in clean_text or "sl hit" in clean_text or "loss:" in clean_text:
            return "loss"
        
        # If no clear indicator, return None
        return None

    def update_risk_level(self):
        """Update the risk level based on the current state"""
        # This method is no longer needed as risk management has been removed
        pass

    def run(self):
        try:
            self.root.mainloop()
        finally:
            # Stop auto-trading if active
            if self.auto_trading:
                self.toggle_auto_trading()
            # Ensure price updates are stopped and session is closed when the app closes
            self.stop_price_update_thread()
            self.session.close()

if __name__ == "__main__":
    app = MarketAnalyzerUI()
    app.run() 