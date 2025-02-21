import customtkinter as ctk
import subprocess
import threading
import queue
from datetime import datetime
import json
import requests
import time
import re

class MarketAnalyzerUI:
    def __init__(self):
        # Set theme and color
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Crypto Market Analyzer")
        self.root.geometry("1460x930")  # Larger default size (width x height)
        self.root.minsize(800, 600)    # Larger minimum window size
        
        # Queue for communication between threads
        self.queue = queue.Queue()
        
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
        
        # Create main container with padding
        self.create_gui()
        
        # Start queue processing
        self.process_queue()
        
        # Start price updates
        self.start_price_updates()

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
        models = [
            ("O1 Mini", "o1_mini"),
            ("O3 Mini", "o3_mini"),
            ("DeepSeek", "deepseek"),
            ("Grok", "grok"),
            ("GPT-4o", "gpt4o")
        ]
        
        # Create model radio buttons with reduced padding
        for text, value in models:
            radio = ctk.CTkRadioButton(
                sidebar_container, 
                text=text, 
                value=value, 
                variable=self.model_var
            )
            radio.pack(pady=2)  # Reduced padding
            
        # Trading Options Section
        trading_frame = ctk.CTkFrame(sidebar_container)
        trading_frame.pack(pady=10, padx=5, fill="x")  # Reduced padding
        
        ctk.CTkLabel(trading_frame, text="Trading Options", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
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
        self.margin_var = ctk.StringVar(value="100")
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
            
        # Granularity section
        ctk.CTkLabel(sidebar_container, text="Time Frame:", font=ctk.CTkFont(weight="bold")).pack(pady=(20,10))
        
        # Analysis buttons with reduced height and padding
        self.five_min_btn = ctk.CTkButton(
            sidebar_container, 
            text="5 MINUTE", 
            command=lambda: self.run_analysis("FIVE_MINUTE"),
            height=32  # Reduced height
        )
        self.five_min_btn.pack(pady=3, padx=10, fill="x")
        
        self.one_hour_btn = ctk.CTkButton(
            sidebar_container, 
            text="1 HOUR", 
            command=lambda: self.run_analysis("ONE_HOUR"),
            height=32  # Reduced height
        )
        self.one_hour_btn.pack(pady=3, padx=10, fill="x")
        
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
        
        # Create main content area with output text
        main_content = ctk.CTkFrame(main_container)
        main_content.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # Output text area
        self.output_text = ctk.CTkTextbox(
            main_content,
            wrap="word",
            font=ctk.CTkFont(family="Courier", size=14)  # Slightly smaller font
        )
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)

    def update_leverage_label(self, value):
        """Update the leverage label when slider moves"""
        self.leverage_label.configure(text=f"{int(value)}x")

    def toggle_trading_options(self):
        """Enable/disable trading options based on execute trades checkbox"""
        state = "normal" if self.execute_trades_var.get() else "disabled"
        self.margin_entry.configure(state=state)
        self.leverage_slider.configure(state=state)
        self.limit_order_cb.configure(state=state)

    def clear_output(self):
        """Clear the output text area"""
        self.output_text.delete("1.0", "end")
        self.status_var.set("Ready")

    def run_analysis(self, granularity):
        """Run the market analysis with selected parameters"""
        # Disable buttons during analysis
        self.five_min_btn.configure(state="disabled")
        self.one_hour_btn.configure(state="disabled")
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
                    self.five_min_btn.configure(state="normal")
                    self.one_hour_btn.configure(state="normal")
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

    def run(self):
        try:
            self.root.mainloop()
        finally:
            # Ensure price updates are stopped and session is closed when the app closes
            self.stop_price_update_thread()
            self.session.close()

if __name__ == "__main__":
    app = MarketAnalyzerUI()
    app.run() 