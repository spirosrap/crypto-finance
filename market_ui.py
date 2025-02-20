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
        self.root.geometry("1000x800")
        
        # Queue for communication between threads
        self.queue = queue.Queue()
        
        # Initialize price tracking
        self.current_price = "-.--"
        self.last_price_update = None
        self.price_update_thread = None
        self.stop_price_updates = False
        
        # ANSI escape sequence pattern
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        
        # Create main container with padding
        self.create_gui()
        
        # Start queue processing
        self.process_queue()
        
        # Start price updates
        self.start_price_updates()

    def create_gui(self):
        # Create left sidebar for controls
        sidebar = ctk.CTkFrame(self.root, width=250)
        sidebar.pack(side="left", fill="y", padx=10, pady=10)
        
        # Title in sidebar
        title = ctk.CTkLabel(sidebar, text="Market Analysis", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=(20,30))
        
        # Product selection
        ctk.CTkLabel(sidebar, text="Select Product:").pack(pady=(0,5))
        self.product_var = ctk.StringVar(value="BTC-USDC")
        products = ["BTC-USDC", "ETH-USDC", "DOGE-USDC", "SOL-USDC", "SHIB-USDC"]
        product_menu = ctk.CTkOptionMenu(sidebar, values=products, variable=self.product_var)
        product_menu.pack(pady=(0,20))
        
        # Add price display after product selection
        self.price_frame = ctk.CTkFrame(sidebar)
        self.price_frame.pack(pady=(0,20), padx=10, fill="x")
        
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
        
        # Model selection
        ctk.CTkLabel(sidebar, text="Select Model:").pack(pady=(0,5))
        self.model_var = ctk.StringVar(value="o1_mini")
        models = [
            ("O1 Mini", "o1_mini"),
            ("O3 Mini", "o3_mini"),
            ("DeepSeek", "deepseek"),
            ("Grok", "grok"),
            ("GPT-4o", "gpt4o")
        ]
        
        # Create model radio buttons
        for text, value in models:
            radio = ctk.CTkRadioButton(
                sidebar, 
                text=text, 
                value=value, 
                variable=self.model_var
            )
            radio.pack(pady=5)
            
        # Trading Options Section
        trading_frame = ctk.CTkFrame(sidebar)
        trading_frame.pack(pady=20, padx=10, fill="x")
        
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
        ctk.CTkLabel(sidebar, text="Time Frame:", font=ctk.CTkFont(weight="bold")).pack(pady=(20,10))
        
        # Analysis buttons
        self.five_min_btn = ctk.CTkButton(
            sidebar, 
            text="5 MINUTE", 
            command=lambda: self.run_analysis("FIVE_MINUTE"),
            height=40
        )
        self.five_min_btn.pack(pady=5, padx=20, fill="x")
        
        self.one_hour_btn = ctk.CTkButton(
            sidebar, 
            text="1 HOUR", 
            command=lambda: self.run_analysis("ONE_HOUR"),
            height=40
        )
        self.one_hour_btn.pack(pady=5, padx=20, fill="x")
        
        # Close Positions button
        self.close_positions_btn = ctk.CTkButton(
            sidebar,
            text="Close All Positions",
            command=self.close_positions,
            fg_color="#b22222",  # Dark red color
            hover_color="#8b0000",  # Darker red on hover
            height=40  # Make it the same height as analysis buttons
        )
        self.close_positions_btn.pack(pady=(20,10), padx=20, fill="x")
        
        # Clear button at bottom of sidebar
        self.clear_btn = ctk.CTkButton(
            sidebar, 
            text="Clear Output", 
            command=self.clear_output,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "#DCE4EE")
        )
        self.clear_btn.pack(pady=(0,20), padx=20, fill="x")
        
        # Status indicator
        self.status_var = ctk.StringVar(value="Ready")
        self.status_label = ctk.CTkLabel(
            sidebar, 
            textvariable=self.status_var,
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(pady=(20,0))
        
        # Create main content area
        main_content = ctk.CTkFrame(self.root)
        main_content.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Output text area with custom styling
        self.output_text = ctk.CTkTextbox(
            main_content,
            wrap="word",
            font=ctk.CTkFont(family="Courier", size=16)
        )
        self.output_text.pack(fill="both", expand=True, padx=10, pady=10)

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
            
            # Restart price updates if they were running before
            if was_updating and (self.price_update_thread is None or not self.price_update_thread.is_alive()):
                self.start_price_updates()
            
        except Exception as e:
            self.queue.put(("append", f"\nError: {str(e)}\n"))
            self.queue.put(("status", "Error occurred"))
            self.queue.put(("enable_buttons", None))
            
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
            
        except Exception as e:
            self.queue.put(("append", f"\nError: {str(e)}\n"))
            self.queue.put(("status", "Error occurred"))
            self.queue.put(("enable_close_button", None))

    def start_price_updates(self):
        """Start the price update thread"""
        self.stop_price_updates = False
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
        while not self.stop_price_updates:
            try:
                # Get current product from dropdown
                product = self.product_var.get()
                
                # Fetch price from Coinbase API
                response = requests.get(f"https://api.coinbase.com/v2/prices/{product}/spot")
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
                
            except Exception as e:
                print(f"Error updating price: {e}")
            
            # Wait 1 second before next update
            time.sleep(1)

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
                
        except queue.Empty:
            pass
        finally:
            # Schedule next queue check
            self.root.after(100, self.process_queue)

    def run(self):
        try:
            self.root.mainloop()
        finally:
            # Ensure price updates are stopped when the app closes
            self.stop_price_update_thread()

if __name__ == "__main__":
    app = MarketAnalyzerUI()
    app.run() 