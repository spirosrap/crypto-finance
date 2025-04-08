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
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import gc

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
        
        # Thread management
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.active_futures = {}  # Track active tasks by name
        self.thread_lock = threading.RLock()  # For thread-safety
        
        # Register app cleanup on exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        
        # Generate and set the application icon
        try:
            from app_icon import create_icon
            icon_path = create_icon()
            if os.path.exists(icon_path):
                # Define icon sizes to try in order of preference
                icon_sizes = ['256x256', '128x128', '64x64', '48x48', '32x32', '16x16']
                
                # Try to find the best available icon
                png_path = None
                for size in icon_sizes:
                    test_path = f'icons/market_analyzer_{size}.png'
                    if os.path.exists(test_path):
                        png_path = test_path
                        break
                
                if png_path:
                    if os.uname().sysname == 'Darwin':  # macOS
                        import tkinter as tk
                        icon_image = tk.PhotoImage(file=png_path)
                        self.root.iconphoto(True, icon_image)
                        
                        # Keep a reference to prevent garbage collection
                        self._icon_image = icon_image
                        
                        # For dock icon on macOS, use a different approach
                        try:
                            from Foundation import NSImage
                            from AppKit import NSApplication
                            image = NSImage.alloc().initByReferencingFile_(os.path.abspath(png_path))
                            NSApplication.sharedApplication().setApplicationIconImage_(image)
                        except Exception as e:
                            print(f"Warning: Could not set dock icon: {str(e)}")
                    else:  # Windows/Linux
                        try:
                            import tkinter as tk
                            icon_image = tk.PhotoImage(file=png_path)
                            self.root.iconphoto(True, icon_image)
                            self._icon_image = icon_image
                        except Exception as e:
                            print(f"Warning: Could not set window icon using PNG: {str(e)}")
                            # Fallback to ICO if PNG fails
                            if os.path.exists(icon_path):
                                self.root.iconbitmap(icon_path)
                elif os.path.exists(icon_path):
                    self.root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Warning: Could not set application icon: {str(e)}\nTraceback: {traceback.format_exc()}")
        
        # Queue for communication between threads
        self.queue = queue.Queue()
        
        # Load settings first
        self.settings = self.load_settings()
        
        # Initialize granularity variable after loading settings
        self.granularity_var = ctk.StringVar(value=self.settings["granularity"])
        
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
        
        # ANSI escape sequence pattern - enhanced version to catch more color codes
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\(B|\[[;\d]*m)')
        
        # Add auto-trading variables
        self.auto_trading = False
        self.auto_trading_thread = None
        self.last_trade_time = None
        
        # Add trade output tracking
        self.current_trade_output = ""
        self.trade_in_progress = False
        
        # Add trade history variable
        self.trade_history = []  # List of trade results (win/loss)
        
        # Add simplified trading variables
        self.simplified_trading = False
        self.simplified_trading_thread = None
        self.last_simplified_check = None
        
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
        
        # Start periodic cleanup to prevent memory buildup
        self.schedule_periodic_cleanup()

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
        self.product_var = ctk.StringVar(value=self.settings["product"])
        products = ["BTC-USDC", "ETH-USDC", "DOGE-USDC", "SOL-USDC", "SHIB-USDC"]
        product_menu = ctk.CTkOptionMenu(
            sidebar_container, 
            values=products, 
            variable=self.product_var,
            command=lambda x: (self.update_training_time(), self.save_settings())
        )
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
        self.model_var = ctk.StringVar(value=self.settings["model"])
        
        # Create model mapping for display names
        model_display_names = {
            "o1_mini": "o1_mini",
            "o3_mini": "o3_mini",
            "o3_mini_effort": "o3_mini_effort",
            "deepseek": "deepseek",
            "reasoner": "reasoner",
            "grok": "grok",
            "gpt4o": "gpt4o",
            "hyperbolic": "hyperbolic"
        }
        
        models = list(model_display_names.keys())
        
        # Create model dropdown menu
        model_menu = ctk.CTkOptionMenu(
            sidebar_container,
            values=[model_display_names[m] for m in models],
            variable=self.model_var,
            command=lambda x: (
                self.model_var.set(next(k for k, v in model_display_names.items() if v == x)),
                self.save_settings()
            )
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
            command=lambda x: (self.update_training_time(), self.save_settings())
        )
        granularity_menu.pack(pady=(0,5), padx=10, fill="x")
        
        # Add timeframe alignment option
        self.timeframe_alignment_frame = ctk.CTkFrame(sidebar_container)
        self.timeframe_alignment_frame.pack(pady=(5,5), padx=5, fill="x")
        
        ctk.CTkLabel(self.timeframe_alignment_frame, text="Timeframe Alignment", 
                    font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(5,0))
        
        # Add explanation
        ctk.CTkLabel(self.timeframe_alignment_frame, text="Analyzes multiple timeframes\nfor stronger signals", 
                    font=ctk.CTkFont(size=10)).pack(pady=(0,5))
        
        # Enable/disable timeframe alignment
        self.alignment_var = ctk.BooleanVar(value=self.settings.get("alignment_enabled", False))
        self.alignment_cb = ctk.CTkCheckBox(
            self.timeframe_alignment_frame,
            text="Enable Alignment",
            variable=self.alignment_var,
            command=self.save_settings
        )
        self.alignment_cb.pack(pady=(0,5))
        
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
        
        # Trade Status Frame - Add this new frame to display trade status information
        self.trade_status_frame = ctk.CTkFrame(sidebar_container)
        self.trade_status_frame.pack(pady=(10,5), padx=5, fill="x")
        
        ctk.CTkLabel(self.trade_status_frame, text="Trade Status", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(5,0))
        
        self.trade_status_label = ctk.CTkLabel(
            self.trade_status_frame,
            text="No updates",
            font=ctk.CTkFont(size=12),
            wraplength=180  # Ensure text wraps within the frame
        )
        self.trade_status_label.pack(pady=(0,5))
        
        self.trade_status_time = ctk.CTkLabel(
            self.trade_status_frame,
            text="Last update: Never",
            font=ctk.CTkFont(size=10)
        )
        self.trade_status_time.pack(pady=(0,5))
        
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
        
        # Add Simplified Trading Button
        self.simplified_trade_btn = ctk.CTkButton(
            trading_frame,
            text="Start Simplified Trading",
            command=self.toggle_simplified_trading,
            fg_color="#2E8B57",  # Sea Green
            hover_color="#1B4D3E",  # Dark Sea Green
            height=32
        )
        self.simplified_trade_btn.pack(pady=(0, 5), padx=10, fill="x")
        
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
        self.tp_sl_var = ctk.DoubleVar(value=self.settings["tp_sl"])
        self.tp_sl_label = ctk.CTkLabel(tp_sl_frame, text=f"{self.settings['tp_sl']}%")
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
        self.execute_trades_var = ctk.BooleanVar(value=self.settings["execute_trades"])
        self.execute_trades_cb = ctk.CTkCheckBox(
            trading_frame,
            text="Execute Trades",
            variable=self.execute_trades_var,
            command=lambda: (self.toggle_trading_options(), self.save_settings())
        )
        self.execute_trades_cb.pack(pady=5)
        
        # Margin input
        margin_frame = ctk.CTkFrame(trading_frame)
        margin_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(margin_frame, text="Margin ($):").pack(side="left", padx=5)
        self.margin_var = ctk.StringVar(value=self.settings["margin"])
        self.margin_entry = ctk.CTkEntry(
            margin_frame, 
            width=80,
            textvariable=self.margin_var
        )
        self.margin_entry.pack(side="right", padx=5)
        
        # Setup margin value tracking
        self.setup_margin_tracking()
        
        # Leverage slider
        ctk.CTkLabel(trading_frame, text=f"Leverage: {self.settings['leverage']}x").pack(pady=(5,0))
        self.leverage_var = ctk.IntVar(value=self.settings["leverage"])
        self.leverage_slider = ctk.CTkSlider(
            trading_frame,
            from_=1,
            to=20,
            number_of_steps=19,
            variable=self.leverage_var,
            command=self.update_leverage_label
        )
        self.leverage_slider.pack(pady=(0,5), padx=10, fill="x")
        self.leverage_label = ctk.CTkLabel(trading_frame, text=f"{self.settings['leverage']}x")
        self.leverage_label.pack()
        
        # Limit order checkbox
        self.limit_order_var = ctk.BooleanVar(value=self.settings["limit_order"])
        self.limit_order_cb = ctk.CTkCheckBox(
            trading_frame,
            text="Use Limit Orders",
            variable=self.limit_order_var,
            command=self.save_settings
        )
        self.limit_order_cb.pack(pady=5)
        
        # Initially disable trading options
        self.toggle_trading_options()

    def update_leverage_label(self, value):
        """Update the leverage label when slider moves"""
        self.leverage_label.configure(text=f"{int(value)}x")
        self.save_settings()

    def update_tp_sl_label(self, value):
        """Update the TP/SL percentage label when slider moves"""
        self.tp_sl_label.configure(text=f"{value:.1f}%")
        self.save_settings()

    def setup_margin_tracking(self):
        """Setup tracking for margin value changes"""
        self.margin_var.trace_add("write", lambda *args: self.save_settings())

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
        """Clear the output text widget and free memory"""
        try:
            # Delete all content from the text widget
            self.output_text.delete("1.0", "end")
            
            # Force garbage collection to free memory
            gc.collect()
            
            # Update status
            self.status_var.set("Output cleared")
            
            # Log the action
            print("Output cleared and memory freed")
        except Exception as e:
            print(f"Error clearing output: {str(e)}")

    def run_analysis(self, granularity):
        """Run the market analysis with selected parameters"""
        # Check if model needs retraining
        current_time = time.time()
        if current_time - self.last_train_time >= self.model_retrain_interval:
            self.queue.put(("append", "\nModel requires retraining before analysis.\n"))
            self.queue.put(("status", "Model needs retraining"))
            return
        
        # Check for open orders or positions - this critical check must be done before running any analysis
        has_open_orders, has_positions = self.check_for_open_orders_and_positions()
        
        if has_positions:
            # Block analysis if there are open positions
            self.queue.put(("append", "\nCannot run analysis: Found open positions. Please close them first.\n"))
            self.queue.put(("status", "Cannot run with open positions"))
            return
            
        if has_open_orders:
            # For normal analysis, we should still block if there are limit orders
            self.queue.put(("append", "\nCannot run analysis: Found open orders. Wait for bracket orders to be set or close existing orders first.\n"))
            self.queue.put(("status", "Cannot run analysis with open orders"))
            return
            
        # Disable button during analysis
        self.analyze_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        
        self.status_var.set("Running analysis...")
        
        # Submit the analysis task to the thread pool for better resource management
        self.submit_task(
            "market_analysis",
            self._run_analysis_thread,
            granularity
        )

    def _run_analysis_thread(self, granularity):
        """Thread function to run the analysis"""
        try:
            # Store current price update thread state
            was_updating = self.price_update_thread is not None and self.price_update_thread.is_alive()
            
            # Use a simplified approach to directly run the command without dependencies on the UI
            print("UI DEBUG: Starting direct command execution for analysis")
            
            # Construct command for timeframe analysis
            model_flag = f"--use_{self.model_var.get()}"
            cmd = [
                "python",
                "-u",  # Unbuffered output
                "prompt_market.py",
                "--product_id",
                self.product_var.get(),
                model_flag,
                "--granularity",
                granularity
            ]
            
            # Add timeframe alignment if enabled
            if self.alignment_var.get():
                self.queue.put(("append", "Timeframe alignment enabled. Running multi-timeframe analysis...\n"))
                # Run additional timeframes for alignment (only use ONE_HOUR, FIFTEEN_MINUTE, FIVE_MINUTE)
                alignment_granularities = ["ONE_HOUR", "FIFTEEN_MINUTE", "FIVE_MINUTE"]
                # Filter out current granularity if it's in the list
                alignment_granularities = [g for g in alignment_granularities if g != granularity]
                # Add remaining timeframes for alignment
                for i, align_gran in enumerate(alignment_granularities):
                    cmd.extend([f"--alignment_timeframe_{i+1}", align_gran])
            
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
            
            print("UI DEBUG: Creating process...")
            # Use a simplified approach for subprocess handling
            # Add a beautiful header with timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.queue.put(("append", "\n\n" + "★" * 80 + "\n"))
            self.queue.put(("append", f"   🚀 CRYPTO MARKET ANALYSIS - {self.product_var.get()} ({granularity}) - {current_time}\n"))
            self.queue.put(("append", "★" * 80 + "\n\n"))
            self.queue.put(("append", "Running AI-powered market analysis... please wait ⏳\n"))
            
            # Enable cancel button
            self.queue.put(("enable_cancel", None))
            
            # Run the process and capture output
            try:
                # Use direct process execution with stdout/stderr capture
                print("UI DEBUG: Starting subprocess direct execution")
                self.queue.put(("append", "Starting analysis subprocess...\n"))
                
                # Run subprocess directly
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                # Process the output - filter and display the relevant parts
                if process.stdout:
                    print(f"UI DEBUG: Got stdout from subprocess: {len(process.stdout)} chars")
                    
                    # Skip displaying all raw output - focus on the formatted recommendation
                    # self.queue.put(("append", "\n--- OUTPUT FROM ANALYSIS ---\n"))
                    # self.queue.put(("append", process.stdout))
                    
                    # Check for specific information in the output
                    if "Timeframe Alignment Analysis" in process.stdout:
                        # Extract the alignment section
                        alignment_start = process.stdout.find("Timeframe Alignment Analysis")
                        alignment_end = process.stdout.find("\nContinuing with primary timeframe", alignment_start)
                        
                        if alignment_start >= 0 and alignment_end >= 0:
                            alignment_info = process.stdout[alignment_start:alignment_end]
                            
                            # Display alignment info in a nice format
                            self.queue.put(("append", "\n\n" + "▓" * 60 + "\n"))
                            self.queue.put(("append", "           🔄 TIMEFRAME ALIGNMENT ANALYSIS 🔄\n"))
                            self.queue.put(("append", "▓" * 60 + "\n\n"))
                            
                            # Format the alignment info
                            lines = alignment_info.strip().split('\n')
                            for line in lines:
                                if "Primary timeframe:" in line:
                                    self.queue.put(("append", f"🕒 {line}\n"))
                                elif "Alignment timeframes:" in line:
                                    self.queue.put(("append", f"⏱️ {line}\n"))
                                elif "ALIGNMENT SCORE:" in line:
                                    score = line.split(':')[1].strip().split('/')[0]
                                    self.queue.put(("append", f"📊 Alignment Score: {score}/100\n"))
                                elif "EXCELLENT ALIGNMENT" in line:
                                    self.queue.put(("append", f"✅✅✅ Excellent Alignment - Very Strong Signal\n"))
                                elif "GOOD ALIGNMENT" in line:
                                    self.queue.put(("append", f"✅✅ Good Alignment - Strong Signal\n"))
                                elif "MODERATE ALIGNMENT" in line:
                                    self.queue.put(("append", f"✅ Moderate Alignment - Decent Signal\n"))
                                elif "POOR ALIGNMENT" in line:
                                    self.queue.put(("append", f"❌ Poor Alignment - Weak/Conflicting Signals\n"))
                            
                            self.queue.put(("append", "\n" + "▓" * 60 + "\n\n"))
                    
                    # Also check for JSON in the stdout
                    import re
                    json_pattern = r'({.*?})'
                    json_matches = re.findall(json_pattern, process.stdout, re.DOTALL)
                    
                    # Only display the first valid recommendation we find
                    if json_matches:
                        # Try to find a valid recommendation
                        found_recommendation = False
                        
                        for json_str in json_matches:
                            # Skip if we already found a valid recommendation
                            if found_recommendation:
                                break
                                
                            try:
                                import json
                                json_obj = json.loads(json_str)
                                
                                # Check if this is a trading recommendation
                                if ("SIGNAL_TYPE" in json_obj or "BUY AT" in json_obj or "SELL AT" in json_obj):
                                    # Mark that we found a valid recommendation
                                    found_recommendation = True
                                    
                                    # Get the signal type
                                    signal_type = json_obj.get('SIGNAL_TYPE', 'UNKNOWN')
                                    if signal_type == 'UNKNOWN':
                                        if 'SELL AT' in json_obj:
                                            signal_type = 'SELL'
                                        elif 'BUY AT' in json_obj:
                                            signal_type = 'BUY'
                                        elif 'HOLD' in json_obj:
                                            signal_type = 'HOLD'
                                    
                                    # Format based on signal type
                                    if signal_type == 'BUY':
                                        header_color = "🟢"  # Green
                                    elif signal_type == 'SELL':
                                        header_color = "🔴"  # Red
                                    else:  # HOLD
                                        header_color = "🟡"  # Yellow
                                    
                                    # Create a beautifully formatted recommendation
                                    self.queue.put(("append", "\n\n" + "═" * 60 + "\n"))
                                    self.queue.put(("append", f"   {header_color} {signal_type} RECOMMENDATION {header_color}\n"))
                                    self.queue.put(("append", "═" * 60 + "\n\n"))
                                    
                                    # Display the formatted information
                                    formatted_json = json.dumps(json_obj, indent=2)
                                    
                                    # Extract key information
                                    price = json_obj.get('PRICE', json_obj.get('BUY AT', json_obj.get('SELL AT', 'N/A')))
                                    probability = json_obj.get('PROBABILITY', 'N/A')
                                    confidence = json_obj.get('CONFIDENCE', 'N/A')
                                    r_r = json_obj.get('R/R_RATIO', 'N/A')
                                    market_regime = json_obj.get('MARKET_REGIME', 'N/A')
                                    reasoning = json_obj.get('REASONING', 'N/A')
                                    
                                    # Format the key info nicely
                                    self.queue.put(("append", f"📈 Price: ${price}\n"))
                                    self.queue.put(("append", f"📊 Probability: {probability}%\n"))
                                    self.queue.put(("append", f"🔍 Confidence: {confidence}\n"))
                                    
                                    if signal_type != 'HOLD':
                                        take_profit = json_obj.get('SELL BACK AT' if signal_type == 'BUY' else 'BUY BACK AT', 'N/A')
                                        stop_loss = json_obj.get('STOP LOSS', 'N/A')
                                        self.queue.put(("append", f"🎯 Take Profit: ${take_profit}\n"))
                                        self.queue.put(("append", f"🛑 Stop Loss: ${stop_loss}\n"))
                                        self.queue.put(("append", f"⚖️ Risk/Reward: {r_r}\n"))
                                    
                                    self.queue.put(("append", f"🌍 Market Regime: {market_regime}\n"))
                                    self.queue.put(("append", f"💡 Analysis: {reasoning}\n"))
                                    
                                    # End with a simple separator
                                    self.queue.put(("append", "\n" + "─" * 60 + "\n"))
                            except Exception as e:
                                print(f"UI DEBUG: Error parsing JSON: {str(e)}")
                
                if process.stderr:
                    print(f"UI DEBUG: Got stderr from subprocess: {len(process.stderr)} chars")
                    self.queue.put(("append", "\nErrors:\n"))
                    self.queue.put(("append", process.stderr))
                    
                print(f"UI DEBUG: Process completed with return code: {process.returncode}")
                
                # Add an elegant completion message
                self.queue.put(("append", "\n\n" + "✨" * 25 + "\n"))
                self.queue.put(("append", "   🏁 Analysis completed successfully! 🏁\n"))
                self.queue.put(("append", "✨" * 25 + "\n\n"))
                
                self.queue.put(("status", "Ready"))
                self.queue.put(("enable_buttons", None))
                self.queue.put(("disable_cancel", None))
            except Exception as e:
                self.queue.put(("append", f"\nError running subprocess: {str(e)}\n"))
                self.queue.put(("status", "Error"))
                self.queue.put(("enable_buttons", None))
                self.queue.put(("disable_cancel", None))
            
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
        
        # Submit the close positions task to the thread pool
        self.submit_task("close_positions", self._close_positions_thread)
        
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
        """Start the price update task in the thread pool"""
        with self.thread_lock:
            self.stop_price_updates = False
            self.price_update_errors = 0  # Reset error counter
            
            # Submit the task to the thread pool
            self.submit_task("price_updates", self._price_update_loop)
            self.queue.put(("status", "Price updates started"))

    def stop_price_update_thread(self):
        """Stop the price update task"""
        with self.thread_lock:
            self.stop_price_updates = True
            
            # Cancel the task if it exists
            if "price_updates" in self.active_futures:
                future = self.active_futures["price_updates"]
                if not future.done():
                    future.cancel()
                    self.queue.put(("status", "Price updates stopped"))

    def _price_update_loop(self):
        """Background loop to update price"""
        retry_delay = 1  # Initial retry delay in seconds
        max_retry_delay = 30  # Maximum retry delay
        
        # Use a session-level timeout to prevent hanging connections
        self.session.request_timeout = 10  # 10 seconds timeout
        
        while not self.stop_price_updates:
            try:
                # Get current product from dropdown (thread-safe access)
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
                    with self.thread_lock:
                        self.price_update_errors = 0
                        retry_delay = 1
                    
                else:
                    with self.thread_lock:
                        self.price_update_errors += 1
                        retry_delay = min(retry_delay * 2, max_retry_delay)
                    self.queue.put(("append", f"\nWarning: Invalid price data received for {product}"))
                
            except requests.exceptions.RequestException as e:
                with self.thread_lock:
                    self.price_update_errors += 1
                    retry_delay = min(retry_delay * 2, max_retry_delay)
                self.queue.put(("append", f"\nWarning: Network error updating price: {str(e)}"))
                self.queue.put(("append", f"\nRetrying in {retry_delay} seconds..."))
            except Exception as e:
                with self.thread_lock:
                    self.price_update_errors += 1
                    retry_delay = min(retry_delay * 2, max_retry_delay)
                self.queue.put(("append", f"\nWarning: Error updating price: {str(e)}"))
            
            # Check if we need to restart due to too many errors
            with self.thread_lock:
                too_many_errors = self.price_update_errors >= self.max_price_update_errors
                
            if too_many_errors:
                self.queue.put(("append", "\nRestarting price updates due to multiple errors..."))
                # Schedule a restart of price updates in the main thread
                self.root.after(0, self.restart_price_updates)
                break
            
            # Wait before next update, using the current retry delay
            # Use a more responsive approach with shorter sleep iterations
            # Sleep for 2 seconds between price updates to reduce resource usage
            sleep_interval = 0.5
            for _ in range(int(max(2, retry_delay) / sleep_interval)):
                if self.stop_price_updates:
                    return
                time.sleep(sleep_interval)

    def restart_price_updates(self):
        """Safely restart the price update task"""
        try:
            # Stop existing task if running
            self.stop_price_update_thread()
            
            # Reset error counter and flags
            with self.thread_lock:
                self.price_update_errors = 0
                self.stop_price_updates = False
            
            # Start new task in thread pool
            self.submit_task("price_updates", self._price_update_loop)
            
            self.queue.put(("append", "\nPrice updates successfully restarted"))
        except Exception as e:
            self.queue.put(("append", f"\nError restarting price updates: {str(e)}"))

    def _strip_ansi_codes(self, text):
        """Remove ANSI escape sequences from text"""
        # First ensure text is a string
        if not isinstance(text, str):
            try:
                text = str(text)
            except:
                return "[Non-string content]"
                
        # Print to console for debugging
        print(f"Raw output: {text}")
        
        # Make direct output more visible if it contains important information
        if "DIRECT TRADING RECOMMENDATION OUTPUT" in text:
            print("FOUND DIRECT TRADING RECOMMENDATION IN OUTPUT!")
            
        # Look for JSON content
        if text.strip().startswith('{') and text.strip().endswith('}'):
            print(f"Found JSON content: {text}")
        
        # Strip ANSI codes
        clean_text = self.ansi_escape.sub('', text)
        
        # Debug
        if clean_text.strip() and "AI Trading Recommendation" in clean_text:
            print(f"Found recommendation. Length: {len(clean_text)}")
            
        return clean_text

    def process_queue(self):
        """Process messages from the queue"""
        try:
            # Process a limited number of messages per cycle to prevent UI freezing
            message_count = 0
            max_messages_per_cycle = 10
            
            while message_count < max_messages_per_cycle:
                action, data = self.queue.get_nowait()
                message_count += 1
                
                if action == "append":
                    # Strip ANSI codes before displaying
                    clean_text = self._strip_ansi_codes(data)
                    
                    # Handle special sections - START/END recommendation content markers
                    if "START RECOMMENDATION CONTENT" in clean_text:
                        self.output_text.insert("end", "\n--- RECOMMENDATION STARTS HERE ---\n\n")
                        continue  # Skip this marker line
                    elif "END RECOMMENDATION CONTENT" in clean_text:
                        self.output_text.insert("end", "\n--- RECOMMENDATION ENDS HERE ---\n\n")
                        continue  # Skip this marker line
                        
                    # Special handling for direct recommendation output section
                    elif "DIRECT TRADING RECOMMENDATION OUTPUT" in clean_text:
                        self.output_text.insert("end", "\n" + "=" * 50 + "\n")
                        self.output_text.insert("end", "RAW TRADING RECOMMENDATION:\n")
                        self.output_text.insert("end", "=" * 50 + "\n")
                        continue
                    
                    # Check for JSON content and try to format it nicely
                    if clean_text.strip().startswith('{') and clean_text.strip().endswith('}'):
                        try:
                            # Parse JSON and pretty-print it
                            json_obj = json.loads(clean_text.strip())
                            formatted_json = json.dumps(json_obj, indent=2)
                            self.output_text.insert("end", formatted_json + "\n")
                            # Also output the JSON in a large, very visible section
                            self.output_text.insert("end", "\n" + "#" * 50 + "\n")
                            self.output_text.insert("end", "TRADING RECOMMENDATION:\n")
                            self.output_text.insert("end", "#" * 50 + "\n")
                            self.output_text.insert("end", formatted_json + "\n")
                            self.output_text.insert("end", "#" * 50 + "\n\n")
                        except json.JSONDecodeError:
                            # If JSON parsing fails, just display the original text
                            self.output_text.insert("end", clean_text)
                    else:
                        # Regular text
                        self.output_text.insert("end", clean_text)
                    
                    # Limit text widget content to prevent memory bloat
                    # Keep only the last ~100,000 characters (adjust as needed)
                    content = self.output_text.get("1.0", "end")
                    if len(content) > 100000:
                        # Delete the oldest content, keeping the last 100,000 characters
                        self.output_text.delete("1.0", f"end-{100000}c")
                        
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
                # Add new action type for trade status updates
                elif action == "update_trade_status":
                    self.trade_status_label.configure(text=data['status'])
                    self.trade_status_time.configure(text=data['time'])
                
        except queue.Empty:
            # Queue is empty, schedule next check
            pass
        except Exception as e:
            print(f"Error processing queue: {str(e)}")
            traceback.print_exc()
        
        # Schedule next queue check with a small delay to prevent UI freezing
        self.root.after(50, self.process_queue)

    def cancel_operation(self):
        """Cancel the current running operation"""
        # First try to cancel the current subprocess if one exists
        if self.current_process and self.current_process.poll() is None:
            # Process is still running, terminate it
            try:
                self.current_process.terminate()
                self.queue.put(("append", "\nSubprocess terminated by user.\n"))
            except Exception as e:
                self.queue.put(("append", f"\nError terminating subprocess: {str(e)}\n"))
        
        # Then cancel any running analysis or trading tasks in the thread pool
        with self.thread_lock:
            high_priority_tasks = ["market_analysis", "auto_trading"]
            cancelled = False
            
            for task_name in high_priority_tasks:
                if task_name in self.active_futures:
                    future = self.active_futures[task_name]
                    if not future.done():
                        future.cancel()
                        cancelled = True
                        self.queue.put(("append", f"\n{task_name.replace('_', ' ').title()} task cancelled.\n"))
            
            if cancelled or (self.current_process and self.current_process.poll() is None):
                # Re-enable buttons
                self.queue.put(("status", "Operation cancelled"))
                self.queue.put(("enable_buttons", None))
                self.queue.put(("enable_close_button", None))
                self.cancel_btn.configure(state="disabled")

    def place_quick_market_order(self, side: str):
        """Place a quick market order with configurable TP/SL percentage"""
        try:
            # Check for open orders or positions
            has_open_orders, has_positions = self.check_for_open_orders_and_positions()
            
            if has_positions:
                # Only block if there are open positions
                self.queue.put(("append", "\nCannot place order: Found open positions. Please close them first.\n"))
                self.queue.put(("status", "Cannot trade with open positions"))
                return
                
            if has_open_orders:
                # Only warn about open orders but allow setting bracket orders
                self.queue.put(("append", "\nWarning: Open orders detected. Proceeding with setting bracket order.\n"))
                self.queue.put(("status", "Setting bracket order with open limit order"))
                # Continue with the order placement
            
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
            
            # Submit the market order task to the thread pool
            self.submit_task(
                f"market_order_{side.lower()}",
                self._run_market_order_thread,
                cmd
            )
            
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
            output = ""  # Initialize output variable
            
            # Read output in real-time with non-blocking I/O
            while True:
                reads = [process.stdout.fileno(), process.stderr.fileno()]
                ret = select.select(reads, [], [])
                for fd in ret[0]:
                    if fd == process.stdout.fileno():
                        output_line = process.stdout.readline()
                        if output_line:
                            self.queue.put(("append", output_line))
                            output += output_line  # Accumulate output
                            
                            # Start capturing trade output when we see a JSON recommendation or order summary
                            if "{\"BUY AT\":" in output_line or "{\"SELL AT\":" in output_line or "=== Order Summary ===" in output_line:
                                capturing_trade_output = True
                                trade_output_buffer = output_line
                            # Continue capturing trade output
                            elif capturing_trade_output:
                                trade_output_buffer += output_line
                            
                            # Update status based on output
                            if "Order placed successfully" in output_line:
                                self.queue.put(("status", "Order placed successfully"))
                                order_placed = True
                                
                                # Save the trade output to file
                                if trade_output_buffer:
                                    self.save_trade_output(trade_output_buffer)
                        
                        # Check for trade completion indicators
                        if "Take profit hit" in output_line or "TP hit" in output_line:
                            self.queue.put(("status", "Trade completed - Take Profit hit"))
                            trade_completed = True
                            # Record as a win
                            self.record_trade_result("win")
                        elif "Stop loss hit" in output_line or "SL hit" in output_line:
                            self.queue.put(("status", "Trade completed - Stop Loss hit"))
                            trade_completed = True
                            # Record as a loss
                            self.record_trade_result("loss")
                        elif "Position closed" in output_line:
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
                    # Check if stderr contains actual error messages (not just INFO logs)
                    error_lines = [line for line in stderr.splitlines() if 'ERROR' in line or 'CRITICAL' in line]
                    if error_lines:
                        self.queue.put(("append", f"\nErrors:\n{stderr}"))
                    else:
                        # If it's just INFO logs, append them without the Errors header
                        self.queue.put(("append", stderr))
                
                # Clear current process
                self.current_process = None
                
                # Wait based on granularity before next analysis
                wait_minutes = {
                    'ONE_MINUTE': 0.3,  # Check every 20 seconds
                    'FIVE_MINUTE': 2,   # Check every 2 minutes
                    'FIFTEEN_MINUTE': 2, # Check every 2 minutes (changed from 5)
                    'THIRTY_MINUTE': 10, # Check every 10 minutes
                    'ONE_HOUR': 5      # Check every 5 minutes (changed from 20)
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
                'ONE_HOUR': 5      # Check every 5 minutes (changed from 20)
            }.get(granularity, 20)  # Default to 20 minutes
            
            self.root.title(f"Crypto Market Analyzer [Auto-Trading ON - {granularity}]")
            self.queue.put(("append", f"\nAuto-trading started. Analyzing {granularity.lower().replace('_', ' ')} timeframe every {wait_minutes} minutes...\n"))
            self.queue.put(("append", "When an order is placed, auto-trading will pause until the order/position is closed, then resume automatically.\n"))
            
            # Start the auto-trading process using the thread pool
            self.submit_task("auto_trading", self._auto_trading_loop)
        else:
            # Stop auto-trading immediately
            self.auto_trading = False
            
            # Force stop any running process
            if self.current_process and self.current_process.poll() is None:
                try:
                    self.current_process.terminate()
                except:
                    pass
                self.current_process = None
            
            # Cancel the auto-trading task without waiting
            with self.thread_lock:
                if "auto_trading" in self.active_futures:
                    future = self.active_futures["auto_trading"]
                    if not future.done():
                        future.cancel()
                    del self.active_futures["auto_trading"]
            
            # Update UI immediately
            self.auto_trade_btn.configure(
                text="Start Auto-Trading",
                fg_color="#4B0082"  # Back to Indigo
            )
            self.root.title("Crypto Market Analyzer")
            self.queue.put(("append", "\nAuto-trading stopped.\n"))
            self.queue.put(("status", "Auto-trading stopped"))

    def is_trading_allowed(self):
        """Check if trading is allowed based on current time"""
        # current_time = datetime.now()
        
        # # Check if it's weekend (5 = Saturday, 6 = Sunday)
        # if current_time.weekday() >= 5:
        #     return False
            
        # current_hour = current_time.hour
        # current_minute = current_time.minute
        # current_time_float = current_hour + current_minute / 60.0
        
        # # Trading is not allowed from 14:00 AM to 6:00 PM
        # if current_time_float >= 14.0 and current_time_float < 18.0:  # 11:30 AM to 5:00 PM
        #     return False
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
                            self.queue.put(("append", "\nTrading paused: Weekend trading is not allowed. Will resume on Monday at 00:00 AM (Greece time) / 10:00 AM Sunday (NYSE time).\n"))
                        else:
                            self.queue.put(("append", "\nTrading paused: Current time is outside trading hours (2:00 PM - 6:00 PM Greece time). Will resume at 6:00 PM (Greece time).\n"))
                        self._trading_paused_logged = True
                    time.sleep(60)  # Check every minute
                    continue
                else:
                    # Reset the logged flag when we're out of the pause period
                    if hasattr(self, '_trading_paused_logged'):
                        del self._trading_paused_logged
                        self.queue.put(("append", "\nTrading resumed: Current time is within trading hours (6:00 PM - 2:00 PM).\n"))

                # Check if model needs retraining
                current_time = time.time()
                if current_time - self.last_train_time >= self.model_retrain_interval:
                    self.queue.put(("append", "\nModel requires retraining. Auto-trading will stop.\n"))
                    self.root.after(0, self.toggle_auto_trading)  # Stop auto-trading
                    return
                
                # Check for open orders or positions
                has_open_orders, has_positions = self.check_for_open_orders_and_positions()
                
                if has_positions:
                    # Wait when there are open positions
                    self.queue.put(("append", "\nFound open positions. Waiting for them to close before continuing...\n"))
                    # Wait for a shorter interval before checking again
                    time.sleep(60)  # Check every minute
                    continue
                    
                if has_open_orders:
                    # For auto-trading, we should wait if there are any open orders
                    self.queue.put(("append", "\nFound open orders. Waiting for all orders to close before continuing...\n"))
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
                
                # Add timeframe alignment if enabled
                if self.alignment_var.get():
                    self.queue.put(("append", "Timeframe alignment enabled. Running multi-timeframe analysis...\n"))
                    # Run additional timeframes for alignment (only use ONE_HOUR, FIFTEEN_MINUTE, FIVE_MINUTE)
                    alignment_granularities = ["ONE_HOUR", "FIFTEEN_MINUTE", "FIVE_MINUTE"]
                    # Filter out current granularity if it's in the list
                    alignment_granularities = [g for g in alignment_granularities if g != granularity]
                    # Add remaining timeframes for alignment
                    for i, align_gran in enumerate(alignment_granularities):
                        cmd.extend([f"--alignment_timeframe_{i+1}", align_gran])
                
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
                        
                        # Check if this is a JSON line that needs to be fully captured
                        if line.strip().startswith('{') and not line.strip().endswith('}'):
                            # This might be the start of a multi-line JSON output
                            json_buffer = line
                            # Read more lines until we get a complete JSON object
                            while not json_buffer.strip().endswith('}'):
                                more = process.stdout.readline()
                                if not more:
                                    break
                                json_buffer += more
                                output += more
                            # Send the complete JSON
                            self.queue.put(("append", json_buffer))
                        else:
                            # Normal output
                            self.queue.put(("append", line))
                        
                        # Highlight alignment information
                        if "Timeframe Alignment Analysis:" in line:
                            self.queue.put(("append", "\n--- TIMEFRAME ALIGNMENT INFORMATION ---\n"))
                            capturing_trade_output = True
                            trade_output_buffer = line
                        # Start capturing trade output when we see a JSON recommendation (including new SIGNAL_TYPE format)
                        elif "{\"SIGNAL_TYPE\": \"BUY\"" in line or "{\"BUY AT\":" in line:
                            capturing_trade_output = True
                            trade_output_buffer = line
                        elif "{\"SIGNAL_TYPE\": \"SELL\"" in line or "{\"SELL AT\":" in line:
                            capturing_trade_output = True
                            trade_output_buffer = line
                        elif "{\"SIGNAL_TYPE\": \"HOLD\"" in line:
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
                    # Check if stderr contains actual error messages (not just INFO logs)
                    error_lines = [line for line in stderr.splitlines() if 'ERROR' in line or 'CRITICAL' in line]
                    if error_lines:
                        self.queue.put(("append", f"\nErrors:\n{stderr}"))
                    else:
                        # If it's just INFO logs, append them without the Errors header
                        self.queue.put(("append", stderr))
                
                # Clear current process
                self.current_process = None
                
                # Wait based on granularity before next analysis
                wait_minutes = {
                    'ONE_MINUTE': 0.3,  # Check every 20 seconds
                    'FIVE_MINUTE': 2,   # Check every 2 minutes
                    'FIFTEEN_MINUTE': 2, # Check every 2 minutes (changed from 5)
                    'THIRTY_MINUTE': 10, # Check every 10 minutes
                    'ONE_HOUR': 5      # Check every 5 minutes (changed from 20)
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
            product = self.product_var.get().lower().replace('-', '_')
            
            model_paths = [
                f"models/{product}_{granularity_formatted.replace(' ', '_')}.joblib",
                f"models/ml_model_{product}_{granularity_formatted.replace(' ', '_')}.joblib",
                f"models/{product}_{granularity_formatted.replace(' ', '_')}_prediction_model.joblib",
                f"models/model_{product}_{granularity.lower()}.joblib",
                f"models/model_{product}_{granularity}.joblib",
                f"model_{product}_{granularity.lower()}.joblib",
                f"model_{product}_{granularity}.joblib"
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
        
        # Submit the check orders task to the thread pool
        self.submit_task("check_orders", self._check_orders_thread)
    
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
        """Monitor the health of tasks and restart if necessary"""
        try:
            with self.thread_lock:
                # Check price updates task
                price_updates_active = "price_updates" in self.active_futures and not self.active_futures["price_updates"].done()
                if not price_updates_active and not self.stop_price_updates:
                    self.queue.put(("append", "\nPrice update task stopped unexpectedly. Restarting...\n"))
                    self.start_price_updates()
                    
                # Check auto-trading task
                auto_trading_active = "auto_trading" in self.active_futures and not self.active_futures["auto_trading"].done()
                if not auto_trading_active and self.auto_trading:
                    self.queue.put(("append", "\nAuto-trading task stopped unexpectedly. Restarting...\n"))
                    self.submit_task("auto_trading", self._auto_trading_loop)
                
                # Clean up completed futures to prevent memory leaks
                for task_name in list(self.active_futures.keys()):
                    future = self.active_futures[task_name]
                    if future.done():
                        # Check if there was an exception
                        try:
                            # This will re-raise any exception that occurred
                            future.result(timeout=0)
                        except Exception as e:
                            # Log the exception but don't remove the future yet
                            # as it might be handled elsewhere
                            pass
                        # Only remove futures that are truly done and processed
                        if task_name not in ["price_updates", "auto_trading"]:
                            del self.active_futures[task_name]
            
            # Periodically update trade statuses (every 5 minutes)
            current_time = time.time()
            if not hasattr(self, 'last_trade_status_update') or current_time - self.last_trade_status_update >= 300:
                self.last_trade_status_update = current_time
                # Run trade status update in thread pool to avoid blocking the UI
                self.submit_task("update_trade_statuses", self._update_trade_statuses_thread)
                
        except Exception as e:
            print(f"Error in thread monitor: {str(e)}")
            traceback.print_exc()
            
        # Schedule next monitoring check (every 30 seconds)
        self.root.after(30000, self.monitor_threads)

    def save_trade_output(self, output_text):
        """Save trade output to a file and display enhanced signal information"""
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
            
            # Try to parse and display enhanced signal information
            self.display_enhanced_signal_info(clean_text)
                
            self.queue.put(("append", "\nTrade output saved to trade_output.txt\n"))
        except Exception as e:
            self.queue.put(("append", f"\nError saving trade output: {str(e)}\n"))
    
    def display_enhanced_signal_info(self, output_text):
        """Parse the recommendation output and display enhanced information"""
        try:
            # Try to find and parse the JSON recommendation
            import re
            import json
            
            # Look for JSON object in the text (between { and })
            json_match = re.search(r'({.*?})', output_text, re.DOTALL)
            if not json_match:
                return
                
            # Try to parse the JSON
            try:
                rec_dict = json.loads(json_match.group(1).replace("'", '"'))
            except json.JSONDecodeError:
                return
                
            # Get signal type (support both old and new format)
            signal_type = rec_dict.get('SIGNAL_TYPE', 'UNKNOWN')
            if signal_type == 'UNKNOWN':
                if 'SELL AT' in rec_dict:
                    signal_type = 'SELL'
                elif 'BUY AT' in rec_dict:
                    signal_type = 'BUY'
                elif 'HOLD' in rec_dict:
                    signal_type = 'HOLD'
            
            # Display enhanced signal summary in the UI
            summary = f"\n{'='*80}\n"
            summary += f"📊 ENHANCED SIGNAL INFORMATION\n"
            summary += f"{'='*80}\n"
            
            # Signal type and confidence
            if signal_type == 'BUY':
                summary += f"Signal: 🟢 BUY\n"
            elif signal_type == 'SELL':
                summary += f"Signal: 🔴 SELL\n"
            elif signal_type == 'HOLD':
                summary += f"Signal: 🟡 HOLD\n"
            else:
                summary += f"Signal: Unknown\n"
                
            # Probability and confidence
            prob = rec_dict.get('PROBABILITY', 'N/A')
            confidence = rec_dict.get('CONFIDENCE', 'N/A')
            summary += f"Probability: {prob}%\n"
            summary += f"Confidence: {confidence}\n"
            
            # Market regime
            market_regime = rec_dict.get('MARKET_REGIME', 'N/A')
            regime_confidence = rec_dict.get('REGIME_CONFIDENCE', 'N/A')
            summary += f"Market Regime: {market_regime} ({regime_confidence} confidence)\n"
            
            # Timeframe alignment (new feature)
            if 'TIMEFRAME_ALIGNMENT' in rec_dict:
                alignment = rec_dict['TIMEFRAME_ALIGNMENT']
                summary += f"Timeframe Alignment: {alignment}/100"
                
                # Add visual indicator for alignment
                if alignment >= 90:
                    summary += " ⭐⭐⭐⭐⭐ (Excellent)\n"
                elif alignment >= 80:
                    summary += " ⭐⭐⭐⭐ (Very Good)\n"
                elif alignment >= 70:
                    summary += " ⭐⭐⭐ (Good)\n"
                elif alignment >= 60:
                    summary += " ⭐⭐ (Fair)\n"
                elif alignment >= 50:
                    summary += " ⭐ (Poor)\n"
                else:
                    summary += " ❌ (Conflicting signals)\n"
            
            # Reasoning (new feature)
            if 'REASONING' in rec_dict and rec_dict['REASONING']:
                summary += f"\nReasoning: {rec_dict['REASONING']}\n"
            
            # Risk metrics
            summary += f"\nRisk Metrics:\n"
            
            if signal_type in ['BUY', 'SELL']:
                # R/R ratio
                if 'R/R_RATIO' in rec_dict:
                    rr_ratio = float(rec_dict['R/R_RATIO'])
                    summary += f"- R/R Ratio: {rr_ratio:.2f}\n"
                
                # Volume strength
                if 'VOLUME_STRENGTH' in rec_dict:
                    summary += f"- Volume: {rec_dict['VOLUME_STRENGTH']}\n"
                
                # Volatility
                if 'VOLATILITY' in rec_dict:
                    summary += f"- Volatility: {rec_dict['VOLATILITY']}\n"
                
                # Pricing (for BUY or SELL)
                if signal_type == 'BUY':
                    entry = rec_dict.get('BUY AT', 'N/A')
                    target = rec_dict.get('SELL BACK AT', 'N/A')
                    stop = rec_dict.get('STOP LOSS', 'N/A')
                    summary += f"\nBUY at: ${entry}\n"
                    summary += f"Target: ${target}\n"
                    summary += f"Stop Loss: ${stop}\n"
                    
                    # Calculate percentages if prices are available
                    try:
                        entry_f = float(entry)
                        target_f = float(target)
                        stop_f = float(stop)
                        profit_pct = ((target_f - entry_f) / entry_f) * 100
                        loss_pct = ((entry_f - stop_f) / entry_f) * 100
                        summary += f"Potential Gain: {profit_pct:.2f}%\n"
                        summary += f"Potential Loss: {loss_pct:.2f}%\n"
                    except (ValueError, TypeError):
                        pass
                    
                elif signal_type == 'SELL':
                    entry = rec_dict.get('SELL AT', 'N/A')
                    target = rec_dict.get('BUY BACK AT', 'N/A')
                    stop = rec_dict.get('STOP LOSS', 'N/A')
                    summary += f"\nSELL at: ${entry}\n"
                    summary += f"Target: ${target}\n"
                    summary += f"Stop Loss: ${stop}\n"
                    
                    # Calculate percentages if prices are available
                    try:
                        entry_f = float(entry)
                        target_f = float(target)
                        stop_f = float(stop)
                        profit_pct = ((entry_f - target_f) / entry_f) * 100
                        loss_pct = ((stop_f - entry_f) / entry_f) * 100
                        summary += f"Potential Gain: {profit_pct:.2f}%\n"
                        summary += f"Potential Loss: {loss_pct:.2f}%\n"
                    except (ValueError, TypeError):
                        pass
            
            # Add message about auto trading
            if self.auto_trading:
                summary += f"\n🤖 Auto-trading is ON. Trades will be executed automatically if conditions are met.\n"
            elif self.execute_trades_var.get():
                summary += f"\n🤖 Trade execution is enabled. This recommendation will be executed if conditions are met.\n"
            else:
                summary += f"\n📝 Trade execution is disabled. Use the Execute Trades checkbox to enable automatic trading.\n"
            
            # Add the summary to the output
            self.queue.put(("append", summary))
            
        except Exception as e:
            # Don't show error to user, just log it
            print(f"Error displaying enhanced signal info: {str(e)}")

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

    def _update_trade_statuses_thread(self):
        """Thread function to update trade statuses in trade_history.csv using update_trade_status.py"""
        try:
            self.queue.put(("status", "Updating trade statuses..."))
            
            # Run update_trade_status.py as a separate process using subprocess
            # This avoids the signal module issue in non-main threads
            import subprocess
            import os
            import datetime
            
            # Set environment variable to indicate this is being run from market_ui
            env = os.environ.copy()
            env['MARKET_UI'] = '1'
            
            # First run update_trade_status.py
            process = subprocess.Popen(
                ["python", "update_trade_status.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env  # Pass the modified environment
            )
            
            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # We don't want to show the output in the UI text window
                    # self.queue.put(("append", output))
                    pass
            
            # Get any remaining output
            stdout, stderr = process.communicate()
            
            # Only show errors in the UI
            if stderr and stderr.strip():  # Only show errors if stderr is not empty after stripping whitespace
                self.queue.put(("append", f"\nErrors from update_trade_status.py:\n{stderr}"))
                if process.returncode != 0:
                    raise Exception(f"update_trade_status.py failed with return code {process.returncode}")
            
            # Now run update_pending_trades.py
            self.queue.put(("status", "Updating pending trades..."))
            
            pending_process = subprocess.Popen(
                ["python", "update_pending_trades.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env  # Pass the modified environment
            )
            
            # Read output in real-time
            while True:
                output = pending_process.stdout.readline()
                if output == '' and pending_process.poll() is not None:
                    break
                if output:
                    # We don't want to show the output in the UI text window
                    # self.queue.put(("append", output))
                    pass
            
            # Get any remaining output
            pending_stdout, pending_stderr = pending_process.communicate()
            
            # Only show actual errors in the UI, filter out INFO logs
            if pending_stderr:
                # Check if stderr contains actual errors or just INFO logs
                stderr_lines = pending_stderr.decode('utf-8') if isinstance(pending_stderr, bytes) else pending_stderr
                error_lines = []
                
                for line in stderr_lines.splitlines():
                    # Skip INFO level log messages
                    if " - INFO - " in line:
                        continue
                    error_lines.append(line)
                
                # Only display if there are actual error lines
                if error_lines:
                    error_text = "\n".join(error_lines)
                    self.queue.put(("append", f"\nErrors from update_pending_trades.py:\n{error_text}"))
                    if pending_process.returncode != 0:
                        raise Exception(f"update_pending_trades.py failed with return code {pending_process.returncode}")
            
            # Update the trade status frame instead of appending to console
            if process.returncode == 0 and pending_process.returncode == 0:
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                self.queue.put(("update_trade_status", {
                    "status": "Trade statuses and pending trades updated successfully",
                    "time": f"Last update: {current_time}"
                }))
            
            self.queue.put(("status", "Trade status update completed"))
        except Exception as e:
            self.queue.put(("append", f"\nError updating trade statuses: {str(e)}\n"))
            self.queue.put(("status", "Error updating trades"))
            self.queue.put(("update_trade_status", {
                "status": f"Error: {str(e)}",
                "time": f"Last update: {datetime.datetime.now().strftime('%H:%M:%S')}"
            }))
    
    def update_risk_level(self):
        """Update the risk level based on the current state"""
        # This method is no longer needed as risk management has been removed
        pass

    def save_settings(self):
        """Save UI settings to a JSON file"""
        try:
            settings = {
                "margin": self.margin_var.get(),
                "leverage": self.leverage_var.get(),
                "tp_sl": self.tp_sl_var.get(),
                "limit_order": self.limit_order_var.get(),
                "product": self.product_var.get(),
                "model": self.model_var.get(),
                "granularity": self.granularity_var.get(),
                "execute_trades": self.execute_trades_var.get(),
                "alignment_enabled": self.alignment_var.get()
            }
            
            with open("ui_settings.json", "w") as f:
                json.dump(settings, f, indent=2)
                
        except Exception as e:
            self.queue.put(("append", f"\nError saving settings: {str(e)}\n"))

    def load_settings(self):
        """Load UI settings from JSON file"""
        try:
            if os.path.exists("ui_settings.json"):
                with open("ui_settings.json", "r") as f:
                    return json.load(f)
            return {
                "margin": "60",
                "leverage": 10,
                "tp_sl": 0.2,
                "limit_order": True,
                "product": "BTC-USDC",
                "model": "o1_mini",
                "granularity": "ONE_HOUR",
                "execute_trades": False,
                "alignment_enabled": False
            }
        except Exception as e:
            self.queue.put(("append", f"\nError loading settings: {str(e)}\n"))
            return {
                "margin": "60",
                "leverage": 10,
                "tp_sl": 0.2,
                "limit_order": True,
                "product": "BTC-USDC",
                "model": "o1_mini",
                "granularity": "ONE_HOUR",
                "execute_trades": False,
                "alignment_enabled": False
            }

    def update_ui(self, update_type, data=None):
        """
        Thread-safe method to update the UI from any thread
        
        Args:
            update_type: The type of update (append, status, etc.)
            data: The data for the update
        """
        # Put the update in the queue for the main thread to process
        self.queue.put((update_type, data))
    
    def submit_task(self, task_name, func, *args, **kwargs):
        """Submit a task to the thread pool with proper error handling"""
        with self.thread_lock:
            # Cancel any existing task with the same name
            if task_name in self.active_futures:
                future = self.active_futures[task_name]
                if not future.done():
                    future.cancel()
                    self.update_ui("append", f"\nCancelled previous {task_name.replace('_', ' ')} task.\n")
            
            # If this is a trade status update task, update the trade status frame
            if task_name == "update_trade_statuses":
                import datetime
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                self.update_ui("update_trade_status", {
                    "status": "Updating trade statuses...",
                    "time": f"Started at: {current_time}"
                })
            
            # Submit new task with error handling wrapper
            def task_wrapper(*args, **kwargs):
                try:
                    # Check if this is a trade status update task
                    is_trade_status_task = task_name == "update_trade_statuses"
                    
                    # Only append to console if not a trade status task
                    if not is_trade_status_task:
                        self.update_ui("append", f"\nStarting {task_name.replace('_', ' ')} task...\n")
                    
                    result = func(*args, **kwargs)
                    
                    # Only append to console if not a trade status task
                    if not is_trade_status_task:
                        self.update_ui("append", f"\nCompleted {task_name.replace('_', ' ')} task.\n")
                    
                    return result
                except Exception as e:
                    # Log the error to the UI
                    error_msg = f"\nError in {task_name}: {str(e)}\n"
                    if kwargs.get('verbose_errors', True):
                        error_msg += f"Traceback: {traceback.format_exc()}\n"
                    
                    # Always show errors in the console
                    self.update_ui("append", error_msg)
                    self.update_ui("status", f"Error in {task_name}")
                    # Re-raise for future.exception() to catch
                    raise
                finally:
                    # Remove from active tasks when done
                    with self.thread_lock:
                        if task_name in self.active_futures:
                            del self.active_futures[task_name]
            
            # Submit wrapped task
            future = self.thread_pool.submit(task_wrapper, *args, **kwargs)
            self.active_futures[task_name] = future
            return future
            
    def cancel_all_tasks(self):
        """Cancel all running tasks in the thread pool"""
        with self.thread_lock:
            for name, future in list(self.active_futures.items()):
                if not future.done():
                    self.queue.put(("append", f"\nCancelling task: {name}\n"))
                    future.cancel()
            self.active_futures.clear()
            
    def on_exit(self):
        """Clean up resources and exit the application"""
        try:
            # Stop price updates first (this is fast)
            self.stop_price_updates = True
            
            # Force stop auto-trading and simplified trading immediately if active
            if self.auto_trading:
                self.auto_trading = False
            if self.simplified_trading:
                self.simplified_trading = False
                
            if self.current_process and self.current_process.poll() is None:
                try:
                    self.current_process.terminate()
                except:
                    pass
                self.current_process = None
            
            # Cancel all tasks without waiting
            with self.thread_lock:
                for name, future in list(self.active_futures.items()):
                    if not future.done():
                        future.cancel()
                self.active_futures.clear()
            
            # Force close the session
            try:
                self.session.close()
            except:
                pass
            
            # Shutdown thread pool with a very short timeout
            try:
                self.thread_pool.shutdown(wait=True, timeout=0.5, cancel_futures=True)
            except:
                # If shutdown times out, force shutdown without waiting
                self.thread_pool.shutdown(wait=False, cancel_futures=True)
            
            # Save settings in a separate thread to prevent hanging
            threading.Thread(target=self.save_settings, daemon=True).start()
            
            # Show shutdown message without waiting for response
            try:
                self.queue.put(("status", "Shutting down..."))
                self.queue.put(("append", "\nApplication is shutting down...\n"))
            except:
                pass
            
            # Schedule the actual window destruction after a very short delay
            self.root.after(100, self._force_exit)
            
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
            self._force_exit()
    
    def _force_exit(self):
        """Force exit the application"""
        try:
            # Force destroy the window
            self.root.quit()
            self.root.destroy()
        except:
            # If normal destroy fails, use os._exit as last resort
            import os
            os._exit(0)
    
    def run(self):
        """Run the application main loop"""
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
        finally:
            # Ensure cleanup if the mainloop exits without calling on_exit
            try:
                # Stop auto-trading if active
                if self.auto_trading:
                    self.toggle_auto_trading()
                # Cancel all tasks and shutdown thread pool
                self.cancel_all_tasks()
                self.thread_pool.shutdown(wait=False)
                # Ensure price updates are stopped and session is closed
                self.stop_price_update_thread()
                self.session.close()
            except Exception as e:
                print(f"Error during emergency cleanup: {str(e)}")

    def schedule_periodic_cleanup(self):
        """Schedule periodic cleanup to prevent memory buildup and UI lag"""
        try:
            # Perform memory cleanup
            self.perform_memory_cleanup()
            
            # Schedule next cleanup (every 5 minutes)
            self.root.after(300000, self.schedule_periodic_cleanup)
        except Exception as e:
            print(f"Error in periodic cleanup: {str(e)}")
            # Ensure cleanup continues even if there's an error
            self.root.after(300000, self.schedule_periodic_cleanup)
    
    def perform_memory_cleanup(self):
        """Perform memory cleanup to prevent UI lag"""
        try:
            # Limit text widget content
            content = self.output_text.get("1.0", "end")
            if len(content) > 100000:
                # Keep only the last 100,000 characters
                self.output_text.delete("1.0", f"end-{100000}c")
                print(f"Cleaned up text widget, removed {len(content) - 100000} characters")
            
            # Clean up completed futures
            with self.thread_lock:
                for task_name in list(self.active_futures.keys()):
                    future = self.active_futures[task_name]
                    if future.done() and task_name not in ["price_updates", "auto_trading"]:
                        del self.active_futures[task_name]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print("Periodic memory cleanup completed")
        except Exception as e:
            print(f"Error during memory cleanup: {str(e)}")

    def toggle_simplified_trading(self):
        """Toggle simplified trading on/off"""
        if not self.simplified_trading:
            # Start simplified trading
            self.simplified_trading = True
            self.simplified_trade_btn.configure(
                text="Stop Simplified Trading",
                fg_color="#B22222"  # Fire Brick Red
            )
            self.root.title("Crypto Market Analyzer [Simplified Trading ON]")
            self.queue.put(("append", "\nSimplified trading started. Using FIVE_MINUTE timeframe with 2-minute checks...\n"))
            self.queue.put(("append", "Strategy: RSI + EMA + Volume analysis\n"))
            self.queue.put(("append", "Will not trade if there are open positions or orders.\n"))
            
            # Start the simplified trading process using the thread pool
            self.submit_task("simplified_trading", self._simplified_trading_loop)
        else:
            # Stop simplified trading immediately
            self.simplified_trading = False
            
            # Force stop any running process
            if self.current_process and self.current_process.poll() is None:
                try:
                    self.current_process.terminate()
                except:
                    pass
                self.current_process = None
            
            # Cancel the simplified trading task without waiting
            with self.thread_lock:
                if "simplified_trading" in self.active_futures:
                    future = self.active_futures["simplified_trading"]
                    if not future.done():
                        future.cancel()
                    del self.active_futures["simplified_trading"]
            
            # Update UI immediately
            self.simplified_trade_btn.configure(
                text="Start Simplified Trading",
                fg_color="#2E8B57"  # Back to Sea Green
            )
            self.root.title("Crypto Market Analyzer")
            self.queue.put(("append", "\nSimplified trading stopped.\n"))
            self.queue.put(("status", "Simplified trading stopped"))

    def _simplified_trading_loop(self):
        """Background loop for simplified trading"""
        while self.simplified_trading:
            try:
                # Check if trading is allowed based on time
                current_time = datetime.now()
                if not self.is_trading_allowed():
                    if not hasattr(self, '_simplified_trading_paused_logged'):
                        if current_time.weekday() >= 5:
                            self.queue.put(("append", "\nTrading paused: Weekend trading is not allowed. Will resume on Monday at 00:00 AM (Greece time) / 10:00 AM Sunday (NYSE time).\n"))
                        else:
                            self.queue.put(("append", "\nTrading paused: Current time is outside trading hours (2:00 PM - 6:00 PM Greece time). Will resume at 6:00 PM (Greece time).\n"))
                        self._simplified_trading_paused_logged = True
                    time.sleep(60)  # Check every minute
                    continue
                else:
                    # Reset the logged flag when we're out of the pause period
                    if hasattr(self, '_simplified_trading_paused_logged'):
                        del self._simplified_trading_paused_logged
                        self.queue.put(("append", "\nTrading resumed: Current time is within trading hours (6:00 PM - 2:00 PM).\n"))

                # Check for open orders or positions
                has_open_orders, has_positions = self.check_for_open_orders_and_positions()
                
                if has_positions:
                    # Wait when there are open positions
                    self.queue.put(("append", "\nFound open positions. Waiting for them to close before continuing...\n"))
                    time.sleep(60)  # Check every minute
                    continue
                    
                if has_open_orders:
                    # Wait if there are any open orders
                    self.queue.put(("append", "\nFound open orders. Waiting for all orders to close before continuing...\n"))
                    time.sleep(60)  # Check every minute
                    continue
                
                # Run simplified trading bot
                self.queue.put(("append", "\nRunning simplified trading analysis...\n"))
                
                # Create and run the simplified trading process
                cmd = [
                    "python",
                    "simplified_trading_bot.py",
                    "--product_id",
                    self.product_var.get(),
                    "--margin",
                    self.margin_var.get(),
                    "--leverage",
                    str(self.leverage_var.get())
                ]
                
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
                        
                        # Start capturing trade output when we see a signal
                        if "[SIGNAL]" in line:
                            capturing_trade_output = True
                            trade_output_buffer = line
                        # Continue capturing trade output
                        elif capturing_trade_output:
                            trade_output_buffer += line
                        
                        # Check if a trade was executed
                        if "Order placed successfully" in line:
                            self.queue.put(("append", "\nTrade executed! Waiting for order to close before continuing...\n"))
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
                    # Check if stderr contains actual error messages (not just INFO logs)
                    error_lines = [line for line in stderr.splitlines() if 'ERROR' in line or 'CRITICAL' in line]
                    if error_lines:
                        self.queue.put(("append", f"\nErrors:\n{stderr}"))
                    else:
                        # If it's just INFO logs, append them without the Errors header
                        self.queue.put(("append", stderr))
                
                # Clear current process
                self.current_process = None
                
                # Wait 2 minutes before next check
                for _ in range(2):
                    if not self.simplified_trading:
                        return
                    time.sleep(60)  # 1 minute intervals
                
            except Exception as e:
                self.queue.put(("append", f"\nError in simplified trading loop: {str(e)}\nTraceback:\n{traceback.format_exc()}\n"))
                # Don't exit the thread on error, just wait a bit and continue
                time.sleep(60)  # Wait 1 minute before retrying on error
                continue

if __name__ == "__main__":
    app = MarketAnalyzerUI()
    app.run() 