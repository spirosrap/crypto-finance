# Crypto Market Analyzer Web Interface

This is a web-based interface for the Crypto Market Analyzer, providing the same functionality as the desktop application but accessible through a web browser.

## Features

- Real-time market analysis using various AI models
- Live price updates for crypto assets
- Manual and automated trading capabilities
- Trade history tracking and visualization
- Configurable trading parameters (leverage, margin type, etc.)
- Support for multiple timeframes and trading pairs

## Installation

1. Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

2. Set up your API keys:
   - Create a `config.py` file with your Coinbase API keys:
   ```python
   API_KEY_PERPS = "your_api_key"
   API_SECRET_PERPS = "your_api_secret"
   OPENAI_KEY = "your_openai_key"
   DEEPSEEK_KEY = "your_deepseek_key"
   XAI_KEY = "your_xai_key"
   OPENROUTER_API_KEY = "your_openrouter_key"
   HYPERBOLIC_KEY = "your_hyperbolic_key"
   ```
   - Alternatively, set these as environment variables

## Usage

1. Start the web server:

```bash
python app.py
```

2. Open your web browser and navigate to:

```
http://localhost:5000
```

3. The interface is divided into two main sections:
   - Left sidebar: Configuration and control panel
   - Main content area: Analysis output and trade history

## Trading Options

- **Product**: Select the cryptocurrency pair to trade (e.g., BTC-USDC)
- **Model**: Choose the AI model for analysis
- **Time Frame**: Select the granularity for analysis (1min to 1hr)
- **Trading Options**:
  - Margin Type: CROSS or ISOLATED
  - Leverage: 1x to 20x
  - TP/SL: Take profit/stop loss percentage
  - Limit Order: Toggle between market and limit orders
  - Risk Level: Low, Medium, or High

## Actions

- **Run Analysis**: Perform market analysis without executing trades
- **LONG/SHORT**: Execute a long or short position
- **Close All Positions**: Close any open positions
- **Check Open Orders**: Check for any open orders or positions
- **Start/Stop Auto-Trading**: Toggle automated trading based on analysis

## Auto-Trading

When auto-trading is enabled, the system will:
1. Check for open orders and positions
2. Run analysis at intervals based on the selected timeframe
3. Execute trades according to the analysis
4. Pause during non-trading hours (7:21 AM to 5:00 PM)

## Security Notes

- This application handles real money and executes real trades
- Always start with small amounts until you're comfortable with the system
- Never share your API keys or expose them in public repositories
- Consider running the application on a secure, private server

## Troubleshooting

- If the application fails to connect to Coinbase, check your API keys
- If analysis fails, ensure all required dependencies are installed
- For any issues with the web interface, check the browser console for errors

## License

This software is provided as-is without warranty. Use at your own risk. 