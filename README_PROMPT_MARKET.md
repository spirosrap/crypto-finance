# AI-Powered Crypto Trading Advisor

`prompt_market.py` is an advanced cryptocurrency trading advisor that uses AI models to analyze market data and generate trading recommendations. This tool combines technical analysis, market regime detection, and advanced risk management to provide actionable trading signals.

## Overview

This tool:
1. Fetches real-time market data and technical indicators
2. Analyzes the data using advanced AI models (OpenAI, DeepSeek, X AI, etc.)
3. Generates structured trading recommendations (BUY, SELL, HOLD)
4. Optionally executes trades with sophisticated risk management

## Requirements

- Python 3.8+
- API keys for one or more of the following:
  - OpenAI (`OPENAI_KEY`)
  - DeepSeek (`DEEPSEEK_KEY`)
  - X AI/Grok (`XAI_KEY`)
  - OpenRouter (`OPENROUTER_API_KEY`)
  - Hyperbolic (`HYPERBOLIC_KEY`)
- Coinbase Advanced API credentials (`API_KEY_PERPS` and `API_SECRET_PERPS`)
- Ollama (optional, for local model inference)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys in config.py or environment variables
```

## Usage

### Basic Usage

```bash
python prompt_market.py --product_id BTC-USDC --granularity ONE_HOUR
```

### Model Selection

Use one of these flags to select the AI model:

```bash
python prompt_market.py --use_deepseek  # Use DeepSeek Chat model
python prompt_market.py --use_grok  # Use X AI Grok model
python prompt_market.py --use_gpt4o  # Use GPT-4o model
python prompt_market.py --use_o1_mini  # Use Claude o1-mini model (faster)
python prompt_market.py --use_o3_mini  # Use Claude o3-mini model (balanced)
python prompt_market.py --use_ollama  # Use local Ollama DeepSeek R1 7B model
python prompt_market.py --use_ollama_1_5b  # Use local Ollama DeepSeek R1 1.5B model
python prompt_market.py --use_ollama_8b  # Use local Ollama DeepSeek R1 8B model
python prompt_market.py --use_ollama_14b  # Use local Ollama DeepSeek R1 14B model
python prompt_market.py --use_ollama_32b  # Use local Ollama DeepSeek R1 32B model
python prompt_market.py --use_ollama_70b  # Use local Ollama DeepSeek R1 70B model
python prompt_market.py --use_ollama_671b  # Use local Ollama DeepSeek R1 671B model
```

### Trade Execution

To automatically execute trades based on recommendations:

```bash
python prompt_market.py --execute_trades --margin 100 --leverage 10
```

For limit orders instead of market orders:
```bash
python prompt_market.py --execute_trades --limit_order
```

## How It Works

### 1. Market Analysis Pipeline

```
Market Data → Technical Analysis → AI Model → Structured Recommendation → Risk Management → Trade Execution
```

The tool follows this process:
1. Fetches market data (price, volume, indicators)
2. Runs technical analysis through `market_analyzer.py`
3. Sends analysis to the selected AI model
4. Receives structured trading recommendation
5. Applies risk management filters
6. Optionally executes trades

### 2. AI Model Recommendation Format

The AI models are prompted to provide recommendations in a structured JSON format:

```json
{
    "SIGNAL_TYPE": "BUY",
    "BUY AT": 45000,
    "SELL BACK AT": 48000,
    "STOP LOSS": 44000,
    "PROBABILITY": 80,
    "CONFIDENCE": "Strong",
    "R/R_RATIO": 3.0,
    "VOLUME_STRENGTH": "Strong",
    "VOLATILITY": "Medium",
    "MARKET_REGIME": "Bullish",
    "REGIME_CONFIDENCE": "High",
    "TIMEFRAME_ALIGNMENT": 85,
    "REASONING": "BTC showing strong bullish momentum with increasing volume, higher lows, and positive news sentiment.",
    "IS_VALID": true
}
```

### 3. Market Regime Analysis

The system categorizes the market into regimes:
- **Strong Bullish**: Sustained uptrend with strong momentum
- **Bullish**: Uptrend with moderate momentum
- **Choppy Bullish**: Upward bias with consolidation
- **Choppy**: Range-bound without clear direction
- **Choppy Bearish**: Downward bias with consolidation
- **Bearish**: Downtrend with moderate momentum
- **Strong Bearish**: Sustained downtrend with strong momentum

Trade signals must align with market regime or have exceptional probability.

### 4. Risk Management

The tool applies multiple risk filters before executing trades:

1. **Probability Threshold**: Requires ≥65% probability (higher for choppy markets)
2. **Market Regime Alignment**: Trade direction must align with market regime
3. **R/R Ratio**: Minimum risk-reward of 1.0 (higher for choppy markets)
4. **Stop Loss Validation**: Ensures stop loss is properly placed (above entry for SELL, below for BUY)
5. **Volatility Assessment**: Adjusts thresholds based on market volatility
6. **Price Deviation**: Checks if current price is too far from recommended entry

### 5. Advanced Position Sizing

Position size is dynamically adjusted based on multiple factors:

1. **Stop Loss Width**:
   - Tight stop loss (≤1.5%): 100% sizing
   - Medium stop loss (1.5-3%): 80% sizing
   - Wide stop loss (>3%): 60% sizing

2. **Market Regime**:
   - Choppy markets: Additional 20% reduction
   - Counter-trend trades: Additional 30% reduction

3. **Timeframe Alignment**:
   - High alignment (≥90): 10% size increase
   - Low alignment (<70): 10% size reduction

4. **Safety Caps**:
   - Maximum size: 100% of base size
   - Minimum size: 30% of base size

### 6. Trade Execution Logic

For executing trades, the tool:

1. Validates all parameters and market conditions
2. Determines optimal order type (market vs. limit)
3. Formats price precision based on asset type
4. Executes trade via Coinbase API
5. Records trade details to history file
6. Provides detailed execution summary

## Command Line Arguments

```
usage: prompt_market.py [-h] [--product_id PRODUCT_ID] [--granularity GRANULARITY] [--margin MARGIN] [--leverage LEVERAGE]
                        [--execute_trades] [--limit_order] [--use_deepseek] [--use_reasoner] [--use_grok] [--use_gpt45_preview]
                        [--use_o1] [--use_o1_mini] [--use_o3_mini] [--use_o3_mini_effort] [--use_gpt4o] [--use_deepseek_r1]
                        [--use_ollama] [--use_ollama_1_5b] [--use_ollama_8b] [--use_ollama_14b] [--use_ollama_32b] 
                        [--use_ollama_70b] [--use_ollama_671b] [--use_hyperbolic]

AI-Powered Crypto Market Analyzer and Trading Recommendation System

This tool analyzes cryptocurrency market data using various AI models and provides trading recommendations.
It supports multiple AI providers including OpenAI, DeepSeek, X AI (Grok), and others.
The analysis includes technical indicators, market sentiment, and risk metrics.

Example usage:
  python prompt_market.py --product_id BTC-USDC --granularity ONE_HOUR
  python prompt_market.py --use_deepseek --margin 200 --leverage 10
  python prompt_market.py --use_grok --execute_trades
  python prompt_market.py --use_ollama_32b --product_id ETH-USDT  # Use local Ollama model
```

## Best Practices

1. **Start Conservative**: Begin with `--use_o1_mini` model without `--execute_trades` to observe recommendations
2. **Use Risk Management**: Keep margin and leverage low until you're comfortable with the system
3. **Timeframe Matters**: ONE_HOUR granularity tends to provide more stable signals than faster timeframes
4. **Market Regimes**: Pay attention to the market regime - avoid counter-trend trades during strong trends
5. **Record Keeping**: Monitor trade history for performance tracking and improvement

## Disclaimer

This tool is for educational and research purposes only. Trading cryptocurrencies involves substantial risk. Always perform your own research and consider consulting a financial advisor.

---

**Note**: This document provides an overview of `prompt_market.py` functionality. Trading strategies and market analysis techniques are constantly evolving. The tool's effectiveness depends on market conditions, chosen parameters, and the quality of input data.