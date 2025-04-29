# Asset-specific parameters for the simplified trading bot

# Asset-specific parameters
ASSET_PARAMS = {
    'ETH-USDC': {
        'RSI_THRESHOLD': 30,
        'RSI_CONFIRMATION_THRESHOLD': 35,
        'VOLUME_LOOKBACK': 15,
        'TP_PERCENT': 0.018,
        'SL_PERCENT': 0.009,
        'LEVERAGE': 5,
        'POSITION_SIZE_USD': 100,
        'VOLUME_THRESHOLD': 1.6,
        'VOLATILITY_THRESHOLD': 0.3,
        'TREND_THRESHOLD': 0.0012,
        'mean_atr_percent': 0.32,
        'std_atr_percent': 0.16,
        # Filter parameters
        'UNCERTAIN_RSI_FILTER': 25,         # Filter 1: Max RSI in uncertain regime
        'UNCERTAIN_VOLUME_FILTER': 1.7,     # Filter 2: Min relative volume in uncertain regime
        'LOW_VOL_ATR_FILTER': 0.25,         # Filter 3: Min ATR% for low volatility filter
        'LOW_VOL_TREND_FILTER': 0.0008,     # Filter 3: Min trend slope for low volatility filter
        'LOW_VOL_RSI_FILTER': 24            # Filter 3: Max RSI for low volatility filter
    },
    'SOL-USDC': {
        'RSI_THRESHOLD': 28,
        'RSI_CONFIRMATION_THRESHOLD': 33,
        'VOLUME_LOOKBACK': 10,
        'TP_PERCENT': 0.02,
        'SL_PERCENT': 0.01,
        'LEVERAGE': 3,
        'POSITION_SIZE_USD': 100,
        'VOLUME_THRESHOLD': 1.8,
        'VOLATILITY_THRESHOLD': 0.35,
        'TREND_THRESHOLD': 0.0015,
        'mean_atr_percent': 0.35,
        'std_atr_percent': 0.18,
        # Filter parameters
        'UNCERTAIN_RSI_FILTER': 26,         # Filter 1: Max RSI in uncertain regime
        'UNCERTAIN_VOLUME_FILTER': 1.8,     # Filter 2: Min relative volume in uncertain regime
        'LOW_VOL_ATR_FILTER': 0.3,          # Filter 3: Min ATR% for low volatility filter
        'LOW_VOL_TREND_FILTER': 0.0012,     # Filter 3: Min trend slope for low volatility filter
        'LOW_VOL_RSI_FILTER': 23            # Filter 3: Max RSI for low volatility filter
    },
    'DOGE-USDC': {
        'RSI_THRESHOLD': 28,  # Lower RSI threshold due to higher volatility
        'RSI_CONFIRMATION_THRESHOLD': 30,
        'VOLUME_LOOKBACK': 8,  # Shorter lookback due to faster price movements
        'TP_PERCENT': 0.02,  # Higher TP due to higher volatility
        'SL_PERCENT': 0.009,  # Higher SL to account for higher volatility
        'LEVERAGE': 5,  # Lower leverage due to higher risk
        'POSITION_SIZE_USD': 100,
        'VOLUME_THRESHOLD': 2.0,  # Higher volume threshold due to higher volatility
        'VOLATILITY_THRESHOLD': 0.6,  # Higher volatility threshold
        'TREND_THRESHOLD': 0.002,  # Higher trend threshold
        'mean_atr_percent': 0.190,  # Higher mean ATR
        'std_atr_percent': 0.048,  # Higher ATR standard deviation
        # Filter parameters
        'UNCERTAIN_RSI_FILTER': 28,         # Filter 1: Max RSI in uncertain regime (higher for DOGE)
        'UNCERTAIN_VOLUME_FILTER': 1.5,     # Filter 2: Min relative volume in uncertain regime (higher for DOGE)
        'LOW_VOL_ATR_FILTER': 0.6,          # Filter 3: Min ATR% for low volatility filter
        'LOW_VOL_TREND_FILTER': 0.003,      # Filter 3: Min trend slope for low volatility filter
        'LOW_VOL_RSI_FILTER': 25            # Filter 3: Max RSI for low volatility filter
    }
}

# Default parameters for unknown assets
DEFAULT_PARAMS = {
    'RSI_THRESHOLD': 30,
    'RSI_CONFIRMATION_THRESHOLD': 35,
    'VOLUME_LOOKBACK': 20,
    'TP_PERCENT': 0.015,
    'SL_PERCENT': 0.007,
    'LEVERAGE': 5,
    'POSITION_SIZE_USD': 100,
    'VOLUME_THRESHOLD': 1.4,
    'VOLATILITY_THRESHOLD': 0.25,
    'TREND_THRESHOLD': 0.001,
    'mean_atr_percent': 0.284,
    'std_atr_percent': 0.148,
    # Filter parameters (default/BTC)
    'UNCERTAIN_RSI_FILTER': 27,         # Filter 1: Max RSI in uncertain regime
    'UNCERTAIN_VOLUME_FILTER': 1.5,     # Filter 2: Min relative volume in uncertain regime
    'LOW_VOL_ATR_FILTER': 0.2,          # Filter 3: Min ATR% for low volatility filter
    'LOW_VOL_TREND_FILTER': 0.001,      # Filter 3: Min trend slope for low volatility filter
    'LOW_VOL_RSI_FILTER': 25            # Filter 3: Max RSI for low volatility filter
}

def get_asset_params(product_id):
    """Get parameters for a specific asset or return defaults"""
    return ASSET_PARAMS.get(product_id, DEFAULT_PARAMS) 