GRANULARITY_SETTINGS = {
    "ONE_MINUTE": {
        "cooldown_period": 5 * 60,  # 5 minutes
        "max_trades_per_day": 48,
        "min_price_change": 0.07,
        "strong_buy_percentage": 0.15,
        "buy_percentage": 0.05
    },
    "FIVE_MINUTE": {
        "cooldown_period": 15 * 60,  # 15 minutes
        "max_trades_per_day": 24,
        "min_price_change": 0.06,
        "strong_buy_percentage": 0.2,
        "buy_percentage": 0.07
    },
    "FIFTEEN_MINUTE": {
        "cooldown_period": 15 * 60,
        "max_trades_per_day": 24,
        "min_price_change": 0.08,
        "strong_buy_percentage": 0.3,
        "buy_percentage": 0.15
    },
   "THIRTY_MINUTE": {
        "cooldown_period": 24 * 60 * 60 * 0.2,
        "max_trades_per_day": 2,
        "min_price_change": 0.08,
        "strong_buy_percentage": 0.45,
        "buy_percentage": 0.15
    },
    "ONE_HOUR": {  # Default settings
        "cooldown_period": 24 * 60 * 60,
        "max_trades_per_day": 1,
        "min_price_change": 0.08,
        "strong_buy_percentage": 0.8,
        "buy_percentage": 0.4
    }
}
