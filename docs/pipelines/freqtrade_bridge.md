## Freqtrade Bridge Workflow

This guide shows how to drive a Freqtrade bot with signals produced by the existing
finder stack (`short_term_crypto_finder.py`, `multi_coin_reservoir_daytrader.py`,
`add_position_from_finder.py`).

### 1. Export Finder Signals

Use the new exporter to translate `finder_short.txt` into a structured JSON file:

```bash
# Generate finder_short.txt as usualFirst
python short_term_crypto_finder.py --profile wide --plain-output finder_short.txt

# Convert the plain-text report into Freqtrade-friendly JSON
python export_finder_signals.py \
  --file finder_short.txt \
  --out signals/freqtrade_signals.json \
  --portfolio-usd 13000 \
  --leverage 50 \
  --expiry-hours 24
```

The exporter:

- Parses every block via `add_position_from_finder.py`.
- Normalizes contract naming (`1000SHIB-PERP-INTX` → `1000SHIB/USDC`).
- Converts TP/SL to the correct perp price scale (e.g., 0.014000 for `1000BONK`).
- Writes `signals/freqtrade_signals.json` with expiry metadata for the strategy.

### 2. Configure Freqtrade

Add the new strategy to your Freqtrade project:

1. Copy `freqtrade/strategies/finder_bridge_strategy.py` into your bot’s `user_data/strategies/`.
2. Update `freqtrade/config.json` (or your profile) with:

```jsonc
{
  "strategy": "FinderBridgeStrategy",
  "timeframe": "1h",
  "max_open_trades": 5,
  "stake_currency": "USDC",
  "stake_amount": "unlimited",
  "custom_info": {
    "finder_signal_path": "signals/freqtrade_signals.json"
  }
}
```

The bridge strategy automatically reloads `finder_signal_path` whenever the file
timestamp changes, so you can refresh signals without restarting the bot.

### 3. Run a Dry-Run

```bash
freqtrade backtesting --strategy FinderBridgeStrategy --timeframe 1h --timerange 20241001-20241030

freqtrade trade --dry-run --strategy FinderBridgeStrategy
```

During live/dry mode the console will print a compact summary such as:

```
8 signal(s) | threshold=0.0030 | timeframe=ONE_HOUR | expiry=24h | generated_utc=2025-10-30T11:43:00Z
1. 1000BONK      SHORT entry=0.014000 tp=0.014000 sl=0.015000 pred=0.55% atr=1.83% rv=0.27 timestamp=2025-10-30 07:00:00Z
...
```

### 4. Optional Enhancements

- **Auto-export**: add `export_finder_signals.py` to the same cron/task that runs the finder.
- **Confidence-weighted sizing**: `position_pct` and `confidence` are preserved in the JSON if you need custom position sizing inside the strategy.
- **Risk controls**: the bridge uses finder TP/SL targets, but you can extend `custom_stoploss`/`custom_exit` hooks for multi-stage exits.

With this workflow you can keep the existing signal generation pipeline while handing
execution, risk management, and analytics over to Freqtrade.
