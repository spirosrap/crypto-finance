# Zero-Fee INTX Intraday Trader

Intraday trading loop designed for the INTX perpetual venue where taker fees are zero. The bot scans a default basket of the 15 most liquid INTX perps on one-minute bars, hunts for short-lived trend pullbacks or momentum bursts, and pushes orders through `CoinbaseService` with cross-margin leverage.

## Quick Start

1. **Dependencies** – ensure the repository environment is set up (`pip install -r requirements.txt`).
2. **Credentials** – provide working INTX API keys via `config.py`, environment variables, or a local `.env` (values must populate `API_KEY_PERPS` and `API_SECRET_PERPS`).
3. **Smoke Test (dry run)**
   ```bash
   python auto_zero_fee_trader.py --products BTC-PERP-INTX --granularity ONE_MINUTE --poll 30 --max-iterations 5
   ```
   This runs in paper mode by default and logs paper fills to `trade_logs/zero_fee_paper_trades.csv`.
4. **Live Launch** – once dry runs look good, enable live trading: 
   ```bash
   export AUTO_ZERO_FEE_BASE_USD=5
   export AUTO_ZERO_FEE_LEVERAGE=50
   python auto_zero_fee_trader.py --granularity ONE_MINUTE --poll 30 --live
   ```
   Drop the environment overrides if you prefer the built-in defaults (see below).

## Architecture Overview

```
auto_zero_fee_trader.py (CLI)
  └── IntradayTraderConfig.from_env()
        ├── SignalEngine (trend + pullback/breakout logic)
        ├── RiskManager (exposure + position cap)
        ├── CooldownTracker (per product+side)
        ├── PaperExecutionEngine or CoinbaseExecutionEngine
        └── ZeroFeePerpTrader
             ├── Fetches 1m candles via CoinbaseService / HistoricalData
             ├── Builds feature frame (EMA, RSI, ATR, volume ratio)
             ├── Evaluates entries
             ├── Monitors open positions for stops, targets, and max age
             └── Dispatches exit orders
```

Candles are always fresh-fetched for intraday use (`force_refresh=True`). Positions are tracked in-memory and evaluated each loop; exits are client-driven (market IOC orders with cross margin) rather than exchange-native brackets.

## Configuration Reference

The trader reads environment variables (prefixed `AUTO_ZERO_FEE_`) or falls back to constructor defaults. All values can also be passed via the CLI or edited directly in `IntradayTraderConfig`.

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_ZERO_FEE_PRODUCTS` | Top 15 INTX perps (`BTC`, `ETH`, `SOL`, `XRP`, `DOGE`, `ADA`, `BCH`, `LTC`, `AVAX`, `DOT`, `LINK`, `MATIC`, `ATOM`, `1000SHIB`, `APT`) | Comma-separated list of products to scan |
| `AUTO_ZERO_FEE_GRANULARITY` | `ONE_MINUTE` | Candle granularity (`ONE_MINUTE`, `TWO_MINUTE`, `THREE_MINUTE`, `FIVE_MINUTE`) |
| `AUTO_ZERO_FEE_POLL_SECONDS` | `60` | Loop sleep after each pass |
| `AUTO_ZERO_FEE_LOOKBACK_BARS` | `720` | Number of past bars requested per product |
| `AUTO_ZERO_FEE_COOLDOWN` | `180` | Seconds to wait before re-entering same product+side |
| `AUTO_ZERO_FEE_BASE_USD` | `5.0` | Base USD size before leverage multiplier |
| `AUTO_ZERO_FEE_LEVERAGE` | `50.0` | Cross leverage applied to each trade |
| `AUTO_ZERO_FEE_MAX_POSITIONS` | `2` | Maximum simultaneous open trades |
| `AUTO_ZERO_FEE_MAX_EXPOSURE` | `5000` | Max notional exposure (entry price × size) |
| `AUTO_ZERO_FEE_STOP_PCT` | `0.002` | Fixed stop-loss percentage if ATR guard inactive |
| `AUTO_ZERO_FEE_TP_PCT` | `0.0035` | Fixed take-profit percentage if ATR guard inactive |
| `AUTO_ZERO_FEE_TRAIL_ATR` | `1.2` | ATR multiple for stop/target clamps |
| `AUTO_ZERO_FEE_PULLBACK_PCT` | `0.0015` | Pullback threshold below fast EMA for longs (mirror for shorts) |
| `AUTO_ZERO_FEE_BREAKOUT_Z` | `1.2` | Z-score threshold for momentum breakout entries |
| `AUTO_ZERO_FEE_MIN_VOL_RATIO` | `1.1` | Required volume spike relative to rolling mean |
| `AUTO_ZERO_FEE_RSI_BUY` / `SELL` | `34` / `66` | RSI filters for long/short entries |
| `AUTO_ZERO_FEE_MAX_MINUTES` | `45` | Force-close positions older than this many minutes |
| `AUTO_ZERO_FEE_LIVE` | unset (`dry_run=True`) | Set to `1` to default the trader to live mode |

Any parameter is overridable via command-line flags where exposed (e.g., `--products`, `--poll`, `--granularity`, `--max-iterations`, `--live`).

## Runtime Flow

1. **Data Fetch** – downloads the most recent `lookback_bars` minutes for each product; cached data is bypassed to avoid stale ticks.
2. **Feature Engineering** – computes EMAs, z-score of returns, RSI, ATR, volume ratios; requires enough bars to satisfy the longest lookback window.
3. **Signal Evaluation** – long setup requires uptrend EMAs, a pullback (price below fast EMA) or positive breakout, depressed RSI, and elevated volume. Short setup mirrors the logic. If multiple rationales fire, they are combined in the `rationale` string.
4. **Risk & Cooldown** – the `RiskManager` ensures open notional stays below `AUTO_ZERO_FEE_MAX_EXPOSURE` and the number of active trades stays below `AUTO_ZERO_FEE_MAX_POSITIONS`. `CooldownTracker` blocks repeated entries on the same symbol+side until the cooldown expires.
5. **Execution** – dry run writes entries to CSV; live mode submits IOC market orders with the configured leverage and cross margin. Size is quantized to Coinbase’s base increment. Stops/targets are checked on every pass; hitting either triggers a market close.

## Testing & Validation

- Unit tests: `python -m unittest tests/test_intraday_zero_fee_trader.py`
- Dry-run smoke: `python auto_zero_fee_trader.py --products BTC-PERP-INTX --granularity ONE_MINUTE --poll 30 --max-iterations 5`
- Live smoke (single product, capped iterations):
  ```bash
  AUTO_ZERO_FEE_BASE_USD=2.5 \
  AUTO_ZERO_FEE_LEVERAGE=50 \
  AUTO_ZERO_FEE_PRODUCTS=BTC-PERP-INTX \
  python auto_zero_fee_trader.py --granularity ONE_MINUTE --poll 30 --max-iterations 6 --live
  ```
  Watch the console for "Placed live order" and confirm the fills inside Coinbase.

## Risk & Monitoring Checklist

- **Funding & Margin** – cross leverage defaults to 50×. Keep sufficient USDC in the INTX portfolio; Coinbase enforces minimum base sizes (~0.001 BTC) and rejects orders below the threshold.
- **Max Age** – positions are auto-liquidated after `AUTO_ZERO_FEE_MAX_MINUTES` (default 45). Tighten this if you want faster churn.
- **Guardrails** – adjust stop/take-profit percentages, ATR multiple, and cooldown to match volatility regimes. For high-event windows (Fed, CPI, etc.) consider raising `AUTO_ZERO_FEE_MIN_VOL_RATIO` or disabling the bot.
- **Logging** – execution logs flow to stdout and to `logs/short_term_crypto_finder/`. Paper trades append to `trade_logs/zero_fee_paper_trades.csv`.
- **Failsafes** – if the process dies, stops/targets are no longer enforced. Run inside `tmux` or systemd and set alerting around PnL and margin usage.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `PREVIEW_INSUFFICIENT_FUNDS_FOR_FUTURES` | INTX portfolio lacks margin for the requested notional | Add USDC collateral or reduce `AUTO_ZERO_FEE_BASE_USD`/`AUTO_ZERO_FEE_LEVERAGE` |
| `PREVIEW_INVALID_BASE_SIZE_TOO_SMALL` | Size below Coinbase minimum (e.g., < 0.001 BTC) | Increase base size or switch to a contract with smaller increments |
| "Failed to build features" | Not enough recent candles fetched | Leave the symbol in rotation—once sufficient data is available the error clears automatically |
| No trades for long periods | Filters too strict or market flat | Relax RSI/volume thresholds, lower cooldown, or expand product list |

## Extending the Strategy

- Swap in alternative signal logic by editing `SignalEngine.evaluate` (e.g., add VWAP reversion or funding-rate bias).
- Persist state across restarts (`ZeroFeePerpTrader.state_dir`) to resume open positions if desired.
- Hook up alerting/telemetry by subscribing to the execution logs or instrumenting the `enter`/`exit` methods.
- Integrate bracket orders (place take-profit/stop as resting orders) by reusing helpers from `coinbaseservice.place_market_order_with_targets` if you want exchange-side protection.

## Related Files

- `trading/intraday_zero_fee_trader.py` – core logic
- `auto_zero_fee_trader.py` – CLI wrapper
- `trade_btc_perp.py` – reference script for Coinbase perps with bracket orders
- `coinbaseservice.py` – Coinbase REST adapter

Keep this README alongside other strategy docs and update it whenever the signal logic, risk controls, or CLI interface change.
