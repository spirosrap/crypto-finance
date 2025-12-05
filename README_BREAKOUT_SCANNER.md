# Breakout Scanner

Simple swing-breakout scanner for liquid crypto majors. It looks for breaks of recent highs/lows with volume thrust, builds structure-based stops, sets 2R targets, and emits finder-style output you can feed to `add_position_from_finder.py` to place trades quickly.

## What it does
- Pulls OHLCV via CCXT (Coinbase Advanced as primary, Kraken as fallback using keys from `.env`).
- Detects long/short breakouts above/below recent swing levels (default lookback 50 bars).
- Sets stops just beyond recent structure and TPs at 2× risk.
- Ranks candidates by RR, volume thrust (3-bar vs 20-bar average), and ADX.
- Writes finder-formatted blocks (entry/SL/TP) to a file for direct ingestion.

## Usage
```bash
python scripts/breakout_scanner.py \
  --symbols BTC,ETH,SOL,XRP,ADA,DOT,AVAX,MATIC,LINK,LTC,DOGE,USDT,USDC \
  --timeframe 4h \
  --lookback 50 \
  --out finder_breakout.txt
```

Notes:
- If the requested timeframe isn’t supported by the exchange (e.g., 4h on Coinbase), it will auto-fallback to 1h and warn.
- Self-quotes like `USDC/USDC` are skipped automatically.
- MATIC/USDC is not available on Coinbase; it will warn and skip.

## Parameters
- `--symbols`: comma-separated list (default majors). If no quote is given, `/USDC` is assumed (e.g., `BTC` → `BTC/USDC`).
- `--timeframe`: CCXT timeframe (default `4h`). Falls back if unsupported.
- `--lookback`: bars to define swing high/low (default `50`).
- `--exchange`: primary CCXT exchange id (default `coinbaseadvanced`; falls back to `kraken`).
- `--out`: output file path in finder format (default `finder_breakout.txt`).

## Output (finder format)
Each candidate is written as:
```
N. SYMBOL — LONG/SHORT
Data Timestamp (UTC): 2025-12-05 18:00:00+00:00
TRADING LEVELS (LONG/SHORT)
Entry Price: $12345.67
Stop Loss: $12000.00
Take Profit: $12700.00
Recommended Position Size: 0.0%
RR=2.00  vol_thrust=1.35  trend=0.012  adx=24.5
----------------------------------------------------------------
```
You can feed this file to `add_position_from_finder.py` to prepare orders.

## Keys and exchanges
- Coinbase Advanced (primary): uses `API_KEY` / `API_SECRET` from `.env` if present.
- Kraken (fallback): uses `KRAKEN_API_KEY` / `KRAKEN_API_SECRET` from `.env` if present.
- Timeout is set to 30s; retries handled via the fallback mechanism in `long_term_crypto_finder.py`.

## Caveats
- This scanner enforces a fixed 2R target; adjust the code if you want different RR.
- It does not apply the full scoring/filters of `short_term_crypto_finder.py`; it’s a lightweight breakout-only view.
- Results depend on the exchange’s supported timeframes and available markets.

