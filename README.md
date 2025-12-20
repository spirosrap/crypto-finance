# Crypto Finance — Current Pipeline (Dec 2025)

## Current Focus
- **Short-Term Crypto Finder (`short_term_crypto_finder.py`)**: Strict RR ≥ 2, ATR(7) gated with tiered caps (3k USD plus bps tiers ≈325/350/400/450 by price bands; tighter cap wins), hard SL/TP. Runs on USDC pairs; logs to `logs/short_term_crypto_finder/`.
- **Breakout Autotrade (`scripts/run_breakout_autotrade.py`)**: Hourly scan of USDC majors (BTC/ETH/SOL/XRP/ADA/DOT/AVAX/LINK/LTC/DOGE/OP/ARB/ATOM/UNI/AAVE/MKR/INJ) with fixed 2R structure, $500 notional / 50x, 24h lock to avoid stacking. Writes finder-format output; uses Coinbase primary with Kraken fallback. Near-breakouts are logged.
- **Safety First**: Downtime is expected; no trades when RR/ATR gates fail. Flat is acceptable.

## Active Tools
- **Finders**: `short_term_crypto_finder.py`, `long_term_crypto_finder.py`.
  - Short-horizon snapshot: `scripts/symbol_snapshot.py` (includes short-term gates + intraday context).
    - Includes ATR-multiple readout for SL/TP distance and an `RR drivers` line showing which caps/floors dominated.
  - Long-horizon snapshot: `scripts/long_term_snapshot.py` (daily candles + long-term indicators like ATR(14)/Sharpe/drawdown).
    - Gate scan (long-term): `python scripts/long_term_snapshot.py --gate-scan --profile wide --top 15 --scan-limit 200`
  - Gate proximity view (short-term): `python scripts/symbol_snapshot.py --gate-scan [--scan-limit N]`.
  - Long-term profile used for LLM + liquidity filters: `python long_term_crypto_finder.py --profile focused_llm_100 --plain-output finder_long.txt --suppress-console-logs`
- **Breakout Suite**: `scripts/breakout_scanner.py` (finder-style output, near-breakout logs), `scripts/run_breakout_autotrade.py` (cron-friendly runner with lock).
- **Watchdogs/Closers**: `watchdog_close_old_positions.py` (optional 24h timeout), `watchdog_dashboard.py` for monitoring (Streamlit).
- **Support**: `add_position_from_finder.py` to stage/execute trades from finder-format text.
- **Diagnostics**: `scripts/watchdog_atr_clip_analysis.py` to bucket closed trades by ATR cap ratio and compare outcomes.

## Pipeline Map (Current)
- Inputs: Coinbase/CCXT candles + CoinGecko metrics + live spread/fee bps (when available).
- Short-term flow: `short_term_crypto_finder.py` -> `finder_short.txt` -> `add_position_from_finder.py`/`add_top5_from_finder.py` -> `trade_logs/` + watchdogs.
- Breakout flow: `scripts/run_breakout_autotrade.py` -> `finder_breakout.txt` -> (optional execute) -> `trade_logs/`.
- Long-term flow: `long_term_crypto_finder.py` -> `finder_long.txt` -> discretionary review + sizing.
- Monitoring: `watchdog_close_old_positions.py` (expiry) + `watchdog_dashboard.py` (equity/health).

## Runbook (Daily/Weekly)

### Daily
1) Snapshot and gate-scan:
   - `python scripts/symbol_snapshot.py --symbols BTC,ETH --profile focused_no_llm_100`
   - `python scripts/symbol_snapshot.py --gate-scan --profile focused_no_llm_100 --top 15 --scan-limit 100`
2) Short-term finder (manual or cron):
   - `python short_term_crypto_finder.py --profile focused_llm_100 --plain-output finder_short.txt --force-refresh`
   - Expect: updated `finder_short.txt` and logs under `logs/short_term_crypto_finder/`.
3) Breakout autotrade (cron) and log review:
   - Check `logs/breakout_autotrade.log` for near-breakouts or triggers.
4) Optional long-term scan:
   - `python long_term_crypto_finder.py --profile focused_llm_100 --plain-output finder_long.txt --suppress-console-logs`

### Weekly
- ATR regime diagnostics:
  - `conda run -n trade python scripts/watchdog_atr_clip_analysis.py --input trade_logs/watchdog_closed_positions.csv`
- Dashboard review:
  - `watchdog_dashboard.py` for equity curve and log health.

## Outputs and Logs
| Artifact | Producer | Purpose | Location | Notes |
| --- | --- | --- | --- | --- |
| `finder_short.txt` | `short_term_crypto_finder.py` | Short-term trade candidates | repo root | Finder-format for `add_position_from_finder.py`. |
| `finder_long.txt` | `long_term_crypto_finder.py` | Long-term candidates | repo root | Discretionary review. |
| `finder_breakout.txt` | `scripts/run_breakout_autotrade.py` | Breakout candidates | repo root | Written on each run. |
| `logs/short_term_crypto_finder/` | finder | Short-term run logs | `logs/` | Rotates with retention. |
| `logs/long_term_crypto_finder/` | finder | Long-term run logs | `logs/` | Rotates with retention. |
| `logs/breakout_autotrade.log` | autotrader | Breakout run log | `logs/` | Near-breakouts + triggers. |
| `trade_logs/watchdog_closed_positions.csv` | watchdog | Closed trade ledger | `trade_logs/` | Source for ATR clip analysis. |
| `trade_logs/watchdog_tp_sl_checkpoint.json` | watchdog | Open trade checkpoints | `trade_logs/` | Used for recovery. |

## What’s Retired/Optional
- Forced daily trading or balanced 5-trade baskets—no longer required.
- Mandatory 24h holds across all strategies—used only when explicitly configured (e.g., breakout autotrade lock).
- Reservoir/multi-basket experiments—on pause; sticking to finder + breakout playbooks.
- Legacy bots/tools (not maintained): simplified_trading_bot v1.2.1f, simplified_trading_bot_past/simplified_trading_bot_v1.2.2.py, reservoir/multi-basket scripts. Kept for reference only.

## Quick Start (Current Flow)
1) Run snapshots: `python scripts/symbol_snapshot.py --symbols BTC,ETH --profile focused_no_llm_100`
   - Gate proximity view (optional): `python scripts/symbol_snapshot.py --gate-scan --profile focused_no_llm_100 --top 15 --scan-limit 100`
   - Long-term sanity check (optional): `python scripts/long_term_snapshot.py --symbols BTC,ETH --profile default`
   - Long-term gate scan (optional): `python scripts/long_term_snapshot.py --gate-scan --profile wide --top 15 --scan-limit 200`
2) Let cron handle:
   - `short_term_crypto_finder.py` (daily) → feed results to `add_position_from_finder.py` or `add_top5_from_finder.py` to stage/execute trades.
   - `run_breakout_autotrade.py` (hourly) for 2R breakouts on USDC majors.
3) Check `logs/breakout_autotrade.log` for near-breakouts/triggers; finder logs under `logs/short_term_crypto_finder/`.
4) Optional: enforce a 24h timeout on positions with `watchdog_close_old_positions.py`.

## Environment & Requirements
- Python 3.11, ccxt ≥ 4.2, pandas ≥ 2.3, numpy ≥ 1.24, TA-Lib ≥ 0.6.7, openai ≥ 1.109.1 (installed), full list in `requirements.txt`.
- Configure API keys in `.env` (Coinbase primary: `API_KEY`/`API_SECRET`; Kraken fallback: `KRAKEN_API_KEY`/`KRAKEN_API_SECRET`). PEM secrets normalize `\\n`.

## Setup (Current)

```bash
conda create -n trade python=3.11
conda activate trade
python scripts/install_requirements.py
```

Notes:
- macOS: Homebrew is required for TA-Lib.
- Windows: install TA-Lib manually, then rerun `scripts/install_requirements.py`.

## API Keys (Current)
- Prefer `.env` or `config.py`; never commit secrets.
- Coinbase (primary): `API_KEY` / `API_SECRET` or perps keys `API_KEY_PERPS` / `API_SECRET_PERPS`.
- Kraken fallback: `KRAKEN_API_KEY` / `KRAKEN_API_SECRET`.
- LLM scoring (optional): `OPENAI_API_KEY`.

## Legacy / Archive
- Detailed setup and older tool documentation moved to `README_LEGACY.md`.
