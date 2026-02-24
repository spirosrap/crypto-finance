# CLAUDE.md - Crypto Finance Development Guide

## Project Structure
- `trading/`: models, portfolio utils, visualization.
- `technical_analysis/`: indicators, regime detection, backtest engines.
- `tests/`: `unittest` suites mirroring module layout.
- `scripts/`: operational pipeline scripts.
- `docs/`: strategy research; `docs/pipelines/cron.md` for cron snapshot.
- Data/logs: `candle_data/`, `trade_logs/`, `backtest_results/`, `volatility_analysis/`.
- ML cache: `cachedir/`, `catboost_info/`.

## Code Style
- PEP 8, 4-space indent, functions ≤ ~50 lines, `snake_case`/`PascalCase`/`UPPER_SNAKE_CASE`.
- Imports: stdlib > third-party > local. Use explicit typing. Logging over `print`. Run `flake8` manually.

## Testing
- `python -m unittest discover -s tests -v` | specific: `python -m unittest tests.test_<module>`
- Mock all outbound APIs (`CoinbaseService`, `KrakenService`, websockets) and file I/O; tests must be offline and fast.

## Secrets & Safety
- Credentials in `.env` (`API_KEY`, `API_SECRET`, `API_KEY_PERPS`, `API_SECRET_PERPS`, `KRAKEN_API_KEY`, `KRAKEN_API_SECRET`, `NEWS_API_KEY`); PEM secrets normalize `\n`. Never hardcode.
- Do not commit generated artifacts (`trade_logs/`, `backtest_results/`, etc.) unless explicitly requested for reporting or reproducibility.
- Disable live exchange calls in tests/examples unless explicitly sanctioned.

## Active Pipelines

### Short-Term Finder
- Daily cron → `finder_short.txt`; trades via `add_position_from_finder.py` or `add_top5_from_finder.py`.
- Run: `python scripts/symbol_snapshot.py --symbols BTC,ETH,SOL,... --profile focused_no_llm_100`
- Logs: `logs/short_term_crypto_finder/`

### Paper Trading
- Stage: `python paper_finder_simulator.py open --finder-output finder_short.txt`
- Update P/L: `python paper_finder_simulator.py update`
- Baseline exits: `scripts/baseline_finder_from_snapshot.py` → `finder_short_baseline.txt`
- Logs: `trade_logs/paper_finder_open_positions.csv` (runtime/local, gitignored), `trade_logs/paper_finder_closed_positions.csv` (tracked for review/dashboard history).
- Cron runner: `scripts/run_gate_scan_paper.sh` (env vars: `PYTHON_BIN`, `RUN_PAPER`, `RUN_LIVE`)

### Breakout Autotrade
- Hourly scan; symbols: BTC,ETH,SOL,XRP,ADA,DOT,AVAX,LINK,LTC,DOGE,OP,ARB,ATOM,UNI,AAVE,MKR,INJ (USDC).
- Run: `python scripts/run_breakout_autotrade.py --timeframe 1h --lookback 50 --portfolio-usd 500 --leverage 50 --out finder_breakout.txt [--execute]`
- 24h trade lock: `.breakout_lock.json`; logs: `logs/breakout_autotrade.log`

### Monitors
- `watchdog_close_old_positions.py`: 24h position expiry (optional `--no-log-closures`).
- `watchdog_dashboard.py`: active trades + cron health. Flags stale logs even if no trades closed.

### Observability
- Gate scan: `scripts/symbol_snapshot.py --gate-scan [--scan-limit N] [--balanced]`
- Paper status: `scripts/paper_trade_progress.py` | Equity report: `scripts/paper_equity_report.py`
- ATR analysis: `scripts/watchdog_atr_clip_analysis.py`
- Ad-hoc SQL: `scripts/duckdb_query.py` (DuckDB over parquet/CSV)
- Daily stop guard: `scripts/daily_stop_guard.py` → `logs/daily_stop_history.json`; thresholds in `config/risk_thresholds.yaml`

## Gate Definitions
- **ATR gate**: ATR7 USD cap ~3000; blocks high-volatility symbols.
- **RR gate**: RR >= 2 required; logged as PASS/SKIP per candidate.
- **Breakout**: candle close strictly beyond swing high/low (dist=0% = touch only, not tradable).
- **Near-breakout**: |dist| <= 0.5%; logged but not traded.

## Triage
- No symbols → check liquidity filter, ATR cap, exchange connectivity; see pipeline logs.
- Exchange timeout → code retries unauth then falls back to Kraken; confirm env vars.
- RR skips → "RR-skipped" block in logs when level breaks but RR < 2.
- Range-break guard → latched until confirmed daily close re-enters range; intraday moves don't clear.
- `TechnicalAnalysisError` → insufficient price data. Candle issues → check `historicaldata.py` logs.

## Workflow
- Update `CLAUDE.md`, `TRADING_PROGRESSION.md` (newest-first), and `config/risk_thresholds.yaml` in the same patch as relevant changes.
- Keep diffs minimal; reference file paths with line numbers when flagging issues.

## Legacy (do not modify unless asked)
- `base.py`, `simplified_trading_bot*.py`, older HFT/mean-reversion runners, `retired_tools/`.
