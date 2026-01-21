# CLAUDE.md - Crypto Finance Development Guide

## Project Snapshot
- Mission: accelerate design, backtesting, and monitoring of systematic crypto trading strategies.
- Primary stack: Python 3.11+, `pandas`, `numpy`, `pandas-ta`, TA-Lib, `scikit-learn`, `xgboost`, `flask`, `websockets`, `openai`, plus `python-dotenv` for secrets.

## Project Structure
- `trading/`: trading models, portfolio utilities, visualization helpers.
- `technical_analysis/`: indicators, regime detection, and backtest engines.
- `tests/`: `unittest` suites (`test_*.py`), mirrors module layout.
- `docs/`: strategy research and design notes.
- `examples/`: minimal runnable demonstrations (e.g., `examples/backtest_ethereum.py`).
- Results and logs: `backtest_results/`, `trade_logs/`, `automated_trades_past/`, `volatility_analysis/`.
- Top-level scripts: `app.py`, `api.py`, `run_backtests.py`, `crypto_alert_monitor_*.py`, and similar entry points.
- Historical data stored in `candle_data/`.
- ML models cache in `cachedir/` and `catboost_info/`.

## Common Run Commands
- Basic backtesting: `python base.py --start_date 2023-01-01 --end_date 2023-12-31`
- Run with specific product: `python base.py --product_id ETH-USDC`
- Market analysis: `python market_analyzer.py`
- Advanced analysis: `python advanced_market_analyzer.py`
- Run multiple backtests: `python run_all_commands.py`
- UI/GUI version: `python market_ui.py`
- Environment setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## Test Commands
- Run all tests: `python -m unittest discover -s tests -v`
- Run specific test: `python -m unittest tests.test_<module>`
- Backtest smoke check: `python run_backtests.py` or `python examples/backtest_ethereum.py`
- UI/API preview: `python app.py` or `python api.py`

## Code Style Guidelines
- Follow PEP 8 with 4-space indentation; keep functions under ~50 lines when possible.
- **Imports**: stdlib > third-party > local; remove unused imports.
- **Naming**: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- **Types**: Use explicit typing (`from typing import List, Dict, Optional`, etc).
- **Documentation**: Succinct docstrings describing purpose, inputs, and outputs.
- **Error handling**: Use specific exception classes and contextual error messages.
- Favor pure or stateless helpers; when mutating shared state, add a short comment explaining why.
- Use descriptive logging via existing logging utilities rather than `print`.
- **Linting**: Run `flake8` manually - not currently in CI pipeline.

## Testing and Quality Assurance
- All new logic must have deterministic `unittest` coverage; mirror directory structure under `tests/`.
- Mock outbound APIs (`CoinbaseService`, `KrakenService`, websockets) and file I/O to keep tests offline.
- Test branches for both success and failure paths; include regression cases for fixed bugs.
- Keep tests fast (under one second when feasible) and independent; avoid relying on prior test state.
- Record skipped validations in handoff notes with justification.

## Data, Secrets, and Safety
- Runtime data lives in `candle_data/`, `trade_logs/`, `backtest_results/`, and `volatility_analysis/`; do not commit generated artifacts unless required.
- Load credentials via `.env` (e.g., `API_KEY`, `API_SECRET`, `NEWS_API_KEY`) using `python-dotenv`; never hardcode secrets or check them in.
- Disable live exchange calls in tests and examples unless explicitly sanctioned.
- When adding new data sources, document schema and retention in `docs/` and update any cleaners.

## Current Operational Pipelines

### Short-Term Finder
- Inputs: CoinbaseAdvanced (auth via `API_KEY`/`API_SECRET` or `API_KEY_PERPS`/`API_SECRET_PERPS`), fallback Kraken keys (`KRAKEN_API_KEY`/`KRAKEN_API_SECRET`), PEM secrets normalize `\n`.
- Gates: ATR7 USD cap (~3000), RR >= 2, swing close must break the level; liquidity filter (vol and vol/mcap), min mcap $100M.
- Cron: daily finder writing `finder_short.txt`; trades placed via `add_position_from_finder.py` or `add_top5_from_finder.py`; logs at `logs/short_term_crypto_finder/`.
- Commands: `python scripts/symbol_snapshot.py --symbols BTC,ETH,SOL,... --profile focused_no_llm_100`.

### Paper Trading (Baseline Tests)
- Helper: `scripts/baseline_finder_from_snapshot.py` turns snapshot picks into `finder_short_baseline.txt` with baseline ATR*RR exits; optional `--open-paper` hands off to `paper_finder_simulator.py`.
- Paper logs: `trade_logs/paper_finder_open_positions.csv` and `trade_logs/paper_finder_closed_positions.csv` for dashboard review.
- Finder-to-paper flow: `paper_finder_simulator.py open --finder-output finder_short.txt` (or `finder_long.txt`) parses finder text and stages paper trades; review them in `watchdog_dashboard.py`.
- After staging trades, run `python paper_finder_simulator.py update` so the dashboard reflects current P/L.
- Gate-scan cron runner: `scripts/run_gate_scan_paper.sh` automates gate-scan + paper trades; set `PYTHON_BIN`, `RUN_PAPER=0|1`, and `RUN_LIVE=0|1`, logs to `logs/gate_scan_paper.log`.

### Breakout Autotrade
- Scan cadence: hourly; symbols: BTC,ETH,SOL,XRP,ADA,DOT,AVAX,LINK,LTC,DOGE,OP,ARB,ATOM,UNI,AAVE,MKR,INJ (USDC quotes).
- Gates: swing close must clear trigger; RR >= 2; 24h lock after a trade (`.breakout_lock.json`); notional $500, 50x; near-breakout logging within ±0.5%.
- Exchange: primary `coinbaseadvanced` with auth; automatic unauth retry; fallback to Kraken with keys; timeout 30s.
- Outputs/logs: `finder_breakout.txt`, `logs/breakout_autotrade.log`; run via `python scripts/run_breakout_autotrade.py --timeframe 1h --lookback 50 --portfolio-usd 500 --leverage 50 --out finder_breakout.txt [--execute]`.

### Monitors / Housekeeping
- `watchdog_close_old_positions.py`: enforces 24h expiry on open positions (optional `--no-log-closures`); use in cron alongside finder/autotrader to avoid stale trades.
- `watchdog_dashboard.py`: status/metrics dashboard; keep it in sync with current pipelines for visibility over active trades and cron health.
- Dashboard health checks include log heartbeats (gate scan / paper update / fill poll / live snapshot). If a log is stale, flag pipeline health even if no trades closed.

### Observability / Scanning
- `scripts/symbol_snapshot.py --gate-scan`: scans the profile-filtered universe and prints the closest symbols to the RR/ATR gates (uses finder-tiered ATR caps and RR target; defaults RR=2, top=15). `--scan-limit N` optionally caps how many symbols are analyzed for speed. `--balanced` requires a mixed long/short set (min floor(top/2) per side).
- `scripts/watchdog_atr_clip_analysis.py`: buckets closed trades by `ATR_bps / cap_bps` to evaluate which volatility regimes produce better outcomes.
- `scripts/convert_to_parquet.py`: emits parquet copies for finder outputs, trade logs, and backtest CSVs.
- `scripts/duckdb_query.py`: run ad-hoc SQL over parquet/CSV diagnostics (via DuckDB).
- `scripts/paper_trade_progress.py`: quick status on paper trades (count vs target, win%/avg%, expectancy, TP/SL/expiry split).
- `scripts/paper_equity_report.py`: shareable equity report (HTML + PNG). Uses daily aggregation, so one day = one point and drawdown is 0% with a single day.
- Cron snapshot for reproducible setup: `docs/pipelines/cron.md` (install with `crontab docs/pipelines/cron.md`, then update paths/envs for the new host).
- Daily stop streak guard: `scripts/daily_stop_guard.py` records live daily stop hits in `logs/daily_stop_history.json` and enforces pause/warn thresholds configured in `config/risk_thresholds.yaml`.

## Gate Definitions (quick reference)
- **ATR gate**: ATR7 USD cap (current default ~3000) blocks symbols with larger recent ranges.
- **RR gate**: RR >= 2 required to accept a breakout; RR is logged per candidate, with PASS/SKIP.
- **Breakout condition**: candle close must be strictly beyond swing high/low (dist=0% is only a touch).
- **Near-breakout**: |dist| <= 0.5% from swing trigger; logged but not tradable.

## Operational Triage
- If no symbols: check liquidity filters, ATR cap, exchange connectivity; see logs in `logs/short_term_crypto_finder/` or `logs/breakout_autotrade.log`.
- Exchange errors: CoinbaseAdvanced may time out; code retries unauthenticated then falls back to Kraken with keys; confirm env vars present.
- RR skips: Scanner prints "RR-skipped" block when a level breaks but RR < threshold.
- Range-break guard is latched until a confirmed daily close re-enters range±buffer (intraday moves do not clear). Use `range_break_confirmed_only` to control whether the trigger itself is confirmed-close-only.

## API Usage Notes
- Coinbase API has rate limits (~15 requests/sec).
- Use error handling with exponential backoff for API calls.
- Cache responses when possible to reduce API usage.
- Monitor API key permissions and rotate regularly.
- Fallback mechanisms in place for API outages.

## Common Debugging Tips
- `TechnicalAnalysisError`: Usually from insufficient price data.
- Connection errors: Check API credentials in `.env`.
- Model errors: Ensure `cachedir` permissions are correct.
- TA-Lib errors: Verify TA-Lib installation for your OS.
- Candle data issues: Check `historicaldata.py` logs.

## Workflow Guidelines
- Before implementing, clarify task intent, scope, and success criteria.
- Keep diffs minimal and well-scoped.
- When adding or changing features/pipelines, update relevant docs (`CLAUDE.md`, `README_*.md`, `TRADING_PROGRESSION.md`) in the same patch.
- When changing risk guards (daily stop, range-break, baseline sizing/caps), update `config/risk_thresholds.yaml` and keep docs in sync.
- When adding journal entries to `TRADING_PROGRESSION.md`, keep entries ordered newest-first within each month section.
- Be explicit about assumptions, especially around time ranges and data availability.
- Reference paths with filenames and line numbers when flagging issues or proposing edits.
- Surface risks early (data staleness, long-running backtests, TA-Lib availability).
- Favor incremental PR-sized changes; document follow-ups if scope must be split.
- End deliverables with next-step suggestions (tests to run, deployment actions, monitoring reminders).

## Legacy / Not in Active Pipeline
- Legacy bots/tools (kept for reference, not maintained): `base.py`, `simplified_trading_bot*.py` (v1.2.x), older HFT/mean-reversion runners, `backtest_trading_bot_past`, and other retired helpers under `retired_tools/`.
- These are outside the current finder + breakout pipelines and should not be modified unless explicitly requested.
