# AGENTS.md - Crypto Finance AI Playbook

## Project Snapshot
- Mission: accelerate design, backtesting, and monitoring of systematic crypto trading strategies.
- Primary stack: Python 3.11+, `pandas`, `numpy`, `pandas-ta`, TA-Lib, `scikit-learn`, `xgboost`, `flask`, `websockets`, `openai`, plus `python-dotenv` for secrets.
- Key directories:
  - `trading/`: trading models, portfolio utilities, visualization helpers.
  - `technical_analysis/`: indicators, regime detection, and backtest engines.
  - `tests/`: `unittest` suites (`test_*.py`), mirrors module layout.
  - `docs/`: strategy research and design notes.
  - `examples/`: minimal runnable demonstrations (for example `examples/backtest_ethereum.py`).
  - Results and logs: `backtest_results/`, `trade_logs/`, `automated_trades_past/`, `volatility_analysis/`.
  - Top-level scripts: `app.py`, `api.py`, `run_backtests.py`, `crypto_alert_monitor_*.py`, and similar entry points.

## Agent Collaboration Framework
- **Planner / Analyst**
  - Clarify task intent, scope, success criteria, and deadlines when relevant.
  - Identify affected modules, data sources, and operational risks before work starts.
  - Produce concise execution notes for the Builder (inputs, expected outputs, validation plan).
- **Builder / Implementer**
  - Follow the Planner notes; keep diffs minimal and well-scoped.
  - Update or create code with clear docstrings and targeted comments when logic is subtle.
  - Stage supporting assets (fixtures, docs) alongside code changes.
  - When adding or changing features/pipelines, update the relevant docs (`AGENTS.md`, `README_*.md`, `SPIROS_TRADING_PROTOCOL.MD`, `TRADING_PROGRESSION.md`) in the same patch so the playbook stays current.
- **Reviewer / QA**
  - Audit diffs for logic issues, edge cases, and style compliance.
  - Ensure tests cover the updated paths and flag gaps or regressions.
  - Confirm artifacts (logs, notebooks, configs) stay out of version control unless required.
- **Researcher / Context Agent**
  - Gather only documented, offline-safe references (historical logs, existing analyses).
  - Summarize findings with file pointers or citations for traceability.
- Handoff checklist for every role: state assumptions, list commands executed, and record any validations skipped.

## Core Workflows
- **Environment setup**
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- **Implement and iterate**
  - Inspect relevant modules with `rg`, `ls`, `sed`, or your editor of choice.
  - Keep functions focused; prefer dependency injection over global state when feasible.
  - Document configuration knobs in `docs/` or inline docstrings.
- **Validation**
  - Unit tests: `python -m unittest discover -s tests -v`
  - Targeted test: `python -m unittest tests.test_<module>`
  - Backtest smoke check: `python run_backtests.py` or `python examples/backtest_ethereum.py`
  - UI/API preview when needed: `python app.py` or `python api.py`
- **Before handoff**
  - Summarize code changes and their rationale.
  - Note any files intentionally untouched but relevant for future follow-up.

## Coding Standards and Style
- Follow PEP 8 with 4-space indentation; keep functions under roughly 50 lines when possible.
- Naming: modules and functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Imports grouped stdlib -> third-party -> local; remove unused imports.
- Prefer type hints and succinct docstrings describing purpose, inputs, and outputs.
- Favor pure or stateless helpers; when mutating shared state, add a short comment explaining why.
- Use descriptive logging via the existing logging utilities rather than `print`.

## Testing and Quality Assurance
- All new logic must have deterministic `unittest` coverage; mirror directory structure under `tests/`.
- Mock outbound APIs (`CoinbaseService`, `KrakenService`, websockets) and file I/O to keep tests offline.
- Test branches for both success and failure paths; include regression cases for fixed bugs.
- Keep tests fast (under one second when feasible) and independent; avoid relying on prior test state.
- Record skipped validations in the handoff notes with justification.

## Data, Secrets, and Safety
- Runtime data lives in `candle_data/`, `trade_logs/`, `backtest_results/`, and `volatility_analysis/`; do not commit generated artifacts unless required.
- Load credentials via `.env` (for example `API_KEY`, `API_SECRET`, `NEWS_API_KEY`) using `python-dotenv`; never hardcode secrets or check them in.
- Disable live exchange calls in tests and examples unless explicitly sanctioned.
- When adding new data sources, document schema and retention in `docs/` and update any cleaners.

## Tooling and References
- Diagnostics: `market_analyzer.py`, `advanced_market_analyzer.py`, `continuous_market_monitor.py`.
- Strategy runners: `run_backtests.py`, `run_all_commands.py`, `backtest_trading_bot.py`.
- Monitoring scripts: `crypto_alert_monitor_*.py`, `trade_guardian.py`, `trade_regime_guardian.py`.
- Visualization: helpers in `trading/visualization.py`, matplotlib scripts under the repository root (for example `plot_atr_histogram.py`).
- Additional context: review notebooks and summaries in `docs/` and prior analyses in `README_*.md` files.

## Current Operational Pipelines
- **Short-Term Finder**
  - Inputs: CoinbaseAdvanced (auth via `API_KEY`/`API_SECRET` or `API_KEY_PERPS`/`API_SECRET_PERPS`), fallback Kraken keys (`KRAKEN_API_KEY`/`KRAKEN_API_SECRET`), PEM secrets normalize `\\n`.
  - Gates: ATR7 USD cap (currently ~3000), RR >= 2, swing close must break the level; liquidity filter (vol and vol/mcap), min mcap $100M.
  - Cron: daily finder writing `finder_short.txt`; trades placed via `add_position_from_finder.py` or `add_top5_from_finder.py`; logs at `logs/short_term_crypto_finder/`.
  - Commands: `python scripts/symbol_snapshot.py --symbols BTC,ETH,SOL,... --profile focused_no_llm_100`.
- **Breakout Autotrade**
  - Scan cadence: hourly; symbols: BTC,ETH,SOL,XRP,ADA,DOT,AVAX,LINK,LTC,DOGE,OP,ARB,ATOM,UNI,AAVE,MKR,INJ (USDC quotes).
  - Gates: swing close must clear trigger; RR >= 2; 24h lock after a trade (`.breakout_lock.json`); notional $500, 50x; near-breakout logging within ±0.5%.
  - Exchange: primary `coinbaseadvanced` with auth; automatic unauth retry; fallback to Kraken with keys; timeout 30s.
  - Outputs/logs: `finder_breakout.txt`, `logs/breakout_autotrade.log`; run via `python scripts/run_breakout_autotrade.py --timeframe 1h --lookback 50 --portfolio-usd 500 --leverage 50 --out finder_breakout.txt [--execute]`.
- **Monitors / Housekeeping**
  - `watchdog_close_old_positions.py`: enforces 24h expiry on open positions (optional `--no-log-closures`); use in cron alongside finder/autotrader to avoid stale trades.
  - `watchdog_dashboard.py`: status/metrics dashboard; keep it in sync with current pipelines for visibility over active trades and cron health.
- **Observability / Scanning**
  - `scripts/symbol_snapshot.py --gate-scan`: scans the profile-filtered universe and prints the closest symbols to the RR/ATR gates (uses finder-tiered ATR caps and RR target; defaults RR=2, top=15). `--scan-limit N` optionally caps how many symbols are analyzed for speed. Profiles are filter presets (not fixed product lists); scan-limit just limits breadth within that preset.
  - `scripts/watchdog_atr_clip_analysis.py`: buckets closed trades by `ATR_bps / cap_bps` to evaluate which volatility regimes produce better outcomes.

## Communication and Delivery Expectations
- Be explicit about assumptions, especially around time ranges and data availability.
- Reference paths with filenames and line numbers when flagging issues or proposing edits.
- Surface risks early (data staleness, long-running backtests, TA-Lib availability).
- Favor incremental pull-request sized changes; document follow-ups if scope must be split.
- End deliverables with next-step suggestions (tests to run, deployment actions, monitoring reminders).

## Gate Definitions (quick reference)
- ATR gate: ATR7 USD cap (current default ~3000) blocks symbols with larger recent ranges.
- RR gate: RR >= 2 required to accept a breakout; RR is logged per candidate, with PASS/SKIP.
- Breakout condition: candle close must be strictly beyond swing high/low (dist=0% is only a touch).
- Near-breakout: |dist| <= 0.5% from swing trigger; logged but not tradable.

## Operational Triage
- If no symbols: check liquidity filters, ATR cap, exchange connectivity; see logs in `logs/short_term_crypto_finder/` or `logs/breakout_autotrade.log`.
- Exchange errors: CoinbaseAdvanced may time out; code retries unauthenticated then falls back to Kraken with keys; confirm env vars present.
- RR skips: Scanner prints “RR-skipped” block when a level breaks but RR < threshold.

## Legacy / Not in Active Pipeline
- Legacy bots/tools (kept for reference, not maintained): `base.py`, `simplified_trading_bot*.py` (v1.2.x), older HFT/mean-reversion runners, `backtest_trading_bot_past`, and other retired helpers under `retired_tools/`.
- These are outside the current finder + breakout pipelines and should not be modified unless explicitly requested.
