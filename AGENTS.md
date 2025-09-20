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
- Visualization: helpers in `trading/visualization`, matplotlib scripts under the repository root (for example `plot_atr_histogram.py`).
- Additional context: review notebooks and summaries in `docs/` and prior analyses in `README_*.md` files.

## Communication and Delivery Expectations
- Be explicit about assumptions, especially around time ranges and data availability.
- Reference paths with filenames and line numbers when flagging issues or proposing edits.
- Surface risks early (data staleness, long-running backtests, TA-Lib availability).
- Favor incremental pull-request sized changes; document follow-ups if scope must be split.
- End deliverables with next-step suggestions (tests to run, deployment actions, monitoring reminders).
