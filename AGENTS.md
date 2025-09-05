# Repository Guidelines

## Project Structure & Module Organization
- `trading/`: Core trading utilities (models, visualization, settings).
- `technical_analysis/`: Indicators, backtests, and analysis helpers.
- `tests/`: Unit tests (unittest). Add new tests here as `test_*.py`.
- `docs/`: Strategy notes and research artifacts.
- `examples/`: Minimal runnable examples (e.g., `examples/backtest_doge.py`).
- Results/data: `backtest_results/`, `trade_logs/`, `automated_trades_past/`, `volatility_analysis/`.
- Apps/scripts: top‑level Python files (e.g., `app.py`, `api.py`, `run_backtests.py`, monitors under `crypto_alert_monitor_*.py`).

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Run tests: `python -m unittest discover -s tests -v`
- Example backtest: `python run_backtests.py` or `python examples/backtest_ethereum.py`
- Launch UI/API (if used): `python app.py` or `python api.py`

## Coding Style & Naming Conventions
- Python, 4‑space indentation, PEP 8.
- Names: modules/files `snake_case.py`, functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Prefer type hints and short, purposeful docstrings.
- Imports grouped: stdlib, third‑party, local. Avoid unused imports.
- Formatting: follow existing style; using `black`/`ruff` locally is OK but not enforced.

## Testing Guidelines
- Framework: `unittest` (no network or live API calls).
- Place tests in `tests/` with `test_*.py`; mirror module paths when possible.
- Mock external services (e.g., `CoinbaseService`, `RESTClient`) and file I/O.
- Cover new code paths and failure cases; keep tests fast and deterministic.
- Run: `python -m unittest discover -s tests -v`

## Commit & Pull Request Guidelines
- Commits: present tense, imperative, concise.
  - Optional scope prefix: `trading: ...`, `technical_analysis: ...`, `service: ...`, `docs: ...`
  - Example: `technical_analysis: add RSI calculation edge-case handling`
- PRs: clear description, motivation, and testing notes; link issues.
  - Include screenshots for UI changes (`app.py`, `market_ui*.py`).
  - Checklist: passes tests, no secrets, updated docs/examples if behavior changes.

## Security & Configuration Tips
- Secrets via environment (`.env`) using `python-dotenv`; never commit keys.
  - Expected: `API_KEY`, `API_SECRET`, `NEWS_API_KEY`, exchange tokens.
- Avoid hitting exchange APIs in tests; prefer fixtures/mocks.
- Data/logs live under `trade_logs/` and results under `backtest_results/`.
