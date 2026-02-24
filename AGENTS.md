# AGENTS.md - Crypto Finance (Concise)

## Scope
This repo supports systematic crypto strategy research, execution, and monitoring.
Primary stack: Python 3.11+, pandas/numpy, TA-Lib, scikit-learn/xgboost, Flask/websockets.

## Key Paths
- `trading/`: models, portfolio logic, visualization helpers
- `technical_analysis/`: indicators, regime detection, backtest engines
- `tests/`: `unittest` suites (`test_*.py`)
- `docs/`: plans, strategy notes, pipeline docs
- `trade_logs/`, `backtest_results/`, `volatility_analysis/`: runtime outputs
- Core scripts: `watchdog_close_old_positions.py`, `watchdog_dashboard.py`, `scripts/symbol_snapshot.py`

## Must-Follow Rules
- Keep diffs minimal and scoped to the request.
- Prefer deterministic tests for new/fixed logic.
- Use existing logging utilities; avoid `print` for production paths.
- Never hardcode secrets. Use `.env` via `python-dotenv`.
- Do not modify legacy tools unless explicitly requested.
- If changing risk guards (daily stop, range-break, sizing/caps), update `config/risk_thresholds.yaml` and docs.

## Standard Workflow
1. Clarify scope, affected files, and risks.
2. Implement minimal change.
3. Run targeted validation (and broader tests when needed).
4. Update docs in the same patch when behavior/workflows changed.
5. Handoff with assumptions, commands run, and any skipped checks.

## Validation Commands
- Full tests: `python -m unittest discover -s tests -v`
- Targeted: `python -m unittest tests.test_<module> -v`
- Backtest smoke: `python run_backtests.py`
- API/UI preview: `python api.py` or `python app.py`

## Active Pipelines
### 1) Short-Term Finder
- Gate scan: `python scripts/symbol_snapshot.py --gate-scan --profile focused_no_llm_100 --top 15 --scan-limit 100`
- Snapshot: `python scripts/symbol_snapshot.py --symbols BTC,ETH,SOL --profile focused_no_llm_100`
- Core gates: ATR cap + RR >= 2 + confirmed breakout close.

### 2) Paper Baseline Flow
- Build baseline picks from snapshot: `scripts/baseline_finder_from_snapshot.py`
- Stage/update paper trades: `paper_finder_simulator.py open ...` then `paper_finder_simulator.py update`
- Logs: `trade_logs/paper_finder_open_positions.csv`, `trade_logs/paper_finder_closed_positions.csv`

### 3) Breakout Autotrade
- Runner: `python scripts/run_breakout_autotrade.py ... [--execute]`
- Outputs: `finder_breakout.txt`, `logs/breakout_autotrade.log`
- Core protections: RR >= 2, strict close-through trigger, lock window.

### 4) Watchdogs / Monitoring
- Close stale positions: `watchdog_close_old_positions.py`
- Dashboard: `watchdog_dashboard.py`
- Daily stop streak tracking: `scripts/daily_stop_guard.py` -> `logs/daily_stop_history.json`
- Dashboard health check must include heartbeat freshness (gate scan / paper update / fill poll / live snapshot).

## Risk/Guard Notes
- Range-break guard stays latched until confirmed daily close re-enters range±buffer.
- `range_break_confirmed_only` controls trigger confirmation behavior.

## Documentation Rules
When features or workflows change, update relevant docs in the same patch:
- `README*.md`
- `SPIROS_TRADING_PROTOCOL.MD`
- `TRADING_PROGRESSION.md`
- `AGENTS.md` (this file)

For `TRADING_PROGRESSION.md`:
- Add entries newest-first within the month.
- Update that month’s “State at a glance” summary too.

## Data / Commit Hygiene
- Runtime artifacts are usually not committed unless requested/required.
- Keep commits focused; avoid mixing unrelated files.

## Quick Triage
- No symbols found: inspect liquidity filter, ATR cap, RR gate, exchange connectivity/logs.
- Exchange/API issues: verify env vars and fallback behavior.
- RR skips: inspect scanner “RR-skipped” output.

## Legacy (Do Not Touch by Default)
Examples: `base.py`, `simplified_trading_bot*.py`, older HFT/mean-reversion tools, `retired_tools/`.
