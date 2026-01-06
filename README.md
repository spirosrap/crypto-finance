# Crypto Finance — Current Pipeline (Dec 2025)

## About This Repository
This repo is my active, rules‑based crypto trading workflow. It focuses on short‑term signal generation, disciplined execution, and transparency through logs and diagnostics. For me, it’s a living system that enforces patience (RR/ATR gates) and reduces emotional trading by making the process explicit.

For others, this repo can be useful as:
- A reference implementation of **systematic crypto trading** with clear gates, snapshots, and audit trails.
- A practical example of **signal → output → execution** plumbing using CCXT + Coinbase.
- A toolbox of **diagnostics** (ATR clip analysis, gate scans, volatility regime readouts) that help validate decision rules.

## Current Focus
- **Short-Term Crypto Finder (`short_term_crypto_finder.py`)**: Strict RR ≥ 2, ATR(7) gated with tiered caps (3k USD plus bps tiers ≈325/350/400/450 by price bands; tighter cap wins), hard SL/TP. Runs on USDC pairs; logs to `logs/short_term_crypto_finder/`.
- **Breakout Autotrade (`scripts/run_breakout_autotrade.py`)**: Hourly scan of USDC majors (BTC/ETH/SOL/XRP/ADA/DOT/AVAX/LINK/LTC/DOGE/OP/ARB/ATOM/UNI/AAVE/MKR/INJ) with fixed 2R structure, $500 notional / 50x, 24h lock to avoid stacking. Writes finder-format output; uses Coinbase primary with Kraken fallback. Near-breakouts are logged. Currently paused/archived (cron commented).
- **Safety First**: Downtime is expected; no trades when RR/ATR gates fail. Flat is acceptable.

## Active Tools
- **Finders**: `short_term_crypto_finder.py`, `long_term_crypto_finder.py`.
  - Short-horizon snapshot: `scripts/symbol_snapshot.py` (includes short-term gates + intraday context).
    - Includes ATR-multiple readout for SL/TP distance and an `RR drivers` line showing which caps/floors dominated.
    - Renders ASCII Rich tables when `rich` is installed; falls back to plain text.
  - Long-horizon snapshot: `scripts/long_term_snapshot.py` (daily candles + long-term indicators like ATR(14)/Sharpe/drawdown).
    - Renders ASCII Rich tables when `rich` is installed; falls back to plain text.
    - Gate scan (long-term): `python scripts/long_term_snapshot.py --gate-scan --profile wide --top 15 --scan-limit 200`
  - Gate proximity view (short-term): `python scripts/symbol_snapshot.py --gate-scan [--scan-limit N]`.
    - Use `--baseline-commands` to print ccxt trade commands for baseline-pass symbols; open live positions are auto-skipped (and listed).
    - Use `--baseline-paper-command` to emit a one-shot `baseline_finder_from_snapshot.py` command for paper trades (skips open live/paper positions unless `--baseline-include-open` is set).
    - Capacity guards: `--baseline-max-open` (default 10) and `--baseline-max-per-cluster` (default 3) suppress commands once the live/paper books are full. Clusters are majors/memecoins/alts.
    - Daily stop + range break: gate-scan suppresses commands if the daily loss gate triggers (default −4%/−$40) or if BTC closes outside the 7‑day range by >0.5× ATR (range-break circuit). Range-break is **latched** until a confirmed daily close returns inside range±buffer (intraday moves do not clear). Trigger mode is configurable via `range_break_confirmed_only` in `config/risk_thresholds.yaml`.
    - Shared defaults live in `config/risk_thresholds.yaml` (used by gate‑scan and the dashboard); CLI flags override per run.
    - Shared risk defaults live in `config/risk_thresholds.yaml` (used by gate-scan + dashboard). CLI flags override per run.
  - Long-term profile used for LLM + liquidity filters: `python long_term_crypto_finder.py --profile focused_llm_100 --plain-output finder_long.txt --suppress-console-logs`
- **Breakout Suite**: `scripts/breakout_scanner.py` (finder-style output, near-breakout logs), `scripts/run_breakout_autotrade.py` (cron-friendly runner with lock; currently paused).
- **Watchdogs/Closers**: `watchdog_close_old_positions.py` (optional 24h timeout), `watchdog_dashboard.py` for monitoring (Streamlit).
- **Support**: `add_position_from_finder.py` to stage/execute trades from finder-format text.
- **Baseline Paper Helper**: `scripts/baseline_finder_from_snapshot.py` to turn snapshot picks into `finder_short_baseline.txt` and optionally open them in `paper_finder_simulator.py`.
  - After opening paper trades, run `python paper_finder_simulator.py update` so the dashboard shows live P/L.
- **Gate-scan Paper Runner**: `scripts/run_gate_scan_paper.sh` automates the gate-scan + paper-trade flow (cron-friendly). Set `PYTHON_BIN`, `RUN_PAPER=0|1`, and `RUN_LIVE=0|1`; logs to `logs/gate_scan_paper.log`.
  - Scheduled every 4 hours to batch entries and reduce impulse/overtrading; it only opens new trades when gates pass.
  - If the daily stop or BTC range-break guard triggers, it auto-closes live positions (`close_positions.py`) and paper positions (`paper_finder_simulator.py close --all`) and suppresses new entries. Range-break stays latched until a confirmed daily close is back inside range±buffer; it can be set to trigger only on confirmed closes via `range_break_confirmed_only`.
- **Paper Trade Updater (cron)**: `paper_finder_simulator.py update` runs every 5 minutes to keep paper P/L and expiries fresh; logs to `logs/paper_finder_update.log`.
- **Watchdog Fill Poll (cron)**: `watchdog_close_old_positions.py --log-fills --skip-close --verbose` runs every 5 minutes to pull the latest fills from the Coinbase account and refresh the closed-trade log; logs to `logs/watchdog_close_update.log`.
- **Live Snapshot Updater (cron)**: `scripts/update_live_snapshot.py` runs every 5 minutes to refresh live positions + USDC balance and write `logs/live_snapshot.json` for the dashboard; logs to `logs/live_snapshot_update.log`.
- **Cron snapshot**: `docs/cron.md` captures the current crontab so the pipeline can be restored quickly on a new machine (`crontab docs/cron.md`, then adjust paths/envs).
- **Paper Progress Check**: `scripts/paper_trade_progress.py` prints progress to the 100‑trade target, win%, avg%, expectancy, and TP/SL/expiry split.
- **Drawdown Breakdown**: `scripts/drawdown_breakdown.py` compares paper vs live over a rolling window (default 24h), highlighting closure reasons, loss symbols, ATR bucket stops, and spread‑OK stop rates.
- **Paper Equity Report**: `scripts/paper_equity_report.py` writes a shareable equity curve (HTML + PNG) and `watchdog_dashboard.py` can export the same. Note: equity is daily‑aggregated, so one trading day = one point (drawdown reads 0% when there’s only one day).
- **Risk Guards (automated)**: daily loss gate (default −4%/−$40 of $1k) and BTC range‑break circuit (7‑day range ±0.5× ATR) suppress new entries until the daily reset or a confirmed daily close re-enters the range. `range_break_confirmed_only` makes the trigger rely on confirmed closes only (less intraday flip‑flop).
  - Range-break status: **inside** = OK to trade; **outside** = pause (latched until confirmed close inside range±buffer).
- **Daily Stop Streak Guard**: if live daily stop hits 3 times in 7 days, live entries pause for 3 days; 5 hits in 14 days prompts a size‑reduction warning; 7 hits in 21 days prompts a tighten‑filters/paper‑only warning. Stored in `logs/daily_stop_history.json` and configured in `config/risk_thresholds.yaml`.
- **Current Paper Experiment (to 150 trades)**: baseline ATR exits (`atr_mult=0.8`, `rr=1.5`, `atr_mode=clipped`) with liquidity/spread gates and ATR ≤ 1.5× cap. Live rules stay unchanged until the 150‑trade sample is complete.
- **Checkpoint plan (live/paper)**:
  - Finish the 100‑trade live checkpoint as‑is (no tweaks).
  - Push to 150 with zero changes; keep the system if PF stays > 1.3 and max drawdown stays < 10%.
  - If weakness persists at 150, first adjustment is to reduce shorts or require a stronger short filter (change one knob only).
- **Metric glossary**:
  - `profit_loss` = USD P&L per trade (drives equity curve).
  - `profit_loss_pct` = % P&L per trade (trade quality).
  - Expectancy (USD) = average `profit_loss`.
  - Expectancy (%) = average `profit_loss_pct`.
  - Profit factor = total wins ÷ total losses (USD).
  - `win_rate_pct` = percent of trades with positive `profit_loss`.
  - `avg_pct` = average `profit_loss_pct` across trades.
  - `avg_win_pct` / `avg_loss_pct` = average % for winners/losers.
  - `median_profit_loss` / `median_profit_loss_pct` = median USD/% per trade (less skewed by outliers).
  - `max_drawdown` / `max_drawdown_pct` = worst peak‑to‑trough equity drop (USD/%).
  - `sharpe_ratio` = risk‑adjusted return of daily equity (higher is better, noisy on small samples).
  - `total_return_pct` = (ending equity − starting equity) ÷ starting equity.
- **Diagnostics**: `scripts/watchdog_atr_clip_analysis.py` to bucket closed trades by ATR cap ratio and compare outcomes.
- **Parquet + SQL**: `scripts/convert_to_parquet.py` to emit parquet copies of finder/trade logs/backtests (use `--skip-bad-lines` if a CSV has malformed rows), plus `scripts/duckdb_query.py` for ad-hoc SQL diagnostics. Parquet keeps analytics fast and lightweight on large logs. Low-level helpers live in `trading/parquet_utils.py`.

## Pipeline Map (Current)
- Inputs: Coinbase/CCXT candles + CoinGecko metrics + live spread/fee bps (when available).
- Short-term flow: `short_term_crypto_finder.py` -> `finder_short.txt` -> `add_position_from_finder.py`/`add_top5_from_finder.py` -> `trade_logs/` + watchdogs.
- Breakout flow: `scripts/run_breakout_autotrade.py` -> `finder_breakout.txt` -> (optional execute) -> `trade_logs/` (currently paused).
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
3) Breakout autotrade (cron) and log review (currently paused):
   - Check `logs/breakout_autotrade.log` for near-breakouts or triggers when re-enabled.
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
| `logs/gate_scan_paper.log` | gate-scan runner | Gate-scan + paper cron log | `logs/` | Output from `scripts/run_gate_scan_paper.sh`. |
| `logs/paper_finder_update.log` | paper updater | Paper trade update log | `logs/` | Output from cron `paper_finder_simulator.py update`. |
| `logs/watchdog_close_update.log` | fill poller | Fill update log | `logs/` | Output from cron `watchdog_close_old_positions.py --log-fills --skip-close`. |
| `trade_logs/watchdog_closed_positions.csv` | watchdog | Closed trade ledger | `trade_logs/` | Source for ATR clip analysis. |
| `trade_logs/watchdog_tp_sl_checkpoint.json` | watchdog | Open trade checkpoints | `trade_logs/` | Used for recovery. |

## Legacy / Not in the Current Pipeline
- Older tools, setup notes, and retired workflows are archived in `README_LEGACY.md`.
- The active pipeline is defined above (Current Focus + Runbook + Outputs/Logs).

## Environment & Requirements
- Python 3.11, ccxt ≥ 4.2, pandas ≥ 2.3, numpy ≥ 1.24, TA-Lib ≥ 0.6.7, pydantic ≥ 2.7, pydantic-settings ≥ 2.3, openai ≥ 1.109.1 (installed), full list in `requirements.txt`.
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
