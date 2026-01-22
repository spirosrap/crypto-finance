# VPS Pipeline Setup Plan

## Goal
Run the crypto-finance pipelines on a VPS using cron (or systemd timers), with safe key handling, reproducible setup, and reliable logs/health checks.

## Assumptions
- VPS runs Ubuntu 22.04/24.04.
- Repo is deployed under a non-root user.
- Secrets live in `.env` and never in git.
- Pipelines should default to paper trading unless explicitly enabled for live.

## Inputs Needed
- VPS provider + region, hostname/IP, SSH access method.
- Target pipelines (short-term finder, breakout autotrade, paper flow, watchdogs).
- Schedule (times + timezone).
- API keys placed in `.env` on the VPS.

## Step-by-Step Outline
1) Provision VPS
- Create a droplet/instance with Ubuntu LTS.
- Attach SSH key (no password auth).
- Enable provider firewall (allow SSH + any web ports if needed).

2) Base Hardening
- Create a non-root user and add to sudo group.
- Configure UFW and fail2ban.
- Disable root SSH login and password auth.
- Set timezone for cron predictability.
- Install ZeroTier for private access from the personal network.
- Use ZeroTier to access the VPS easily from other machines on the network.

3) System Dependencies
- Install `python3.11`, `python3.11-venv`, `git`, build tools.
- Install TA-Lib system libs if needed by `pandas-ta` / TA-Lib bindings.

4) Deploy Repo
- Clone repo into `/home/<user>/crypto-finance`.
- Create `.venv` and install `requirements.txt`.
- Verify import sanity (quick `python -c` import check).

5) Secrets + Config
- Create `.env` with required keys (`API_KEY`, `API_SECRET`, etc.).
- Normalize PEM secrets (escaped newlines).
- `chmod 600 .env` and restrict directory access.

6) Data + Logs
- Ensure log/data directories exist (`logs/`, `trade_logs/`, `backtest_results/`).
- Add log rotation (system `logrotate` or simple log pruning).

7) Cron Wiring
- Start from `docs/pipelines/cron.md`.
- Set `PYTHON_BIN`, `RUN_PAPER`, `RUN_LIVE` as needed.
- Add cron entries for:
  - Gate scan + finder.
  - Paper simulator updates.
  - Breakout autotrade (hourly).
  - Watchdog tasks (close old positions, daily stop guard, dashboard).

8) Validation
- Run one manual cycle:
  - `python scripts/symbol_snapshot.py --gate-scan ...`
  - `python short_term_crypto_finder.py` or `python scripts/run_breakout_autotrade.py ...`
- Confirm logs appear and outputs update.

9) Health Checks
- Verify log heartbeat updates in watchdog scripts.
- Add a simple daily log check or dashboard run.
- Optional: push alerts to email/Slack if logs are stale.

10) Handoff + Ops
- Document schedules, env vars, and log locations.
- Confirm paper vs live toggles and risk thresholds.
- Plan a monthly OS/package update window.

## Validation Plan
- `python -m unittest discover -s tests -v` (optional, if runtime allows).
- Run a single pipeline pass; check `logs/` and `finder_*.txt` outputs.
- Ensure `.env` permissions and no secrets are in git.

## Risks / Watchouts
- TA-Lib install issues (system package required).
- Cron environment differences (PATH, locale, timezone).
- Stale logs if cron is misconfigured or permissions are wrong.

## Next Actions
- Pick VPS provider/region and schedule.
- Provide SSH access details and confirm which pipelines to enable.
- Place `.env` on the VPS (avoid sharing secrets in chat).
