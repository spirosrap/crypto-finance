# Cron Jobs Snapshot
# Install on a new machine with:
#   crontab docs/pipelines/cron.md
# Notes:
# - Update /home/spiros/crypto-finance if the repo lives elsewhere.
# - Update /home/spiros/anaconda3/envs/trade/bin/python if the env path differs.
# - RUN_LIVE/RUN_PAPER toggle whether live/paper trades are executed.

15 12 * * * PYTHON_BIN=/home/spiros/anaconda3/envs/trade/bin/python /home/spiros/crypto-finance/scripts/archive_finder_output.sh >> /home/spiros/crypto-finance/logs/finder_archive.log 2>&1
20 12 * * * FINDER_ARCHIVE_ALERT_EMAIL=spirosrap@gmail.com /home/spiros/crypto-finance/scripts/check_finder_archive.sh >> /home/spiros/crypto-finance/logs/finder_archive.log 2>&1
*/45 * * * * cd /home/spiros/crypto-finance && { set -a; . ./.env; set +a; PYTHON_BIN=/home/spiros/anaconda3/envs/trade/bin/python scripts/finder_alert.sh; } >> /home/spiros/crypto-finance/logs/finder_alert.log 2>&1
# 0 * * * * cd /home/spiros/crypto-finance && { set -a; . ./.env; set +a; /home/spiros/anaconda3/envs/trade/bin/python scripts/run_breakout_autotrade.py --timeframe 1h --lookback 50 --portfolio-usd 500 --leverage 50 --out finder_breakout.txt --execute; } >> /home/spiros/crypto-finance/logs/breakout_autotrade.log 2>&1
5 */4 * * * cd /home/spiros/crypto-finance && PYTHON_BIN=/home/spiros/anaconda3/envs/trade/bin/python RUN_LIVE=1 /bin/bash -lc "scripts/run_gate_scan_paper.sh" >> /home/spiros/crypto-finance/logs/gate_scan_paper.log 2>&1
# Gate-scan runner streams output live and uses the 400-product balanced scan defaults in the script.
# It now applies performance filtering + side score gates and adds TP1 partials for paper/live.
# Live commands use a small marketable limit offset (BASELINE_LIMIT_BPS) to keep CCXT fills and TP1 brackets.

# paper_finder_update
*/5 * * * * cd /home/spiros/crypto-finance && /home/spiros/anaconda3/envs/trade/bin/python /home/spiros/crypto-finance/paper_finder_simulator.py update >> /home/spiros/crypto-finance/logs/paper_finder_update.log 2>&1

# watchdog_close_update
*/3 * * * * cd /home/spiros/crypto-finance && /home/spiros/anaconda3/envs/trade/bin/python /home/spiros/crypto-finance/watchdog_close_old_positions.py --log-fills --verbose --dust-notional-usd 5 --move-sl-after-tp1 >> /home/spiros/crypto-finance/logs/watchdog_close_update.log 2>&1

# live_snapshot_update
*/5 * * * * cd /home/spiros/crypto-finance && /home/spiros/anaconda3/envs/trade/bin/python /home/spiros/crypto-finance/scripts/update_live_snapshot.py >> /home/spiros/crypto-finance/logs/live_snapshot_update.log 2>&1

# daily stop guard
*/5 * * * * cd /home/spiros/crypto-finance && RUN_LIVE=1 RUN_PAPER=1 /home/spiros/anaconda3/envs/trade/bin/python /home/spiros/crypto-finance/scripts/daily_stop_guard.py >> /home/spiros/crypto-finance/logs/daily_stop_guard.log 2>&1
