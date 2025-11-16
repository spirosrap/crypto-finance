# Finder Backtesting Workflow

This doc explains how to capture daily `short_term_crypto_finder.py` output and replay it later with `paper_finder_backtest.py`. The goal is to mirror the “top-5 balanced or skip the day” workflow while being able to answer performance questions without waiting weeks for live paper trading results.

## 1. Capture Finder Output Daily

Run the provided helper (`scripts/archive_finder_output.sh`) once per day (cron/systemd/task scheduler):

```bash
/home/spiros/crypto-finance/scripts/archive_finder_output.sh \
  >> /home/spiros/crypto-finance/logs/finder_archive.log 2>&1
```

The script runs:

```
python short_term_crypto_finder.py \
  --profile focused_no_llm_100 \
  --plain-output finder_logs/YYYY-MM-DD.txt \
  --force-refresh \
  --suppress-console-logs
```

Each invocation writes exactly one plain-text finder report to `finder_logs/`. The `--force-refresh` flag matches the manual workflow (always pull fresh candles), and the `--suppress-console-logs` flag keeps cron output clean. If the finder cannot reach Coinbase/other APIs, the script exits non‑zero; check `logs/finder_archive.log` to diagnose network or credential issues.

### Cron Setup

1. Ensure the helper is executable:

   ```bash
   chmod +x /home/spiros/crypto-finance/scripts/archive_finder_output.sh
   ```

2. Edit your user crontab:

   ```bash
   crontab -e
   ```

3. Add a daily entry (example: 12:15 UTC) and save:

   ```
   15 12 * * * /home/spiros/crypto-finance/scripts/archive_finder_output.sh >> /home/spiros/crypto-finance/logs/finder_archive.log 2>&1
   ```

Adjust the minute/hour to suit your preferred run window. The only requirement is to run it consistently so each UTC day has exactly one entry in `finder_logs/`.

## 2. Replay History With `paper_finder_backtest.py`

Once you have a few days of archived files, you can re-run the workflow offline:

```bash
python paper_finder_backtest.py \
  --finder-path finder_logs/*.txt \
  --days 30 \
  --balanced-top \
  --min-trades 5 \
  --expiry-hours 24 \
  --portfolio-usd 25000 \
  --output-csv backtest_results/finder_backtest.csv
```

Key behaviours:

- Uses the same candidate parsing and selection logic as `paper_finder_simulator.py`.
- Defaults to the “balanced top-5” approach (`--balanced-top` true) and skips any day where fewer than `--min-trades` candidates survive.
- Pulls historical INTX perp OHLCV via CCXT (so your environment must have network access and valid Coinbase API credentials if required).
- Simulates each trade until TP, SL, or expiry and records MAE/MFE plus P/L in the CSV.

After the run, the script prints a summary (total trades, win rate, drawdown) and writes per-trade rows when `--output-csv` is provided.

## 3. Compare to Live Paper Trading

This workflow complements the existing `paper_finder_simulator.py`:

- `archive_finder_output.sh` preserves the raw signals each day.
- `paper_finder_simulator.py open --top 5 --balanced-top` logs the paper trades in `trade_logs/paper_finder_*.csv`.
- `paper_finder_backtest.py` lets you replay any archived period instantly if you want to evaluate variations (different expiry, fewer trades, etc.) without waiting for more live paper trading data.

## Troubleshooting Checklist

1. **Finder archive fails (cron log shows CCXT errors):** verify internet access and that `short_term_crypto_finder.py` runs manually. Issues like DNS failures or missing credentials will bubble up here.
2. **Backtester exits with “Finder file … missing 'Generated on'”:** ensure the logs you point to come directly from the finder (plain text format) and haven’t been edited.
3. **Backtester hits CCXT NetworkError:** rerun later or check Coinbase availability; the script retries transient failures but still requires outbound network access.
4. **Days skipped with fewer than 5 trades:** this mirrors the desk rule; if you want to include smaller batches, lower `--min-trades`.

With these steps in place, you’ll have a rolling archive of authentic finder signals plus a way to replay any date range at will. Keep the `finder_logs/` directory under version control only if you need a sample; typically you’d gitignore it and store logs locally.
