#!/usr/bin/env bash
# Run the short-term finder on the focused_no_llm_100 profile, write a clean
# plain-text report, and send an email alert when at least N opportunities are
# present. Designed for cron (default every 45 minutes).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/spiros/anaconda3/envs/trade/bin/python}"
ALERT_EMAIL="${FINDER_ALERT_EMAIL:-}"  # set in the crontab entry
OUT_FILE="${FINDER_ALERT_OUT_FILE:-$REPO_ROOT/finder_short.txt}"
LOG_FILE="${FINDER_ALERT_LOG_FILE:-$REPO_ROOT/logs/finder_alert.log}"
MIN_OPPS="${FINDER_ALERT_MIN_OPPS:-5}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
    echo "No python interpreter found. Set PYTHON_BIN before running." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT_FILE")" "$(dirname "$LOG_FILE")"

run_stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[$run_stamp] Running focused_no_llm_100 finder..." >> "$LOG_FILE"

if ! "$PYTHON_BIN" short_term_crypto_finder.py \
    --profile focused_no_llm_100 \
    --plain-output "$OUT_FILE" \
    --force-refresh \
    --suppress-console-logs
then
    echo "[$run_stamp] Finder run failed" >> "$LOG_FILE"
    exit 1
fi

opps_line=$(grep -E '^Total opportunities listed:' "$OUT_FILE" || true)
opps_count=$(echo "$opps_line" | awk '{print $4+0}' 2>/dev/null || echo 0)

if [[ -z "$ALERT_EMAIL" ]]; then
    echo "[$run_stamp] ALERT_EMAIL not set; skipping email. Opportunities: $opps_count" >> "$LOG_FILE"
    exit 0
fi

if (( opps_count >= MIN_OPPS )); then
    subject="Finder: $opps_count opportunities (focused_no_llm_100)"
    if command -v mail >/dev/null 2>&1; then
        mail -s "$subject" "$ALERT_EMAIL" < "$OUT_FILE"
        echo "[$run_stamp] Sent alert to $ALERT_EMAIL (opps=$opps_count)" >> "$LOG_FILE"
    else
        echo "[$run_stamp] mail command not found; cannot send email (opps=$opps_count)" >> "$LOG_FILE"
    fi
else
    echo "[$run_stamp] Opportunities below threshold (opps=$opps_count, min=$MIN_OPPS); no email." >> "$LOG_FILE"
fi

