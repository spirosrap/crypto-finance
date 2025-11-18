#!/usr/bin/env bash
#
# Trigger the short-term finder with the focused_no_llm_100 profile and
# store its plain-text output in finder_logs/YYYY-MM-DD.txt.
#
# Usage (cron):
#   00 14 * * * /home/spiros/crypto-finance/scripts/archive_finder_output.sh >> /home/spiros/crypto-finance/logs/finder_archive.log 2>&1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/spiros/anaconda3/envs/trade/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
    echo "No python interpreter found. Set PYTHON_BIN before running." >&2
    exit 1
fi

STAMP="$(date -u +%Y-%m-%d)"
OUT_DIR="$REPO_ROOT/finder_logs"
OUT_FILE="$OUT_DIR/${STAMP}.txt"
OK_FILE="$OUT_DIR/${STAMP}.ok"

MAX_RETRIES="${FINDER_ARCHIVE_MAX_RETRIES:-3}"
RETRY_DELAY="${FINDER_ARCHIVE_RETRY_DELAY:-120}"

mkdir -p "$OUT_DIR"

attempt=1
success=0
rm -f "$OK_FILE"

while [[ $attempt -le $MAX_RETRIES ]]; do
    echo "[${STAMP}T$(date -u +%H:%M:%S)Z] Finder run attempt ${attempt}/${MAX_RETRIES}..."
    if "$PYTHON_BIN" short_term_crypto_finder.py \
        --profile focused_no_llm_100 \
        --plain-output "$OUT_FILE" \
        --force-refresh \
        --suppress-console-logs
    then
        success=1
        break
    fi
    echo "[${STAMP}T$(date -u +%H:%M:%S)Z] Finder run failed (attempt ${attempt})."
    if [[ $attempt -lt $MAX_RETRIES ]]; then
        echo "Sleeping ${RETRY_DELAY}s before retry..."
        sleep "$RETRY_DELAY"
    fi
    attempt=$((attempt + 1))
done

if [[ $success -eq 1 ]]; then
    echo "[${STAMP}T$(date -u +%H:%M:%S)Z] Finder output archived to $OUT_FILE"
    printf "ok %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OK_FILE"
else
    echo "[${STAMP}T$(date -u +%H:%M:%S)Z] Finder archiving FAILED after ${MAX_RETRIES} attempts." >&2
    rm -f "$OUT_FILE"
    exit 1
fi
