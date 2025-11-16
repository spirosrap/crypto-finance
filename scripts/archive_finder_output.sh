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

STAMP="$(date -u +%Y-%m-%d)"
OUT_DIR="$REPO_ROOT/finder_logs"
OUT_FILE="$OUT_DIR/${STAMP}.txt"

mkdir -p "$OUT_DIR"

echo "[${STAMP}T$(date -u +%H:%M:%S)Z] Running short_term_crypto_finder..."
python short_term_crypto_finder.py \
    --profile focused_no_llm_100 \
    --plain-output "$OUT_FILE" \
    --force-refresh \
    --suppress-console-logs

echo "[${STAMP}T$(date -u +%H:%M:%S)Z] Finder output archived to $OUT_FILE"
