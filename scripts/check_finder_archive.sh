#!/usr/bin/env bash
#
# Verify that today's finder archive succeeded. If the success marker is
# missing, send an email (requires $FINDER_ARCHIVE_ALERT_EMAIL) and exit 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y-%m-%d)"
OK_FILE="${REPO_ROOT}/finder_logs/${STAMP}.ok"
ALERT_EMAIL="${FINDER_ARCHIVE_ALERT_EMAIL:-}"

if [[ -f "$OK_FILE" ]]; then
    exit 0
fi

msg="Finder archive missing or failed for ${STAMP}. No ${OK_FILE} marker was created."
echo "$msg" >&2

if [[ -n "$ALERT_EMAIL" ]] && command -v mail >/dev/null 2>&1; then
    printf "%s\n" "$msg" | mail -s "[Finder] Archive missing for ${STAMP}" "$ALERT_EMAIL"
else
    echo "Skipping email notification; set FINDER_ARCHIVE_ALERT_EMAIL and ensure 'mail' is installed." >&2
fi

exit 1
