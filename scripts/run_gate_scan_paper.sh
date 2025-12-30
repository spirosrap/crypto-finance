#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/spiros/anaconda3/envs/trade/bin/python}
RUN_LIVE=${RUN_LIVE:-0}
RUN_PAPER=${RUN_PAPER:-1}

cd "${REPO_ROOT}"

if [[ -f ".env" ]]; then
  set -a
  . ./.env
  set +a
fi

LOCKFILE="${REPO_ROOT}/logs/gate_scan_paper.lock"
mkdir -p "$(dirname "${LOCKFILE}")"
exec 9>"${LOCKFILE}"
if ! flock -n 9; then
  echo "gate_scan_paper: lock held, skipping."
  exit 0
fi

SCAN_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/symbol_snapshot.py"
  --gate-scan
  --profile focused_no_llm_100
  --top 15
  --scan-limit 100
  --baseline-commands
  --baseline-portfolio-usd 5000
  --baseline-position-pct 5
  --baseline-atr-mult 0.8
  --baseline-rr 1.5
  --baseline-atr-mode clipped
  --baseline-leverage 50
  --baseline-expiry 30d
  --baseline-live-position-usd 125
  --baseline-paper-command
)

OUTPUT=$("${SCAN_CMD[@]}" 2>&1)
printf "%s\n" "${OUTPUT}"

if [[ "${RUN_PAPER}" == "1" ]]; then
  paper_cmd=$(printf "%s\n" "${OUTPUT}" | awk '/^python scripts\/baseline_finder_from_snapshot.py /{print; exit}')
  if [[ -n "${paper_cmd}" ]]; then
    paper_cmd=${paper_cmd/#python /${PYTHON_BIN} }
    eval "${paper_cmd}"
  fi
fi

if [[ "${RUN_LIVE}" == "1" ]]; then
  while IFS= read -r line; do
    if [[ "${line}" == python\ ccxt_trade_perp.py* ]]; then
      line=${line/#python /${PYTHON_BIN} }
      eval "${line}"
    fi
  done <<< "${OUTPUT}"
fi
