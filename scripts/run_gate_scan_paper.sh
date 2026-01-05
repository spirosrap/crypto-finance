#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/spiros/anaconda3/envs/trade/bin/python}
RUN_LIVE=${RUN_LIVE:-0}
RUN_PAPER=${RUN_PAPER:-1}
RUN_PAPER_UPDATE=${RUN_PAPER_UPDATE:-1}
LIVE_DELAY_SECONDS=${LIVE_DELAY_SECONDS:-10}
LIVE_SLEEP_SECONDS=${LIVE_SLEEP_SECONDS:-2}

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

if [[ "${RUN_PAPER}" == "1" && "${RUN_PAPER_UPDATE}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/paper_finder_simulator.py" update
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
  --baseline-live-position-usd 250
  --baseline-paper-command
)

OUTPUT=$("${SCAN_CMD[@]}" 2>&1)
printf "%s\n" "${OUTPUT}"

daily_stop_live_active=0
daily_stop_paper_active=0
range_break_active=0
live_pause_active=0

if printf "%s\n" "${OUTPUT}" | grep -qE "Daily stop \\(live\\): Daily stop \\(ACTIVE\\)"; then
  daily_stop_live_active=1
fi
if printf "%s\n" "${OUTPUT}" | grep -qE "Daily stop \\(paper\\): Daily stop \\(ACTIVE\\)"; then
  daily_stop_paper_active=1
fi
if printf "%s\n" "${OUTPUT}" | grep -qE "Range break ACTIVE"; then
  range_break_active=1
fi
if printf "%s\n" "${OUTPUT}" | grep -qE "Live pause \\(stop streak\\): ACTIVE"; then
  live_pause_active=1
fi

if [[ "${range_break_active}" == "1" ]]; then
  echo "Guard active: range break triggered. Closing positions and suppressing new entries."
  if [[ "${RUN_LIVE}" == "1" ]]; then
    echo "Closing live positions due to range break."
    if ! "${PYTHON_BIN}" "${REPO_ROOT}/close_positions.py"; then
      echo "Live close_positions.py failed (continuing)."
    fi
  fi
  if [[ "${RUN_PAPER}" == "1" ]]; then
    echo "Closing paper trades due to range break."
    if ! "${PYTHON_BIN}" "${REPO_ROOT}/paper_finder_simulator.py" close --all --reason "range_break"; then
      echo "Paper close failed (continuing)."
    fi
  fi
  exit 0
fi

if [[ "${RUN_LIVE}" == "1" && "${daily_stop_live_active}" == "1" ]]; then
  echo "Daily stop active (live): closing live positions and suppressing live entries."
  if ! "${PYTHON_BIN}" "${REPO_ROOT}/close_positions.py"; then
    echo "Live close_positions.py failed (continuing)."
  fi
fi
if [[ "${RUN_PAPER}" == "1" && "${daily_stop_paper_active}" == "1" ]]; then
  echo "Daily stop active (paper): closing paper positions and suppressing paper entries."
  if ! "${PYTHON_BIN}" "${REPO_ROOT}/paper_finder_simulator.py" close --all --reason "daily_stop"; then
    echo "Paper close failed (continuing)."
  fi
fi

if [[ "${RUN_PAPER}" == "1" && "${daily_stop_paper_active}" != "1" ]]; then
  paper_cmd=$(printf "%s\n" "${OUTPUT}" | awk '/^python scripts\/baseline_finder_from_snapshot.py /{print; exit}')
  if [[ -n "${paper_cmd}" ]]; then
    paper_cmd=${paper_cmd/#python /${PYTHON_BIN} }
    eval "${paper_cmd}"
  fi
fi

if [[ "${RUN_LIVE}" == "1" && "${daily_stop_live_active}" != "1" && "${live_pause_active}" != "1" ]]; then
  sleep "${LIVE_DELAY_SECONDS}"
  while IFS= read -r line; do
    if [[ "${line}" == python\ ccxt_trade_perp.py* ]]; then
      line=${line/#python /SKIP_LOAD_MARKETS=1 ${PYTHON_BIN} }
      echo "Executing live command: ${line}"
      if ! eval "${line}"; then
        echo "Live command failed (continuing): ${line}"
      fi
      sleep "${LIVE_SLEEP_SECONDS}"
    fi
  done <<< "${OUTPUT}"
elif [[ "${RUN_LIVE}" == "1" && "${live_pause_active}" == "1" ]]; then
  echo "Live pause active (stop streak): skipping live commands."
fi
