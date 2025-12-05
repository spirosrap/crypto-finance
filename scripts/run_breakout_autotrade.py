#!/usr/bin/env python
"""
One-shot breakout auto-runner for BTC.

- Scans BTC/USDC for breakouts (via breakout_scanner) on the chosen timeframe.
- Picks the top candidate with RR >= 2.
- Writes finder-format levels to an output file.
- Optionally invokes add_position_from_finder.py to place an order.
- Enforces a 24h lock: if a trade was taken, subsequent runs exit until the lock expires.

Default behavior is dry-run (no execution). Pass --execute to place the order.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOCK_PATH = Path(".breakout_lock.json")
DEFAULT_OUT = Path("finder_breakout.txt")


def lock_active() -> bool:
    if not LOCK_PATH.exists():
        return False
    try:
        data = json.loads(LOCK_PATH.read_text())
        exp = data.get("expires_at")
        if not exp:
            return False
        exp_dt = datetime.fromisoformat(exp)
        return exp_dt > datetime.now(timezone.utc)
    except Exception:
        return False


def set_lock(hours: int = 24) -> None:
    exp = datetime.now(timezone.utc) + timedelta(hours=hours)
    LOCK_PATH.write_text(json.dumps({"expires_at": exp.isoformat()}))


def run_scanner(out_path: Path, timeframe: str, lookback: int) -> int:
    cmd = [
        sys.executable,
        "scripts/breakout_scanner.py",
        "--symbols",
        "BTC",
        "--timeframe",
        timeframe,
        "--lookback",
        str(lookback),
        "--out",
        str(out_path),
    ]
    return subprocess.call(cmd)


def has_candidate(out_path: Path, rr_threshold: float = 2.0) -> bool:
    if not out_path.exists():
        return False
    text = out_path.read_text()
    if "TRADING LEVELS" not in text:
        return False
    # crude RR extraction
    for line in text.splitlines():
        if line.startswith("RR="):
            try:
                val = float(line.split()[0].split("=")[1])
                if val >= rr_threshold:
                    return True
            except Exception:
                continue
    return False


def place_order(out_path: Path, portfolio_usd: float, leverage: float, execute: bool) -> int:
    cmd = [
        sys.executable,
        "add_position_from_finder.py",
        "--file",
        str(out_path),
        "--portfolio-usd",
        str(portfolio_usd),
        "--leverage",
        str(leverage),
        "--order",
        "market",
    ]
    if execute:
        cmd.append("--execute")
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BTC breakout scan and optionally place a trade.")
    parser.add_argument("--timeframe", default="1h", help="Scan timeframe (default: 1h; will fallback in scanner if unsupported)")
    parser.add_argument("--lookback", type=int, default=50, help="Swing lookback (default: 50)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output file path (finder format)")
    parser.add_argument("--portfolio-usd", type=float, default=500.0, help="Notional per trade (default: $500)")
    parser.add_argument("--leverage", type=float, default=50.0, help="Leverage (default: 50x)")
    parser.add_argument("--execute", action="store_true", help="Actually place the order via add_position_from_finder.py")
    parser.add_argument("--rr-threshold", type=float, default=2.0, help="Minimum RR to accept (default: 2.0)")
    args = parser.parse_args()

    out_path = Path(args.out)

    if lock_active():
        print("Active lock present (trade in flight). Exiting.")
        sys.exit(0)

    rc = run_scanner(out_path, args.timeframe, args.lookback)
    if rc != 0:
        print(f"Scanner failed with code {rc}")
        sys.exit(rc)

    if not has_candidate(out_path, args.rr_threshold):
        print("No qualifying breakout found (RR below threshold or no signals).")
        sys.exit(0)

    rc = place_order(out_path, args.portfolio_usd, args.leverage, args.execute)
    if rc != 0:
        print(f"Order placement exited with code {rc}")
        sys.exit(rc)

    # Set 24h lock after placing an order (even dry-run to avoid spamming)
    set_lock(24)
    print("Lock set for 24h. Clear .breakout_lock.json manually if needed.")


if __name__ == "__main__":
    main()
