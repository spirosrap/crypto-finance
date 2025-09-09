#!/usr/bin/env python3
"""
Trade Guardian — daily stop/target manager

Given one or more open trades (symbol/product, side, entry, SL/TP),
this tool fetches the latest market data from Coinbase, recalculates
key risk/structure metrics (ATR, recent swing levels), and recommends:
  - KEEP (no change)
  - RAISE_SL / LOWER_SL (tighten protection; long/short respectively)
  - EXTEND_TP (optional, if requested)
  - EXIT_HIT (if price has already hit SL/TP)

It reuses the indicator logic from long_term_crypto_finder for
consistency (ATR, volatility, etc.).

Usage examples:
  - Single trade (LONG):
      python trade_guardian.py --symbol BTC --side long --entry 60000 \
          --sl 54000 --tp 78000

  - Single trade (SHORT):
      python trade_guardian.py --symbol ETH --side short --entry 3500 \
          --sl 3800 --tp 2800

  - From JSON file (list of trades):
      python trade_guardian.py --file my_trades.json --output json

JSON schema example:
  [
    {"symbol":"BTC","side":"long","entry_price":60000,"stop_loss":54000,"take_profit":78000,
     "initial_stop":54000,"opened_at":"2025-09-01T00:00:00Z"}
  ]

Notes:
  - If both product_id and symbol are provided, product_id wins.
  - Default quotes preference: USDC,USD,USDT (overridable via --quotes).
  - To extend TP dynamically to maintain a target R:R (default 3:1),
    pass --extend-tp (optional, conservative by default).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from coinbaseservice import CoinbaseService
from historicaldata import HistoricalData
from long_term_crypto_finder import LongTermCryptoFinder, CryptoFinderConfig


# ------------- Logging -------------
def _setup_logger() -> logging.Logger:
    logs_dir = Path("logs") / "trade_guardian"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("trade_guardian")
    if logger.handlers:
        return logger
    level = os.getenv("TRADE_GUARDIAN_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    fh = logging.FileHandler(logs_dir / "trade_guardian.log", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    for h in (fh, ch):
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)
    logger.info("Trade Guardian logging initialized")
    return logger


logger = _setup_logger()


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    symbol: Optional[str] = None
    product_id: Optional[str] = None
    side: str = "LONG"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: Optional[float] = None
    initial_stop: Optional[float] = None  # optional: SL at entry time for 1R/2R logic
    opened_at: Optional[str] = None  # ISO8601
    quantity: Optional[float] = None

    def clean(self) -> None:
        if self.symbol:
            self.symbol = self.symbol.strip().upper()
        if self.product_id:
            self.product_id = self.product_id.strip().upper()
        s = str(self.side or "LONG").upper()
        self.side = Side.SHORT.value if s in ("SHORT", "S") else Side.LONG.value


@dataclass
class Recommendation:
    action: str
    new_stop_loss: float
    new_take_profit: Optional[float]
    current_price: float
    rationale: str
    status: str  # OK | EXIT_HIT | INSUFFICIENT_DATA | ERROR


def _finite(x: Union[float, int]) -> float:
    try:
        xv = float(x)
        if not np.isfinite(xv):
            return 0.0
        return xv
    except Exception:
        return 0.0


def _normalize_perp_product(sym: str) -> Optional[str]:
    """Normalize various user inputs to a Coinbase INTX perp product id.

    Accepts forms like:
      - "BTC-PERP-INTX" (already normalized)
      - "BTC-INTX-PERP" (reordered)
      - "BTC-PERP" (append INTX)
      - "BTC-INTX" (not perp) -> None
      - "POL-INTX-PERP" -> "POL-PERP-INTX"
    Returns normalized product id or None if it doesn't look like a perp.
    """
    s = sym.upper().strip()
    if "PERP" not in s:
        return None
    parts = [p for p in s.split("-") if p]
    if not parts:
        return None
    base = parts[0]
    has_intx = any(p == "INTX" for p in parts)
    # If already in BASE-PERP-INTX form
    if len(parts) >= 2 and parts[1] == "PERP" and ((len(parts) >= 3 and parts[2] == "INTX") or not has_intx):
        return f"{base}-PERP-INTX" if has_intx or len(parts) < 3 else s
    # If in BASE-INTX-PERP form or mixed order
    return f"{base}-PERP-INTX"


def _resolve_product_id(finder: LongTermCryptoFinder, trade: Trade, quotes: List[str]) -> Optional[str]:
    # Explicit product id wins
    if trade.product_id:
        return trade.product_id.strip().upper()
    if not trade.symbol:
        return None
    sym = trade.symbol.strip().upper()

    # If the symbol looks like a perp, normalize and use as-is (no quote suffix)
    norm_perp = _normalize_perp_product(sym)
    if norm_perp:
        return norm_perp

    # If the symbol looks like an explicit spot product (e.g., BTC-USDC), use as-is
    if "-" in sym and sym.split("-")[-1] in {"USD", "USDC", "USDT", "EUR", "GBP"}:
        return sym

    # Otherwise treat it as a base symbol and map via available spot products
    prod_map = finder._fetch_usdc_products()  # symbol -> {product_id, base_name, quote}
    rec = prod_map.get(sym)
    if rec and rec.get("product_id"):
        return rec["product_id"]
    # Fallback: build with preferred quotes (first one wins)
    for q in quotes:
        return f"{sym}-{q.upper()}"
    return None


def _latest_price_from_hourly(h: HistoricalData, product_id: str) -> Optional[Tuple[float, datetime]]:
    try:
        end = datetime.utcnow()
        start = end - timedelta(hours=24)
        candles = h.get_historical_data(product_id, start, end, "ONE_HOUR")
        if not candles:
            return None
        last = candles[-1]
        price = float(last.get("close"))
        ts = last.get("start") or last.get("time")
        if isinstance(ts, (int, float)):
            dt = datetime.utcfromtimestamp(int(ts))
        else:
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except Exception:
                dt = end
        return price, dt
    except Exception as e:
        logger.warning(f"Hourly price fetch failed for {product_id}: {e}")
        return None


def _swing_levels(df: pd.DataFrame, lookback: int = 10) -> Tuple[float, float]:
    try:
        if len(df) < max(lookback, 2):
            return 0.0, 0.0
        low = float(df["low"].tail(lookback).min())
        high = float(df["high"].tail(lookback).max())
        return low, high
    except Exception:
        return 0.0, 0.0


def _r_multiple(side: Side, entry: float, stop_basis: float, current: float) -> float:
    try:
        if side == Side.LONG:
            R = max(1e-9, entry - stop_basis)
            return (current - entry) / R
        else:
            R = max(1e-9, stop_basis - entry)
            return (entry - current) / R
    except Exception:
        return 0.0


def _recommend_for_trade(finder: LongTermCryptoFinder, trade: Trade, analysis_days: int, extend_tp: bool,
                         rr_target: float, quotes: List[str]) -> Recommendation:
    try:
        trade.clean()
        pid = _resolve_product_id(finder, trade, quotes)
        if not pid:
            return Recommendation("KEEP", trade.stop_loss, trade.take_profit, 0.0, "No product_id/symbol mapping", "INSUFFICIENT_DATA")

        rationale_bits: List[str] = []

        # Fetch data (prefer product's own bars; for PERP fallback to spot if unavailable)
        df = finder.get_historical_data(pid, days=analysis_days)
        bars_pid = pid
        if (df is None or len(df) < 30) and ("PERP" in pid):
            base = pid.split("-")[0]
            try:
                spot_map = finder._fetch_usdc_products()
                spid = (spot_map.get(base) or {}).get("product_id")
                if spid:
                    df = finder.get_historical_data(spid, days=analysis_days)
                    bars_pid = spid if df is not None and len(df) >= 30 else pid
                    if bars_pid != pid and df is not None and len(df) >= 30:
                        rationale_bits.append("Used spot bars for indicators (no PERP candles)")
            except Exception:
                pass
        if df is None or len(df) < 30:
            return Recommendation("KEEP", trade.stop_loss, trade.take_profit, 0.0, "Insufficient historical data", "INSUFFICIENT_DATA")

        # Indicators (ATR, etc.)
        tech = finder.calculate_technical_indicators(df)
        atr = float(tech.get("atr", 0.0) or 0.0)

        # Current price (hourly)
        # Current price: try product first, then bars source if differed
        lp = _latest_price_from_hourly(finder.historical_data, pid)
        if not lp:
            if bars_pid != pid:
                lp = _latest_price_from_hourly(finder.historical_data, bars_pid)
        if not lp:
            # fallback to last daily close (bars source)
            cur_price = float(df["price"].iloc[-1])
            cur_ts = df.index.max().to_pydatetime()
        else:
            cur_price, cur_ts = lp

        side = Side(trade.side)
        entry = _finite(trade.entry_price)
        sl_old = _finite(trade.stop_loss)
        tp_old = _finite(trade.take_profit) if trade.take_profit is not None else None
        init_stop = _finite(trade.initial_stop) if (trade.initial_stop is not None) else sl_old

        # Swing levels
        low10, high10 = _swing_levels(df, 10)
        low20, high20 = _swing_levels(df, 20)
        k_atr = float(os.getenv("TRADE_GUARD_ATR_MULT", "2.0"))
        min_gap = float(os.getenv("TRADE_GUARD_MIN_GAP_PCT", "0.001"))  # 0.1%
        min_step = float(os.getenv("TRADE_GUARD_MIN_STEP_PCT", "0.0025"))  # 0.25%

        action = "KEEP"
        # rationale_bits initialized earlier

        # 1) Check exits already hit
        if side == Side.LONG:
            if cur_price <= sl_old:
                return Recommendation("EXIT_HIT", sl_old, tp_old, cur_price, "Stop already hit", "EXIT_HIT")
            if tp_old and cur_price >= tp_old:
                return Recommendation("EXIT_HIT", sl_old, tp_old, cur_price, "Target already hit", "EXIT_HIT")
        else:
            if cur_price >= sl_old:
                return Recommendation("EXIT_HIT", sl_old, tp_old, cur_price, "Stop already hit", "EXIT_HIT")
            if tp_old and cur_price <= tp_old:
                return Recommendation("EXIT_HIT", sl_old, tp_old, cur_price, "Target already hit", "EXIT_HIT")

        # 2) Trailing stop candidates
        if side == Side.LONG:
            c_atr = cur_price - k_atr * atr if atr > 0 else 0.0
            c_sup = max(low10 * 0.98 if low10 else 0.0, low20 * 0.98 if low20 else 0.0)
            # 1R/2R/3R steps relative to initial stop
            Rmult = _r_multiple(side, entry, init_stop, cur_price)
            c_be = 0.0
            if Rmult >= 3.0:
                c_be = entry + (entry - init_stop) * 1.0  # lock 1R
            elif Rmult >= 2.0:
                c_be = entry + (entry - init_stop) * 0.5
            elif Rmult >= 1.0:
                c_be = entry
            candidates = [x for x in [c_atr, c_sup, c_be] if x and x > 0]
            if candidates:
                proposed = max(candidates)
                proposed = min(proposed, cur_price * (1 - min_gap))  # keep below price
                new_sl = max(sl_old, proposed)
                if new_sl > sl_old * (1 + min_step):
                    action = "RAISE_SL"
                    rationale_bits.append("ATR/support/1R progression suggests higher stop")
                else:
                    action = "KEEP"
            else:
                new_sl = sl_old
        else:  # SHORT
            c_atr = cur_price + k_atr * atr if atr > 0 else 0.0
            c_res = min(high10 * 1.02 if high10 else float("inf"), high20 * 1.02 if high20 else float("inf"))
            Rmult = _r_multiple(side, entry, init_stop, cur_price)
            c_be = float("inf")
            if Rmult >= 3.0:
                c_be = entry - (init_stop - entry) * 1.0  # lock 1R (still above price? clamp later)
            elif Rmult >= 2.0:
                c_be = entry - (init_stop - entry) * 0.5
            elif Rmult >= 1.0:
                c_be = entry
            candidates = [x for x in [c_atr, c_res, c_be] if np.isfinite(x) and x > 0]
            if candidates:
                proposed = min(candidates)
                # keep above price and not increase vs current stop
                proposed = max(proposed, cur_price * (1 + min_gap))
                new_sl = min(sl_old, proposed)
                if new_sl < sl_old * (1 - min_step):
                    action = "LOWER_SL"
                    rationale_bits.append("ATR/resistance/1R progression suggests lower stop")
                else:
                    action = "KEEP"
            else:
                new_sl = sl_old

        # 3) Optional TP extension to maintain target R:R with the updated stop
        new_tp = tp_old
        if extend_tp:
            if side == Side.LONG:
                rr_tp = entry + rr_target * max(entry - new_sl, 0.0)
                new_tp = max(tp_old or 0.0, rr_tp)
                if tp_old is None or new_tp > (tp_old + max(1e-6, tp_old * min_step)):
                    rationale_bits.append(f"Extend TP to maintain ~{rr_target:.1f}:1 RR")
            else:
                rr_tp = entry - rr_target * max(new_sl - entry, 0.0)
                new_tp = min(tp_old or float("inf"), rr_tp)
                if tp_old is None or new_tp < (tp_old - max(1e-6, (tp_old if np.isfinite(tp_old) else 1.0) * min_step)):
                    rationale_bits.append(f"Extend TP to maintain ~{rr_target:.1f}:1 RR")

        rationale = ", ".join(rationale_bits) if rationale_bits else "No material change"
        return Recommendation(action, float(new_sl), (float(new_tp) if new_tp is not None else None), float(cur_price), rationale, "OK")
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        return Recommendation("KEEP", trade.stop_loss, trade.take_profit, 0.0, f"Error: {e}", "ERROR")


def _load_trades_from_file(path: str) -> List[Trade]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    trades: List[Trade] = []
    if isinstance(data, list):
        for item in data:
            trades.append(Trade(
                symbol=item.get("symbol"),
                product_id=item.get("product_id"),
                side=item.get("side", "LONG"),
                entry_price=float(item.get("entry_price", 0) or 0),
                stop_loss=float(item.get("stop_loss", 0) or 0),
                take_profit=float(item.get("take_profit")) if item.get("take_profit") is not None else None,
                initial_stop=float(item.get("initial_stop")) if item.get("initial_stop") is not None else None,
                opened_at=item.get("opened_at"),
                quantity=float(item.get("quantity")) if item.get("quantity") is not None else None,
            ))
    else:
        raise ValueError("Trade file must contain a JSON list of trades")
    return trades


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily stop/target manager for open trades (Coinbase products)")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--file", type=str, help="Path to JSON list of trades")
    src.add_argument("--symbol", type=str, help="Symbol (e.g., BTC)")
    ap.add_argument("--product-id", type=str, help="Override product_id (e.g., BTC-USDC)")
    ap.add_argument("--side", type=str, choices=["long", "short"], help="Position side for single trade")
    ap.add_argument("--entry", type=float, help="Entry price for single trade")
    ap.add_argument("--sl", type=float, help="Current stop loss for single trade")
    ap.add_argument("--tp", type=float, help="Current take profit for single trade (optional)")
    ap.add_argument("--initial-sl", type=float, help="Initial stop at entry time (optional)")
    ap.add_argument("--quantity", type=float, help="Quantity (optional, informational)")

    ap.add_argument("--output", type=str, choices=["console", "json"], default="console", help="Output format")
    ap.add_argument("--extend-tp", action="store_true", help="Extend TP to maintain target R:R with new stop")
    ap.add_argument("--rr-target", type=float, default=3.0, help="Target reward:risk when extending TP (default 3.0)")
    ap.add_argument("--analysis-days", type=int, default=365, help="Lookback days for indicators (default 365)")
    ap.add_argument("--offline", action="store_true", help="Avoid external HTTP when possible (use cache)")
    ap.add_argument("--quotes", type=str, default="USDC,USD,USDT", help="Preferred quote currencies")

    args = ap.parse_args()

    # Build finder for indicator consistency and product mapping
    quotes = [q.strip().upper() for q in (args.quotes or "USDC,USD,USDT").split(",") if q.strip()]
    cfg = CryptoFinderConfig.from_env()
    cfg.analysis_days = args.analysis_days
    cfg.offline = bool(args.offline)
    cfg.quotes = quotes
    finder = LongTermCryptoFinder(config=cfg)

    if args.file:
        trades = _load_trades_from_file(args.file)
    else:
        if not (args.symbol and args.side and args.entry is not None and args.sl is not None):
            print("Provide --file or all of: --symbol --side --entry --sl", file=sys.stderr)
            sys.exit(2)
        trades = [Trade(
            symbol=args.symbol,
            product_id=args.product_id,
            side=args.side,
            entry_price=args.entry,
            stop_loss=args.sl,
            take_profit=args.tp,
            initial_stop=args.initial_sl,
            opened_at=None,
            quantity=args.quantity,
        )]

    results: List[Dict[str, Union[str, float]]] = []
    for t in trades:
        pid_resolved = _resolve_product_id(finder, t, quotes) or t.product_id or ""
        base_from_pid = pid_resolved.split("-")[0] if pid_resolved else (t.symbol or "")
        rec = _recommend_for_trade(finder, t, args.analysis_days, bool(args.extend_tp), float(args.rr_target), quotes)
        row = {
            "symbol": base_from_pid,
            "product_id": pid_resolved,
            "side": t.side,
            "entry_price": _finite(t.entry_price),
            "stop_loss": _finite(t.stop_loss),
            "take_profit": _finite(t.take_profit) if t.take_profit is not None else None,
            "current_price": _finite(rec.current_price),
            "recommended_stop": _finite(rec.new_stop_loss),
            "recommended_takeprofit": _finite(rec.new_take_profit) if rec.new_take_profit is not None else None,
            "action": rec.action,
            "status": rec.status,
            "rationale": rec.rationale,
        }
        results.append(row)

    if args.output == "json":
        print(json.dumps({
            "version": "1.0",
            "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "results": results
        }, indent=2))
    else:
        print("\n=== Trade Guardian Recommendations ===")
        print(f"Generated (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['symbol']} ({r['product_id']}) — {r['side']}")
            print(f"   Entry: ${r['entry_price']:.6f}  Current: ${r['current_price']:.6f}")
            sl = r['stop_loss']
            sln = r['recommended_stop']
            tp = r['take_profit']
            tpn = r['recommended_takeprofit']
            print(f"   Stop:  old ${sl:.6f} -> new ${sln:.6f}")
            if tp is not None or tpn is not None:
                tp_old_s = f"${tp:.6f}" if tp is not None else "N/A"
                tp_new_s = f"${tpn:.6f}" if tpn is not None else "N/A"
                print(f"   Target: old {tp_old_s} -> new {tp_new_s}")
            print(f"   Action: {r['action']}  Status: {r['status']}")
            print(f"   Reason: {r['rationale']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
