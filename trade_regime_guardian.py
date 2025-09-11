#!/usr/bin/env python3
"""
Trade Regime Guardian — correlation/regime risk monitor for open trades

Purpose
- While `long_term_crypto_finder.py` helps find opportunities and
  `trade_guardian.py` updates SL/TP dynamically, this tool adds an
  in-trade check for adverse market regime and BTC-correlation risk.

What it does
- Computes rolling correlation between the instrument and BTC.
- Scores the BTC market regime from technicals (trend strength, ADX, MACD).
- Blends correlation and regime into a risk score (0..1).
- Emits a conservative recommendation:
    KEEP | RISK_OFF_TIGHTEN | EXIT_RISK_OFF | RISK_ON

Notes
- Uses Coinbase historical candles via the same `HistoricalData` used by
  `LongTermCryptoFinder` to stay consistent with indicators.
- Prefers spot bars for regime/correlation even if the position is a PERP.
- Does not place orders; prints JSON/console recommendations for downstream
  consumers (bots, UIs, or manual review).

Usage examples
  Single trade:
    python trade_regime_guardian.py --symbol SOL --side long --entry 155.2

  From JSON (same schema as trade_guardian):
    python trade_regime_guardian.py --file my_trades.json --output json
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

from long_term_crypto_finder import LongTermCryptoFinder, CryptoFinderConfig


# ----------------- Logging -----------------
def _setup_logger() -> logging.Logger:
    logs_dir = Path("logs") / "trade_regime_guardian"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("trade_regime_guardian")
    if logger.handlers:
        return logger
    level = os.getenv("TRADE_REGIME_GUARDIAN_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    fh = logging.FileHandler(logs_dir / "trade_regime_guardian.log", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    for h in (fh, ch):
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)
    logger.info("Trade Regime Guardian logging initialized")
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
    entry_price: Optional[float] = None
    opened_at: Optional[str] = None  # ISO8601

    def clean(self) -> None:
        if self.symbol:
            self.symbol = self.symbol.strip().upper()
        if self.product_id:
            self.product_id = self.product_id.strip().upper()
        s = str(self.side or "LONG").upper()
        self.side = Side.SHORT.value if s in ("SHORT", "S") else Side.LONG.value


@dataclass
class RegimeRecommendation:
    action: str  # KEEP | RISK_OFF_TIGHTEN | EXIT_RISK_OFF | RISK_ON | INSUFFICIENT_DATA
    risk_score: float
    correlation_btc: float
    btc_regime_score: float
    current_price: float
    rationale: str
    status: str  # OK | INSUFFICIENT_DATA | ERROR


# ----------------- Helpers -----------------
def _finite(x: Union[float, int]) -> float:
    try:
        xv = float(x)
        if not np.isfinite(xv):
            return 0.0
        return xv
    except Exception:
        return 0.0


def _normalize_perp_product(sym: str) -> Optional[str]:
    """Normalize user inputs to Coinbase INTX perp product id.

    Accepts forms like:
      - "BTC-PERP-INTX" (already normalized)
      - "BTC-INTX-PERP" (reordered)
      - "BTC-PERP" (append INTX)
      - "BTC-INTX" (not perp) -> None
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
    if len(parts) >= 2 and parts[1] == "PERP" and ((len(parts) >= 3 and parts[2] == "INTX") or not has_intx):
        return f"{base}-PERP-INTX" if has_intx or len(parts) < 3 else s
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

    # If already a product id
    if "-" in sym and sym.split("-")[-1] in {"USD", "USDC", "USDT", "EUR", "GBP"}:
        return sym

    # Map base symbol to preferred quote via products map
    prod_map = finder._fetch_usdc_products()
    rec = prod_map.get(sym)
    if rec and rec.get("product_id"):
        return rec["product_id"]
    for q in quotes:
        return f"{sym}-{q.upper()}"
    return None


def _btc_spot_product_id(finder: LongTermCryptoFinder, quotes: List[str]) -> str:
    """Pick a BTC spot product id using preferred quotes (default: USDC,USD,USDT)."""
    products = finder._fetch_usdc_products()  # symbol -> {product_id, base_name, quote}
    btc = products.get("BTC")
    if btc and btc.get("product_id"):
        return str(btc["product_id"])  # e.g., BTC-USDC
    # Fallback
    for q in quotes:
        if q.upper() in {"USDC", "USD", "USDT"}:
            return f"BTC-{q.upper()}"
    return "BTC-USDC"


def _latest_price_from_hourly(finder: LongTermCryptoFinder, product_id: str) -> Optional[Tuple[float, datetime]]:
    try:
        end = datetime.utcnow()
        start = end - timedelta(hours=24)
        candles = finder.historical_data.get_historical_data(product_id, start, end, "ONE_HOUR")
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
    except Exception:
        return None


def _daily_df(finder: LongTermCryptoFinder, product_id: str, days: int) -> Optional[pd.DataFrame]:
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    try:
        raw = finder.historical_data.get_historical_data(product_id, start, end, "ONE_DAY")
        if not raw:
            return None
        df = pd.DataFrame(raw)
        # normalize columns
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
        # unify timestamp index
        if "start" in df.columns:
            df["ts"] = pd.to_datetime(pd.to_numeric(df["start"], errors="coerce"), unit="s", utc=True)
        elif "time" in df.columns:
            df["ts"] = pd.to_datetime(pd.to_numeric(df["time"], errors="coerce"), unit="s", utc=True)
        else:
            df["ts"] = pd.RangeIndex(len(df))
        df = df.dropna(subset=["close"]).set_index("ts").sort_index()
        return df
    except Exception:
        return None


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> float:
    try:
        x = np.log(a).diff().dropna()
        y = np.log(b).diff().dropna()
        j = x.index.intersection(y.index)
        if len(j) < max(window // 2, 10):
            return 0.0
        return float(x.loc[j].tail(window).corr(y.loc[j].tail(window)))
    except Exception:
        return 0.0


def _btc_regime_score(tech: Dict) -> float:
    """Map BTC technicals into a regime score in [-1, 1].

    - Trend strength (bps) dominates
    - ADX contributes when trend is strong
    - MACD histogram sign as tie-breaker
    """
    try:
        trend_strength = float(tech.get("trend_strength", 0.0) or 0.0)  # bps over ~60d (finder-defined)
        adx = float(tech.get("adx", 0.0) or 0.0)
        macd_hist = float(tech.get("macd_hist", 0.0) or 0.0)

        # Normalize trend_strength to [-1,1] around +/- 100bps per 60d
        t_norm = max(-1.0, min(1.0, trend_strength / 100.0))
        # ADX 10..40 -> 0..1, then signed by trend direction
        adx_comp = max(0.0, min(1.0, (adx - 10.0) / 30.0))
        adx_signed = adx_comp if t_norm >= 0 else -adx_comp
        macd_comp = 1.0 if macd_hist >= 0 else -1.0
        # Blend (trend is primary)
        score = 0.6 * t_norm + 0.3 * adx_signed + 0.1 * macd_comp
        return max(-1.0, min(1.0, float(score)))
    except Exception:
        return 0.0


def _risk_score(corr_btc: float, btc_regime: float, side: Side) -> float:
    """Compute a 0..1 risk score from correlation and BTC regime vs the trade side.

    High positive correlation + adverse BTC regime -> high risk.
    """
    corr = abs(float(corr_btc))  # risk uses magnitude of linkage
    # For longs, negative regime is adverse; for shorts, positive regime is adverse
    adverse = -btc_regime if side == Side.LONG else btc_regime
    adverse01 = (adverse + 1.0) / 2.0  # map [-1,1] -> [0,1]
    # Blend with a slight convexity on correlation
    score = 0.6 * (corr ** 1.2) + 0.4 * adverse01
    return max(0.0, min(1.0, float(score)))


# ----------------- Core logic -----------------
def _recommend_for_trade(
    finder: LongTermCryptoFinder,
    trade: Trade,
    analysis_days: int,
    corr_window: int,
    quotes: List[str],
    tighten_threshold: float,
    exit_threshold: float,
) -> RegimeRecommendation:
    try:
        trade.clean()
        pid = _resolve_product_id(finder, trade, quotes)
        if not pid:
            return RegimeRecommendation("INSUFFICIENT_DATA", 0.0, 0.0, 0.0, 0.0, "No product_id/symbol mapping", "INSUFFICIENT_DATA")

        # Prefer spot bars for regime/correlation even if the position is on PERP
        bars_pid = pid
        if "PERP" in pid:
            base = pid.split("-")[0]
            spot_map = finder._fetch_usdc_products()
            spid = (spot_map.get(base) or {}).get("product_id")
            if spid:
                bars_pid = spid

        # Load daily bars for instrument and BTC using the finder (ensures 'price' column)
        df_inst = finder.get_historical_data(bars_pid, days=analysis_days)
        btc_pid = _btc_spot_product_id(finder, quotes)
        df_btc = finder.get_historical_data(btc_pid, days=analysis_days)
        if df_inst is None or df_btc is None or len(df_inst) < 30 or len(df_btc) < 30:
            return RegimeRecommendation("INSUFFICIENT_DATA", 0.0, 0.0, 0.0, 0.0, "Insufficient daily bars for correlation/regime", "INSUFFICIENT_DATA")

        # BTC regime technicals using finder's indicator set
        tech_btc = finder.calculate_technical_indicators(df_btc)
        btc_regime = _btc_regime_score(tech_btc)

        # Rolling correlation (window)
        corr_btc = _rolling_corr(df_inst["price"], df_btc["price"], window=int(corr_window))

        # Current price from hourly (prefer instrument product id original so it matches what user sees)
        lp = _latest_price_from_hourly(finder, pid) or _latest_price_from_hourly(finder, bars_pid)
        cur_price = float(lp[0]) if lp else float(df_inst["close"].iloc[-1])

        # Risk score
        side = Side(trade.side)
        rscore = _risk_score(corr_btc, btc_regime, side)

        # Policy thresholds
        action = "KEEP"
        rationale_bits: List[str] = []
        rationale_bits.append(f"BTC regime {btc_regime:+.2f} (trend/ADX/MACD blend)")
        rationale_bits.append(f"{bars_pid} vs BTC correlation {corr_btc:+.2f} (window {corr_window}d)")

        if rscore >= exit_threshold and abs(corr_btc) >= 0.6:
            action = "EXIT_RISK_OFF"
            rationale_bits.append(f"Risk score {rscore:.2f} >= exit threshold {exit_threshold:.2f}")
        elif rscore >= tighten_threshold and abs(corr_btc) >= 0.4:
            action = "RISK_OFF_TIGHTEN"
            rationale_bits.append(f"Risk score {rscore:.2f} >= tighten threshold {tighten_threshold:.2f}")
        else:
            if (side == Side.LONG and btc_regime > 0.3 and corr_btc > 0.5) or (side == Side.SHORT and btc_regime < -0.3 and corr_btc > 0.5):
                action = "RISK_ON"
                rationale_bits.append("Favorable BTC regime with positive linkage")

        rationale = ", ".join(rationale_bits)
        return RegimeRecommendation(action, float(rscore), float(corr_btc), float(btc_regime), float(cur_price), rationale, "OK")
    except Exception as e:
        logger.exception("Regime recommendation failed")
        return RegimeRecommendation("INSUFFICIENT_DATA", 0.0, 0.0, 0.0, 0.0, f"Error: {e}", "ERROR")


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
                entry_price=float(item.get("entry_price")) if item.get("entry_price") is not None else None,
                opened_at=item.get("opened_at"),
            ))
    else:
        raise ValueError("Trade file must contain a JSON list of trades")
    return trades


# ----------------- CLI -----------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Market-regime and BTC-correlation guard for open trades")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--file", type=str, help="Path to JSON list of trades")
    src.add_argument("--symbol", type=str, help="Symbol (e.g., SOL or SOL-USDC or SOL-PERP-INTX)")
    ap.add_argument("--product-id", type=str, help="Override product_id (e.g., SOL-USDC)")
    ap.add_argument("--side", type=str, choices=["long", "short"], help="Position side for single trade")
    ap.add_argument("--entry", type=float, help="Entry price for single trade (optional, informational)")

    ap.add_argument("--analysis-days", type=int, default=180, help="Lookback days for regime/correlation (default 180)")
    ap.add_argument("--corr-window", type=int, default=45, help="Rolling window (days) for correlation (default 45)")
    ap.add_argument("--quotes", type=str, default="USDC,USD,USDT", help="Preferred quote currencies")
    ap.add_argument("--tighten-threshold", type=float, default=0.50, help="Risk score to suggest tightening (default 0.50)")
    ap.add_argument("--exit-threshold", type=float, default=0.70, help="Risk score to suggest risk-off exit (default 0.70)")
    ap.add_argument("--output", type=str, choices=["console", "json"], default="console", help="Output format")
    ap.add_argument("--offline", action="store_true", help="Avoid external HTTP where possible (use cache)")

    args = ap.parse_args()

    quotes = [q.strip().upper() for q in (args.quotes or "USDC,USD,USDT").split(",") if q.strip()]
    cfg = CryptoFinderConfig.from_env()
    cfg.analysis_days = max(cfg.analysis_days, int(args.analysis_days))
    cfg.offline = bool(args.offline)
    cfg.quotes = quotes
    finder = LongTermCryptoFinder(config=cfg)

    if args.file:
        trades = _load_trades_from_file(args.file)
    else:
        if not args.symbol or not args.side:
            print("Provide --file or both: --symbol --side", file=sys.stderr)
            sys.exit(2)
        trades = [Trade(
            symbol=args.symbol,
            product_id=args.product_id,
            side=args.side,
            entry_price=args.entry,
            opened_at=None,
        )]

    results: List[Dict[str, Union[str, float]]] = []
    for t in trades:
        rec = _recommend_for_trade(
            finder,
            t,
            analysis_days=int(args.analysis_days),
            corr_window=int(args.corr_window),
            quotes=quotes,
            tighten_threshold=float(args.tighten_threshold),
            exit_threshold=float(args.exit_threshold),
        )
        pid_resolved = _resolve_product_id(finder, t, quotes) or t.product_id or ""
        base_from_pid = pid_resolved.split("-")[0] if pid_resolved else (t.symbol or "")
        row = {
            "symbol": base_from_pid,
            "product_id": pid_resolved,
            "side": t.side,
            "entry_price": _finite(t.entry_price) if t.entry_price is not None else None,
            "current_price": _finite(rec.current_price),
            "action": rec.action,
            "status": rec.status,
            "risk_score": float(rec.risk_score),
            "corr_btc": float(rec.correlation_btc),
            "btc_regime": float(rec.btc_regime_score),
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
        print("\n=== Trade Regime Guardian ===")
        print(f"Generated (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['symbol']} ({r['product_id']}) — {r['side']}")
            cp = r['current_price']
            cp_s = f"${cp:.6f}" if isinstance(cp, (int, float)) else str(cp)
            print(f"   Current: {cp_s}")
            print(f"   Action: {r['action']}  Status: {r['status']}")
            print(f"   Risk Score: {r['risk_score']:.2f}  Corr(BTC): {r['corr_btc']:+.2f}  BTC Regime: {r['btc_regime']:+.2f}")
            print(f"   Reason: {r['rationale']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
