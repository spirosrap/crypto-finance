#!/usr/bin/env python3
"""
Trade Unified Guardian — combines stop/TP management with market-regime risk

This tool aggregates outputs from:
  - trade_guardian.py          (stop/TP recommendation per trade)
  - trade_regime_guardian.py   (BTC correlation and regime risk recommendation)

It resolves the product once, computes indicators via LongTermCryptoFinder,
and emits a unified view per trade with a combined suggested action.

Usage examples
  - Single trade (console):
      python trade_unified_guardian.py --symbol BTC --side long \
        --entry 60000 --sl 54000 --tp 78000

  - From JSON file (same schema as trade_guardian):
      python trade_unified_guardian.py --file my_trades.json --output json

Notes
  - The combined suggestion is conservative: EXIT_HIT > EXIT_RISK_OFF >
    tighten/raise/lower SL > KEEP. When RISK_OFF_TIGHTEN aligns with
    a stop update, both are surfaced in the message.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Import engines (keep module namespace to avoid helper name clashes)
import trade_guardian as tg
import trade_regime_guardian as rg
from long_term_crypto_finder import LongTermCryptoFinder, CryptoFinderConfig


def _finite(x: Union[float, int, None]) -> float:
    try:
        xv = float(x) if x is not None else 0.0
        return xv if np.isfinite(xv) else 0.0
    except Exception:
        return 0.0


def _combined_action(
    trade_action: str,
    trade_status: str,
    regime_action: str,
    regime_status: str,
) -> Tuple[str, str]:
    """Combine granular actions into a single conservative suggestion.

    Priority order:
      1) EXIT_HIT (from trade engine)
      2) EXIT_RISK_OFF (from regime engine)
      3) RISK_OFF_TIGHTEN + any stop update (RAISE_SL/LOWER_SL)
      4) RAISE_SL / LOWER_SL
      5) RISK_ON (if no stop change)
      6) KEEP

    Returns (action, rationale_suffix).
    """
    # Hard exits
    if trade_status == "EXIT_HIT" or trade_action == "EXIT_HIT":
        return "EXIT_HIT", "stop/target already hit"
    if regime_action == "EXIT_RISK_OFF" and regime_status == "OK":
        return "EXIT_RISK_OFF", "adverse BTC regime + high correlation"

    # Tighten + stop update synergy
    if regime_action == "RISK_OFF_TIGHTEN" and trade_action in ("RAISE_SL", "LOWER_SL"):
        dir_txt = "raise" if trade_action == "RAISE_SL" else "lower"
        return f"{trade_action}+RISK_OFF_TIGHTEN", f"conservative {dir_txt} due to risk-off regime"

    # Plain stop update
    if trade_action in ("RAISE_SL", "LOWER_SL"):
        return trade_action, "price/ATR/swing logic"

    # Positive regime signaling (only if no stop change)
    if regime_action == "RISK_ON" and trade_action == "KEEP":
        return "KEEP", "favorable BTC regime; keep stops"

    return "KEEP", "no change warranted"


def _load_trades_from_file(path: str) -> List[tg.Trade]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Trade file must be a JSON list")
    trades: List[tg.Trade] = []
    for item in data:
        trades.append(
            tg.Trade(
                symbol=item.get("symbol"),
                product_id=item.get("product_id"),
                side=item.get("side", "LONG"),
                entry_price=float(item.get("entry_price", 0) or 0),
                stop_loss=float(item.get("stop_loss", 0) or 0),
                take_profit=(
                    float(item.get("take_profit"))
                    if item.get("take_profit") is not None
                    else None
                ),
                initial_stop=(
                    float(item.get("initial_stop"))
                    if item.get("initial_stop") is not None
                    else None
                ),
                opened_at=item.get("opened_at"),
                quantity=(
                    float(item.get("quantity"))
                    if item.get("quantity") is not None
                    else None
                ),
            )
        )
    return trades


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified stop/TP + regime risk guardian")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--file", type=str, help="Path to JSON list of trades")
    src.add_argument("--symbol", type=str, help="Symbol (e.g., BTC or BTC-USDC or BTC-PERP-INTX)")
    ap.add_argument("--product-id", type=str, help="Override product_id (e.g., BTC-USDC)")
    ap.add_argument("--side", type=str, choices=["long", "short"], help="Position side for single trade")
    ap.add_argument("--entry", type=float, help="Entry price for single trade")
    ap.add_argument("--sl", type=float, help="Current stop loss for single trade")
    ap.add_argument("--tp", type=float, help="Current take profit for single trade (optional)")
    ap.add_argument("--initial-sl", type=float, help="Initial stop at entry time (optional)")
    ap.add_argument("--quantity", type=float, help="Quantity (optional, informational)")

    # Output + shared options
    ap.add_argument("--output", type=str, choices=["console", "json"], default="console", help="Output format")
    ap.add_argument("--quotes", type=str, default="USDC,USD,USDT", help="Preferred quote currencies")
    ap.add_argument("--offline", action="store_true", help="Avoid external HTTP when possible (use cache)")

    # Trade-guardian tuning
    ap.add_argument("--analysis-days", type=int, default=365, help="Lookback days for indicators (default 365)")
    ap.add_argument("--extend-tp", action="store_true", help="Extend TP to maintain target R:R with new stop")
    ap.add_argument("--rr-target", type=float, default=3.0, help="Target reward:risk when extending TP (default 3.0)")

    # Regime-guardian tuning
    ap.add_argument("--regime-analysis-days", type=int, default=180, help="Lookback days for regime/correlation (default 180)")
    ap.add_argument("--corr-window", type=int, default=45, help="Rolling window (days) for correlation (default 45)")
    ap.add_argument("--tighten-threshold", type=float, default=0.50, help="Risk score to suggest tightening (default 0.50)")
    ap.add_argument("--exit-threshold", type=float, default=0.70, help="Risk score to suggest risk-off exit (default 0.70)")

    ap.add_argument("--auto-tighten-on-regime", action="store_true", default=True,
                    help="When regime is RISK_OFF_TIGHTEN but stop engine keeps, auto-tighten SL using ATR cushion")

    args = ap.parse_args()

    # Finder once for both engines
    quotes = [q.strip().upper() for q in (args.quotes or "USDC,USD,USDT").split(",") if q.strip()]
    cfg = CryptoFinderConfig.from_env()
    cfg.analysis_days = max(int(args.analysis_days), int(args.regime_analysis_days))
    cfg.offline = bool(args.offline)
    cfg.quotes = quotes
    finder = LongTermCryptoFinder(config=cfg)

    # Build trades list
    if args.file:
        trades = _load_trades_from_file(args.file)
    else:
        if not (args.symbol and args.side and args.entry is not None and args.sl is not None):
            raise SystemExit("Provide --file or all of: --symbol --side --entry --sl")
        trades = [
            tg.Trade(
                symbol=args.symbol,
                product_id=args.product_id,
                side=args.side,
                entry_price=args.entry,
                stop_loss=args.sl,
                take_profit=args.tp,
                initial_stop=args.initial_sl,
                opened_at=None,
                quantity=args.quantity,
            )
        ]

    results: List[Dict[str, Union[str, float]]] = []
    for t in trades:
        # Resolve product once
        pid_resolved = tg._resolve_product_id(finder, t, quotes) or t.product_id or ""
        base_from_pid = pid_resolved.split("-")[0] if pid_resolved else (t.symbol or "")

        # Run stop/TP engine
        rec_tg = tg._recommend_for_trade(
            finder,
            t,
            analysis_days=int(args.analysis_days),
            extend_tp=bool(args.extend_tp),
            rr_target=float(args.rr_target),
            quotes=quotes,
        )

        # Build a minimal RG Trade from TG Trade fields
        t_rg = rg.Trade(
            symbol=t.symbol,
            product_id=t.product_id,
            side=t.side,
            entry_price=t.entry_price,
            opened_at=t.opened_at,
        )

        # Run regime engine
        rec_rg = rg._recommend_for_trade(
            finder,
            t_rg,
            analysis_days=int(args.regime_analysis_days),
            corr_window=int(args.corr_window),
            quotes=quotes,
            tighten_threshold=float(args.tighten_threshold),
            exit_threshold=float(args.exit_threshold),
        )

        # Optional auto-tighten overlay when regime warns but stop engine kept
        overlay_sl: Optional[float] = None
        overlay_action: Optional[str] = None
        if args.auto_tighten_on_regime and rec_rg.action == "RISK_OFF_TIGHTEN" and rec_tg.action == "KEEP" and rec_tg.status == "OK":
            try:
                # Compute ATR-based cushion using same bars policy as trade engine
                bars_pid = pid_resolved
                if "PERP" in (pid_resolved or ""):
                    try:
                        base = pid_resolved.split("-")[0]
                        spot_map = finder._fetch_usdc_products()
                        spid = (spot_map.get(base) or {}).get("product_id")
                        if spid:
                            bars_pid = spid
                    except Exception:
                        pass
                df_for_atr = finder.get_historical_data(bars_pid or (t.symbol or ""), days=int(args.analysis_days))
                tech = finder.calculate_technical_indicators(df_for_atr) if df_for_atr is not None else {"atr": 0.0}
                atr = float(tech.get("atr", 0.0) or 0.0)
                cur_price = float(rec_tg.current_price or 0.0)
                old_sl = float(t.stop_loss or 0.0)
                min_gap = float(os.getenv("TRADE_GUARD_MIN_GAP_PCT", "0.001"))  # 0.1%
                min_step = float(os.getenv("TRADE_GUARD_MIN_STEP_PCT", "0.0025"))  # 0.25%
                min_atr_cush = float(os.getenv("TRADE_GUARD_MIN_ATR_CUSH", "0.6"))
                risk_off_cush = float(os.getenv("UNIFIED_RISK_OFF_CUSH_ATR", "1.0"))
                target_cush = max(min_atr_cush, risk_off_cush)
                if atr > 0 and cur_price > 0 and old_sl > 0:
                    if str(t.side).upper() == "LONG":
                        proposal = min(cur_price * (1 - min_gap), cur_price - target_cush * atr)
                        overlay_sl = max(old_sl, proposal)
                        if not (overlay_sl > old_sl * (1 + min_step)):
                            overlay_sl = None
                        else:
                            overlay_action = "RAISE_SL"
                    else:  # SHORT
                        proposal = max(cur_price * (1 + min_gap), cur_price + target_cush * atr)
                        overlay_sl = min(old_sl, proposal)
                        if not (overlay_sl < old_sl * (1 - min_step)):
                            overlay_sl = None
                        else:
                            overlay_action = "LOWER_SL"
            except Exception:
                overlay_sl = None
                overlay_action = None

            # Fallback: percentage-based tighten to ensure a concrete proposal
            if overlay_sl is None:
                try:
                    step_pct = float(os.getenv("UNIFIED_RISK_OFF_STEP_PCT", "0.005"))  # 0.5%
                    cur_price = float(rec_tg.current_price or 0.0)
                    old_sl = float(t.stop_loss or 0.0)
                    if str(t.side).upper() == "LONG":
                        proposal = min(old_sl * (1.0 + step_pct), cur_price * (1.0 - min_gap)) if cur_price > 0 else old_sl * (1.0 + step_pct)
                        if proposal > old_sl:
                            overlay_sl = proposal
                            overlay_action = "RAISE_SL"
                    else:
                        proposal = max(old_sl * (1.0 - step_pct), cur_price * (1.0 + min_gap)) if cur_price > 0 else old_sl * (1.0 - step_pct)
                        if proposal < old_sl:
                            overlay_sl = proposal
                            overlay_action = "LOWER_SL"
                except Exception:
                    overlay_sl = None
                    overlay_action = None

        # Combine (consider overlay_action if present)
        eff_trade_action = overlay_action if overlay_action else rec_tg.action
        combined_action, combo_reason = _combined_action(
            trade_action=eff_trade_action,
            trade_status=rec_tg.status,
            regime_action=rec_rg.action,
            regime_status=rec_rg.status,
        )

        row: Dict[str, Union[str, float]] = {
            # identity
            "symbol": base_from_pid,
            "product_id": pid_resolved,
            "side": t.side,
            # prices
            "entry_price": _finite(t.entry_price),
            "current_price": _finite(rec_tg.current_price or rec_rg.current_price),
            # current and recommended levels
            "stop_loss": _finite(t.stop_loss),
            "take_profit": _finite(t.take_profit) if t.take_profit is not None else None,
            "recommended_stop": _finite(overlay_sl if overlay_sl is not None else rec_tg.new_stop_loss),
            "recommended_takeprofit": _finite(rec_tg.new_take_profit) if rec_tg.new_take_profit is not None else None,
            # per-engine actions
            "trade_action": eff_trade_action or rec_tg.action,
            "trade_status": rec_tg.status,
            "trade_confidence": float(rec_tg.confidence or 0.0),
            "trade_rationale": rec_tg.rationale,
            "regime_action": rec_rg.action,
            "regime_status": rec_rg.status,
            "risk_score": float(rec_rg.risk_score or 0.0),
            "corr_btc": float(rec_rg.correlation_btc or 0.0),
            "btc_regime": float(rec_rg.btc_regime_score or 0.0),
            "regime_rationale": rec_rg.rationale,
            # combined
            "combined_action": combined_action,
            "combined_note": combo_reason,
        }
        results.append(row)

    if args.output == "json":
        print(
            json.dumps(
                {
                    "version": "1.0",
                    "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "results": results,
                },
                indent=2,
            )
        )
        return

    # Console output
    print("\n=== Trade Unified Guardian ===")
    print(f"Generated (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['symbol']} ({r['product_id']}) — {r['side']}")
        entry = r["entry_price"] or 0.0
        cur = r["current_price"] or 0.0
        print(f"   Entry: ${entry:.6f}  Current: ${cur:.6f}")
        print(f"   Stop:  old ${r['stop_loss']:.6f} -> new ${r['recommended_stop']:.6f}")
        tp_old = r.get("take_profit")
        tp_new = r.get("recommended_takeprofit")
        if tp_old is not None or tp_new is not None:
            tp_old_s = f"${float(tp_old):.6f}" if tp_old is not None else "N/A"
            tp_new_s = f"${float(tp_new):.6f}" if tp_new is not None else "N/A"
            print(f"   Target: old {tp_old_s} -> new {tp_new_s}")
        print(f"   Trade Engine: {r['trade_action']}  Status: {r['trade_status']}  Conf: {float(r['trade_confidence'])*100:.1f}%")
        print(
            "   Regime Engine: {act}  Status: {st}  Risk: {rs:.2f}  Corr(BTC): {cb:+.2f}  BTC Regime: {br:+.2f}".format(
                act=r["regime_action"],
                st=r["regime_status"],
                rs=float(r["risk_score"]),
                cb=float(r["corr_btc"]),
                br=float(r["btc_regime"]),
            )
        )
        print(f"   Combined: {r['combined_action']} — {r['combined_note']}")
        # Compact reasons
        print(f"   Trade Reason:  {r['trade_rationale']}")
        print(f"   Regime Reason: {r['regime_rationale']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
