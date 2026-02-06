from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from coinbaseservice import CoinbaseService
from credentials import get_perps_credentials, get_primary_credentials
from fills_pnl import fetch_fills


UTC = timezone.utc


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def portfolio_sync_enabled() -> bool:
    return _env_bool("ENABLE_PORTFOLIO_SYNC", True)


def get_coinbase_credentials() -> Tuple[str, str]:
    """Resolve Coinbase credentials with COINBASE_* taking priority."""

    key = os.getenv("COINBASE_API_KEY", "").strip()
    secret = os.getenv("COINBASE_API_SECRET", "").strip()
    if key and secret:
        return key, secret
    return get_primary_credentials()


def get_coinbase_perps_credentials() -> Tuple[str, str]:
    key = os.getenv("COINBASE_PERP_API_KEY", "").strip()
    secret = os.getenv("COINBASE_PERP_API_SECRET", "").strip()
    if key and secret:
        return key, secret
    return get_perps_credentials()


def _as_dict(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        try:
            data = vars(value)
        except Exception:
            return None
        if isinstance(data, dict):
            return data
    return None


def _extract_money_value(container: Any) -> Optional[float]:
    if container is None:
        return None
    if isinstance(container, dict):
        raw = container.get("value")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    if hasattr(container, "value"):
        try:
            return float(getattr(container, "value"))
        except (TypeError, ValueError):
            return None
    try:
        return float(container)
    except (TypeError, ValueError):
        return None


def fetch_portfolio_balances(cb: CoinbaseService) -> Dict[str, Any]:
    """Fetch summary balances across Coinbase portfolios (best-effort)."""

    try:
        ports = cb.client.get_portfolios()
    except Exception:
        return {}

    if isinstance(ports, dict):
        portfolios_list = ports.get("portfolios", []) or []
    else:
        portfolios_list = getattr(ports, "portfolios", []) or []

    rows: List[Dict[str, Any]] = []
    totals = {
        "total_balance": 0.0,
        "available_balance": 0.0,
    }

    for p in portfolios_list:
        pdict = _as_dict(p) or {}
        p_type = pdict.get("type") or pdict.get("portfolio_type") or ""
        p_uuid = pdict.get("uuid") or pdict.get("portfolio_uuid") or ""
        if not p_uuid:
            continue
        try:
            breakdown = cb.client.get_portfolio_breakdown(portfolio_uuid=p_uuid)
        except Exception:
            continue

        bdict = _as_dict(breakdown) or {}
        inner = bdict.get("breakdown") if isinstance(bdict.get("breakdown"), dict) else bdict.get("breakdown")
        if inner is None:
            inner = bdict
        if not isinstance(inner, dict):
            inner = _as_dict(inner) or {}
        balances = inner.get("portfolio_balances")
        if balances is None and isinstance(inner.get("breakdown"), dict):
            balances = inner["breakdown"].get("portfolio_balances")
        balances_dict = _as_dict(balances) or {}

        total_val = _extract_money_value(balances_dict.get("total_balance"))
        avail_val = _extract_money_value(balances_dict.get("available_balance"))

        if total_val is not None:
            totals["total_balance"] += float(total_val)
        if avail_val is not None:
            totals["available_balance"] += float(avail_val)

        rows.append(
            {
                "type": p_type,
                "uuid": p_uuid,
                "total_balance": total_val,
                "available_balance": avail_val,
            }
        )

    return {"totals": totals, "portfolios": rows}


def fetch_recent_fills_last_24h(cb: CoinbaseService, *, limit: int = 500) -> List[Dict[str, Any]]:
    fills = fetch_fills(cb, limit=int(limit))
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    recent = [f for f in fills if isinstance(f.get("time"), datetime) and f["time"] >= cutoff]
    recent.sort(key=lambda f: f["time"], reverse=True)
    return recent


def build_positions_alignment_table(
    open_positions_df: pd.DataFrame,
    regimes_by_asset: Dict[str, str],
) -> pd.DataFrame:
    if open_positions_df is None or open_positions_df.empty:
        return pd.DataFrame()

    df = open_positions_df.copy()
    if "Product" in df.columns:
        product_col = "Product"
    else:
        product_col = "product_id" if "product_id" in df.columns else None
    if product_col is None:
        return pd.DataFrame()

    def _base_asset(product_id: str) -> str:
        pid = (product_id or "").upper()
        if "-PERP-" in pid:
            return pid.split("-PERP-")[0]
        if "-" in pid:
            return pid.split("-")[0]
        return pid

    def _alignment(product_id: str, side: str) -> Tuple[str, str]:
        base = _base_asset(product_id)
        regime = regimes_by_asset.get(base)
        if not regime:
            return "n/a", "n/a"
        s = (side or "").upper()
        if regime == "BULLISH" and s == "LONG":
            return regime, "With trend"
        if regime == "BEARISH" and s == "SHORT":
            return regime, "With trend"
        if regime in {"BULLISH", "BEARISH"} and s in {"LONG", "SHORT"}:
            return regime, "Against trend"
        return regime, "n/a"

    regimes: List[str] = []
    aligns: List[str] = []
    for _, row in df.iterrows():
        pid = str(row.get(product_col, "") or "")
        side = str(row.get("Side") or row.get("side") or "")
        r, a = _alignment(pid, side)
        regimes.append(r)
        aligns.append(a)
    df["Regime"] = regimes
    df["Alignment"] = aligns
    return df

