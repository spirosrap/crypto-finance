#!/usr/bin/env python3
"""
Paper trading harness for `short_term_crypto_finder.py` output.

Key capabilities:
- Parse finder text blocks and select a subset of trades to stage on paper.
- Persist open trades (with entry/TP/SL and allocation) to `trade_logs/`.
- Periodically refresh prices (Coinbase Advanced perps via CCXT by default),
  auto-closing trades when stops, take-profits, or expiries fire.
- Emit closed-trade rows that mirror the watchdog CSV schema so the existing
  dashboard/metrics stack can plot simulated equity curves.
- Track basic config such as starting equity, default leverage, and fallback
  position sizing logic via a small JSON file.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Set

import pandas as pd

from add_position_from_finder import ParsedFinder, parse_finder_text
from watchdog_close_old_positions import compute_mae_mfe_from_history

logger = logging.getLogger("paper_finder_simulator")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

UTC = timezone.utc
DEFAULT_CONFIG_PATH = Path("trade_logs/paper_finder_config.json")
OPEN_CSV = Path("trade_logs/paper_finder_open_positions.csv")
CLOSED_CSV = Path("trade_logs/paper_finder_closed_positions.csv")
DERIVED_PERPS_PATH = Path(__file__).resolve().with_name("derived_perps_intx.txt")
DEFAULT_EXCLUDED_PERPS_PATH = Path("config/excluded_perps.txt")

DEFAULT_CONFIG = {
    "initial_capital": 25000.0,
    "default_leverage": 3.0,
    "default_expiry_hours": 24,
    "default_position_pct": 3.0,
}

EXPIRY_BREAKEVEN_PCT = 0.10

OPEN_COLUMNS = [
    "trade_id",
    "symbol",
    "product_id",
    "position_side",
    "entry_price",
    "stop_loss",
    "take_profit",
    "partial_tp_pct",
    "partial_tp_rr",
    "partial_tp_price",
    "partial_tp_done",
    "partial_tp_move_sl",
    "position_usd",
    "leverage",
    "opened_at",
    "expires_at",
    "status",
    "last_price",
    "last_price_at",
    "unrealized_pnl",
    "unrealized_pct",
    "finder_score",
    "finder_rank",
    "recommended_position_pct",
    "tag",
    "notes",
]

CLOSED_COLUMNS = [
    "closed_at",
    "product_id",
    "position_side",
    "net_size",
    "leverage",
    "opened_at",
    "closure_reason",
    "entry_price",
    "exit_price",
    "profit_loss",
    "profit_loss_pct",
    "mae",
    "mfe",
    "duration_seconds",
]

THOUSAND_UNIT_BASES = {"SHIB", "BONK", "PEPE", "FLOKI"}
_SUPPORTED_PERPS: Optional[Set[str]] = None
_CCXT_PRODUCTS: Optional[Set[str]] = None
_EXCLUDED_PERPS: Optional[Set[str]] = None
_CB_SERVICE: Optional["CoinbaseService"] = None
_CB_SERVICE_READY: bool = False


def _isoformat(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value).astimezone(UTC)
    except Exception:
        return None


def _canonical_symbol(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    if not s:
        return ""
    if s.startswith("1000") and s[4:] in THOUSAND_UNIT_BASES:
        return s
    if s in THOUSAND_UNIT_BASES:
        return f"1000{s}"
    return s


def _product_id(symbol: str) -> Optional[str]:
    base = _canonical_symbol(symbol)
    return f"{base}-PERP-INTX" if base else None


def _load_supported_perps(path: Path = DERIVED_PERPS_PATH) -> Set[str]:
    entries: Set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip().upper()
                if not value or value.startswith("#"):
                    continue
                entries.add(value)
    except FileNotFoundError:
        logger.warning("Supported perps list %s not found; skipping symbol validation.", path)
    except Exception as exc:
        logger.warning("Unable to load supported perps from %s (%s)", path, exc)
    return entries


def _supported_perps() -> Set[str]:
    global _SUPPORTED_PERPS
    if _SUPPORTED_PERPS is None:
        _SUPPORTED_PERPS = _load_supported_perps()
    return _SUPPORTED_PERPS


def _ccxt_products() -> Set[str]:
    global _CCXT_PRODUCTS
    if _CCXT_PRODUCTS is not None:
        return _CCXT_PRODUCTS
    products: Set[str] = set()
    try:
        import ccxt  # type: ignore
    except ImportError:
        logger.debug("ccxt not available; skipping live market verification.")
        _CCXT_PRODUCTS = set()
        return _CCXT_PRODUCTS
    try:
        exchange = ccxt.coinbaseadvanced({"enableRateLimit": True})
        exchange.load_markets()
    except Exception as exc:
        logger.debug("Unable to load Coinbase markets via ccxt (%s); falling back to static list.", exc)
        _CCXT_PRODUCTS = set()
        return _CCXT_PRODUCTS
    for market in exchange.markets.values():
        if not market.get("contract"):
            continue
        symbol_ccxt = market.get("symbol") or ""
        if ":USDC" not in symbol_ccxt:
            continue
        base = market.get("base") or ""
        base = _canonical_symbol(str(base))
        if not base:
            continue
        products.add(f"{base}-PERP-INTX")
    _CCXT_PRODUCTS = products
    return _CCXT_PRODUCTS


def _is_supported_product(product_id: str) -> bool:
    product_id = (product_id or "").upper()
    if not product_id:
        return False
    ccxt_products = _ccxt_products()
    if ccxt_products:
        return product_id in ccxt_products
    products = _supported_perps()
    if products:
        return product_id in products
    return True


def _load_excluded_perps(path: Path = DEFAULT_EXCLUDED_PERPS_PATH) -> Set[str]:
    entries: Set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip().upper()
                if not value or value.startswith("#"):
                    continue
                entries.add(value)
    except FileNotFoundError:
        return set()
    except Exception as exc:
        logger.warning("Unable to read excluded perps from %s (%s)", path, exc)
    return entries


def _excluded_perps() -> Set[str]:
    global _EXCLUDED_PERPS
    if _EXCLUDED_PERPS is None:
        _EXCLUDED_PERPS = _load_excluded_perps()
    return _EXCLUDED_PERPS


def _get_perps_credentials() -> Tuple[str, str]:
    try:
        from credentials import get_perps_credentials
        return get_perps_credentials()
    except Exception:
        try:  # pragma: no cover - fallback for older setups
            from config import API_KEY_PERPS, API_SECRET_PERPS  # type: ignore
        except Exception:
            return ("", "")
        return (API_KEY_PERPS or "", API_SECRET_PERPS or "")  # type: ignore[arg-type]


def _get_coinbase_service() -> Optional["CoinbaseService"]:
    global _CB_SERVICE, _CB_SERVICE_READY
    if _CB_SERVICE_READY:
        return _CB_SERVICE
    _CB_SERVICE_READY = True
    api_key, api_secret = _get_perps_credentials()
    if not api_key or not api_secret:
        logger.warning("MAE/MFE disabled: Coinbase perps credentials not found.")
        return None
    try:
        from coinbaseservice import CoinbaseService
    except Exception as exc:
        logger.warning("MAE/MFE disabled: unable to import CoinbaseService (%s).", exc)
        return None
    try:
        _CB_SERVICE = CoinbaseService(api_key, api_secret)
    except Exception as exc:
        logger.warning("MAE/MFE disabled: failed to init CoinbaseService (%s).", exc)
        _CB_SERVICE = None
    return _CB_SERVICE


def _compute_mae_mfe(
    row: Dict[str, object],
    exit_price: float,
    close_time: datetime,
    position_usd_override: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    cb = _get_coinbase_service()
    if cb is None:
        return None, None
    product_id = str(row.get("product_id") or "")
    if not product_id:
        return None, None
    entry = _safe_float(row.get("entry_price"), default=exit_price)
    position_usd = (
        float(position_usd_override)
        if position_usd_override is not None and position_usd_override > 0
        else _safe_float(row.get("position_usd"), default=0.0)
    )
    if entry <= 0 or position_usd <= 0:
        return None, None
    net_size = position_usd / entry
    side = (row.get("position_side") or "LONG").upper()
    if side == "SHORT":
        net_size = -abs(net_size)
    open_time = _parse_iso(str(row.get("opened_at") or ""))
    try:
        return compute_mae_mfe_from_history(
            cb=cb,
            product_id=product_id,
            net_size=net_size,
            entry_price=entry,
            open_time=open_time,
            close_time=close_time,
            exit_price=exit_price,
        )
    except Exception as exc:
        logger.warning("MAE/MFE fetch failed for %s (%s)", product_id, exc)
        return None, None


def _ccxt_symbol(product_id: str) -> str:
    product_id = (product_id or "").strip().upper()
    if not product_id:
        raise ValueError("Missing product id")
    if product_id.endswith("-PERP-INTX"):
        base = product_id.split("-")[0]
        return f"{base}/USDC:USDC"
    if "/" in product_id:
        return product_id
    raise ValueError(f"Unrecognised product id: {product_id}")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, float]:
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            raise RuntimeError(f"Invalid config at {path}: {exc}") from exc
        return {**DEFAULT_CONFIG, **data}
    _ensure_parent(path)
    path.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    return dict(DEFAULT_CONFIG)


def save_config(updates: Dict[str, float], path: Path = DEFAULT_CONFIG_PATH) -> None:
    cfg = load_config(path)
    cfg.update({k: v for k, v in updates.items() if v is not None})
    path.write_text(json.dumps(cfg, indent=2))


def _split_blocks(text: str) -> List[str]:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    marker = "Short-Line Summaries"
    idx = t.find(marker)
    if idx != -1:
        t = t[:idx]
    heads = [m.start() for m in re.finditer(r"(?m)^\s*\d+\.\s+\S+\s*\(", t)]
    if not heads:
        return [t.strip()] if t.strip() else []
    heads.append(len(t))
    blocks = []
    for i in range(len(heads) - 1):
        block = t[heads[i]:heads[i + 1]].strip()
        if block:
            blocks.append(block)
    return blocks


def _parse_score(text: str) -> float:
    patterns = [
        r"Overall\s*Score\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Signal\s*Score\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Confidence\s*Score\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Score\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.I)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                continue
    return 0.0


@dataclass
class FinderCandidate:
    rank: int
    block: str
    parsed: ParsedFinder
    score: float
    product_id: Optional[str] = None


def gather_candidates(text: str) -> List[FinderCandidate]:
    blocks = _split_blocks(text)
    candidates: List[FinderCandidate] = []
    for i, block in enumerate(blocks, start=1):
        try:
            parsed = parse_finder_text(block)
        except Exception as exc:
            logger.debug("Skipping block %s: %s", i, exc)
            continue
        score = _parse_score(block)
        product_id = _product_id(parsed.symbol)
        candidates.append(FinderCandidate(rank=i, block=block, parsed=parsed, score=score, product_id=product_id))
    return candidates


def _load_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        missing = [c for c in columns if c not in df.columns]
        for col in missing:
            df[col] = pd.NA
        return df
    return pd.DataFrame(columns=columns)


def _save_csv(df: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    _ensure_parent(path)
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[columns]
    df.to_csv(path, index=False)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _desired_position_usd(
    parsed: ParsedFinder,
    portfolio_usd: float,
    fixed_position_usd: Optional[float],
    default_pct: float,
) -> float:
    if fixed_position_usd and fixed_position_usd > 0:
        return float(fixed_position_usd)
    pct = parsed.pos_size_pct or default_pct
    pct = max(pct, 0.0)
    return portfolio_usd * (pct / 100.0)


def _compute_unrealized_pct(side: str, entry: float, price: float) -> float:
    if entry <= 0 or price <= 0:
        return 0.0
    if side.upper() == "LONG":
        return (price - entry) / entry * 100.0
    return (entry - price) / entry * 100.0


def _maybe_close_reason(
    side: str,
    price: float,
    take_profit: float,
    stop_loss: float,
    entry: float,
    expires_at: Optional[datetime],
    now: datetime,
) -> Optional[str]:
    if side.upper() == "LONG":
        if price >= take_profit > 0:
            return "take_profit"
        if price <= stop_loss < take_profit:
            return "stop_loss"
    else:
        if price <= take_profit < stop_loss:
            return "take_profit"
        if price >= stop_loss > 0:
            return "stop_loss"
    if expires_at and now >= expires_at:
        if entry <= 0:
            return "expired_breakeven"
        pct = _compute_unrealized_pct(side, entry, price)
        if abs(pct) <= EXPIRY_BREAKEVEN_PCT:
            return "expired_breakeven"
        return "expired_profit" if pct > 0 else "expired_loss"
    return None


def _build_closed_record(
    row: Dict[str, object],
    price: float,
    reason: str,
    now: datetime,
    mae: Optional[float] = None,
    mfe: Optional[float] = None,
    position_usd_override: Optional[float] = None,
) -> Dict[str, object]:
    entry = _safe_float(row.get("entry_price"), default=price)
    side = (row.get("position_side") or "LONG").upper()
    position_usd = (
        float(position_usd_override)
        if position_usd_override is not None and position_usd_override > 0
        else _safe_float(row.get("position_usd"), default=0.0)
    )
    pct = _compute_unrealized_pct(side, entry, price)
    pnl = position_usd * pct / 100.0
    opened_at = _parse_iso(str(row.get("opened_at") or ""))
    duration = (now - opened_at).total_seconds() if opened_at else 0.0
    leverage = row.get("leverage")
    net_size = 0.0
    if entry > 0:
        net_size = position_usd / entry
        if side == "SHORT":
            net_size = -abs(net_size)
    return {
        "closed_at": _isoformat(now),
        "product_id": row.get("product_id"),
        "position_side": side,
        "net_size": net_size,
        "leverage": leverage,
        "opened_at": row.get("opened_at"),
        "closure_reason": reason,
        "entry_price": entry,
        "exit_price": price,
        "profit_loss": round(pnl, 2),
        "profit_loss_pct": round(pct, 4),
        "mae": "" if mae is None else round(mae, 2),
        "mfe": "" if mfe is None else round(mfe, 2),
        "duration_seconds": int(duration),
    }


def _format_time_left(expires_value: object, now: datetime) -> str:
    expires = _parse_iso(str(expires_value or ""))
    if not expires:
        return "n/a"
    delta = expires - now
    seconds = int(math.ceil(delta.total_seconds()))
    if seconds <= 0:
        return "expired"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _close_and_update_rows(
    open_rows: List[Dict[str, object]],
    price_lookup: Callable[[str], float],
    now: datetime,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    updated: List[Dict[str, object]] = []
    closed: List[Dict[str, object]] = []
    for row in open_rows:
        product_id = str(row.get("product_id") or "")
        if not product_id:
            logger.warning("Skipping trade without product id: %s", row)
            continue
        try:
            price = float(price_lookup(product_id))
        except Exception as exc:
            logger.error("Unable to fetch price for %s: %s", product_id, exc)
            updated.append(row)
            continue

        entry = _safe_float(row.get("entry_price"), default=price)
        tp = _safe_float(row.get("take_profit"), default=entry)
        sl = _safe_float(row.get("stop_loss"), default=entry)
        expires_at = _parse_iso(str(row.get("expires_at") or ""))
        position_usd = _safe_float(row.get("position_usd"), default=0.0)
        partial_pct = _safe_float(row.get("partial_tp_pct"), default=0.0)
        partial_rr = _safe_float(row.get("partial_tp_rr"), default=0.0)
        partial_price = _safe_float(row.get("partial_tp_price"), default=0.0)
        partial_done = _truthy(row.get("partial_tp_done"))
        partial_move_sl = _truthy(row.get("partial_tp_move_sl"))

        if partial_pct > 0 and partial_price > 0 and not partial_done and position_usd > 0:
            hit_partial = False
            if str(row.get("position_side") or "LONG").upper() == "LONG":
                hit_partial = price >= partial_price
            else:
                hit_partial = price <= partial_price
            if hit_partial:
                close_usd = position_usd * (partial_pct / 100.0)
                close_usd = min(close_usd, position_usd)
                if close_usd > 0:
                    mae, mfe = _compute_mae_mfe(row, price, now, position_usd_override=close_usd)
                    closed.append(
                        _build_closed_record(
                            row,
                            price,
                            "partial_take",
                            now,
                            mae=mae,
                            mfe=mfe,
                            position_usd_override=close_usd,
                        )
                    )
                    remaining = max(0.0, position_usd - close_usd)
                    row["position_usd"] = round(remaining, 2)
                    row["partial_tp_done"] = True
                    row["partial_tp_rr"] = partial_rr
                    row["partial_tp_pct"] = partial_pct
                    row["partial_tp_price"] = partial_price
                    row["partial_tp_move_sl"] = partial_move_sl
                    if partial_move_sl and entry > 0:
                        row["stop_loss"] = entry
                    position_usd = remaining
                if position_usd <= 0:
                    continue
        sl = _safe_float(row.get("stop_loss"), default=entry)
        tp = _safe_float(row.get("take_profit"), default=entry)
        reason = _maybe_close_reason(
            side=str(row.get("position_side") or "LONG"),
            price=price,
            take_profit=tp,
            stop_loss=sl,
            entry=entry,
            expires_at=expires_at,
            now=now,
        )
        pct = _compute_unrealized_pct(str(row.get("position_side") or "LONG"), entry, price)
        pnl = _safe_float(row.get("position_usd"), 0.0) * pct / 100.0

        row["last_price"] = price
        row["last_price_at"] = _isoformat(now)
        row["unrealized_pct"] = round(pct, 4)
        row["unrealized_pnl"] = round(pnl, 2)

        if reason:
            mae, mfe = _compute_mae_mfe(row, price, now)
            closed.append(_build_closed_record(row, price, reason, now, mae=mae, mfe=mfe))
        else:
            updated.append(row)
    return updated, closed


def _select_balanced_top(candidates: List[FinderCandidate], total: int) -> List[FinderCandidate]:
    """Return up to ``total`` candidates prioritising 2 longs, 2 shorts, then best overall."""
    if total <= 0 or not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    picks: List[FinderCandidate] = []

    def _take_side(side: str, count: int) -> None:
        for cand in sorted_candidates:
            if cand in picks:
                continue
            if cand.parsed.side.upper() == side:
                picks.append(cand)
            if sum(1 for p in picks if p.parsed.side.upper() == side) >= count:
                break

    _take_side("LONG", min(2, total))
    _take_side("SHORT", min(2, total))

    for cand in sorted_candidates:
        if cand in picks:
            continue
        picks.append(cand)
        if len(picks) >= total:
            break

    return picks[:total]


def _filter_supported_candidates(
    candidates: List[FinderCandidate],
    excluded_perps: Optional[Set[str]] = None,
) -> List[FinderCandidate]:
    filtered: List[FinderCandidate] = []
    excluded_lookup = {p.upper() for p in excluded_perps} if excluded_perps is not None else _excluded_perps()
    for cand in candidates:
        product_id = cand.product_id or _product_id(cand.parsed.symbol)
        if not product_id:
            logger.warning("Skipping %s (cannot derive perp symbol).", cand.parsed.symbol)
            continue
        cand.product_id = product_id
        pid_upper = product_id.upper()
        if excluded_lookup and pid_upper in excluded_lookup:
            logger.info("Skipping %s (perp %s is excluded).", cand.parsed.symbol, product_id)
            continue
        if not _is_supported_product(product_id):
            logger.warning("Skipping %s (perp %s unsupported on Coinbase).", cand.parsed.symbol, product_id)
            continue
        filtered.append(cand)
    return filtered


def _select_candidates(
    candidates: List[FinderCandidate],
    symbols: Optional[Sequence[str]],
    picks: Optional[Sequence[int]],
    top: int,
    balanced_top: bool = False,
) -> List[FinderCandidate]:
    selected: List[FinderCandidate] = []
    seen_symbols: Set[str] = set()
    symbol_set = {s.strip().upper() for s in symbols or [] if s.strip()}
    index_set = {int(p) for p in picks or [] if int(p) > 0}

    def _add_candidate(cand: FinderCandidate) -> None:
        key = cand.parsed.symbol.upper()
        if key in seen_symbols or cand in selected:
            return
        seen_symbols.add(key)
        selected.append(cand)

    if symbol_set:
        for cand in candidates:
            if cand.parsed.symbol.upper() in symbol_set:
                _add_candidate(cand)

    if index_set:
        for cand in candidates:
            if cand.rank in index_set and cand not in selected:
                _add_candidate(cand)

    if top > 0:
        remaining_slots = max(0, top - len(selected))
        pool = [cand for cand in candidates if cand not in selected]
        if balanced_top and remaining_slots > 0:
            balanced = _select_balanced_top(pool, total=remaining_slots if top > len(selected) else top)
            for cand in balanced:
                _add_candidate(cand)
        elif remaining_slots > 0:
            scored = sorted(pool, key=lambda c: c.score, reverse=True)
            for cand in scored[:remaining_slots]:
                _add_candidate(cand)

    return selected


def _open_selected_trades(
    selected: List[FinderCandidate],
    *,
    portfolio_usd: float,
    leverage: float,
    expiry_hours: float,
    default_pct: float,
    tag: str,
    note: str,
    fixed_position_usd: Optional[float],
    partial_tp_rr: float,
    partial_tp_pct: float,
    partial_tp_move_sl: bool,
    dry_run: bool,
) -> None:
    if not selected:
        raise SystemExit("No finder candidates selected.")

    open_df = _load_csv(OPEN_CSV, OPEN_COLUMNS)

    for cand in selected:
        product_id = cand.product_id or _product_id(cand.parsed.symbol)
        if not product_id:
            logger.warning("Skipping %s (unsupported perp).", cand.parsed.symbol)
            continue
        if not _is_supported_product(product_id):
            logger.warning("Skipping %s (perp %s unsupported on Coinbase).", cand.parsed.symbol, product_id)
            continue
        trade_id = uuid.uuid4().hex[:10]
        position_usd = _desired_position_usd(
            parsed=cand.parsed,
            portfolio_usd=portfolio_usd,
            fixed_position_usd=fixed_position_usd,
            default_pct=default_pct,
        )
        partial_rr = float(partial_tp_rr or 0.0)
        partial_pct = float(partial_tp_pct or 0.0)
        partial_price = 0.0
        if partial_rr > 0 and partial_pct > 0:
            risk = abs(cand.parsed.entry - cand.parsed.stop)
            if risk > 0:
                if cand.parsed.side.upper() == "LONG":
                    partial_price = cand.parsed.entry + risk * partial_rr
                else:
                    partial_price = cand.parsed.entry - risk * partial_rr
        opened_at = datetime.now(tz=UTC)
        expires_at = opened_at + timedelta(hours=expiry_hours)
        row = {
            "trade_id": trade_id,
            "symbol": cand.parsed.symbol.upper(),
            "product_id": product_id,
            "position_side": cand.parsed.side.upper(),
            "entry_price": cand.parsed.entry,
            "stop_loss": cand.parsed.stop,
            "take_profit": cand.parsed.take_profit,
            "partial_tp_pct": round(partial_pct, 2) if partial_pct > 0 else "",
            "partial_tp_rr": round(partial_rr, 2) if partial_rr > 0 else "",
            "partial_tp_price": round(partial_price, 6) if partial_price > 0 else "",
            "partial_tp_done": False,
            "partial_tp_move_sl": bool(partial_tp_move_sl),
            "position_usd": round(position_usd, 2),
            "leverage": leverage,
            "opened_at": _isoformat(opened_at),
            "expires_at": _isoformat(expires_at),
            "status": "OPEN",
            "last_price": cand.parsed.entry,
            "last_price_at": _isoformat(opened_at),
            "unrealized_pnl": 0.0,
            "unrealized_pct": 0.0,
            "finder_score": cand.score,
            "finder_rank": cand.rank,
            "recommended_position_pct": cand.parsed.pos_size_pct,
            "tag": tag,
            "notes": note,
        }
        row_df = pd.DataFrame([row])[OPEN_COLUMNS]
        if open_df.empty:
            open_df = row_df
        else:
            open_df = pd.concat([open_df, row_df], ignore_index=True)
        logger.info("Opened paper trade %s (%s %s)", trade_id, cand.parsed.side, product_id)

    if not dry_run:
        _save_csv(open_df, OPEN_CSV, OPEN_COLUMNS)
    else:
        logger.info("Dry-run enabled; not writing open positions.")


def _price_provider(overrides: Dict[str, float]) -> Callable[[str], float]:
    cache: Dict[str, float] = {}
    exchange = None

    def _inner(product_id: str) -> float:
        pid = product_id.upper()
        if pid in overrides:
            cache[pid] = overrides[pid]
            return overrides[pid]
        if pid in cache:
            return cache[pid]
        nonlocal exchange
        if exchange is None:
            import ccxt  # imported lazily to avoid pytest/network overhead

            params = {"enableRateLimit": True}
            api_key = os.getenv("COINBASE_PERP_API_KEY") or os.getenv("API_KEY_PERPS")
            api_secret = os.getenv("COINBASE_PERP_API_SECRET") or os.getenv("API_SECRET_PERPS")
            if api_key and api_secret:
                params.update({"apiKey": api_key, "secret": api_secret})
            exchange = ccxt.coinbaseadvanced(params)
            exchange.load_markets()
        symbol = _ccxt_symbol(pid)
        ticker = exchange.fetch_ticker(symbol)
        last = ticker.get("last") or ticker.get("close")
        if last is None:
            raise RuntimeError(f"No ticker price for {pid}")
        cache[pid] = float(last)
        return cache[pid]

    return _inner


def handle_init(args: argparse.Namespace) -> None:
    updates = {}
    if args.initial_capital is not None:
        updates["initial_capital"] = args.initial_capital
    if args.default_leverage is not None:
        updates["default_leverage"] = args.default_leverage
    if args.default_expiry_hours is not None:
        updates["default_expiry_hours"] = args.default_expiry_hours
    if args.default_position_pct is not None:
        updates["default_position_pct"] = args.default_position_pct
    save_config(updates)
    logger.info("Updated config: %s", json.dumps(load_config(), indent=2))


def handle_candidates(args: argparse.Namespace) -> None:
    text = Path(args.finder_output).read_text(encoding="utf-8")
    candidates = gather_candidates(text)
    if not candidates:
        logger.warning("No candidates found in %s", args.finder_output)
        return
    rows = []
    for cand in candidates:
        rows.append(
            {
                "rank": cand.rank,
                "symbol": cand.parsed.symbol,
                "side": cand.parsed.side,
                "score": cand.score,
                "entry": cand.parsed.entry,
                "take_profit": cand.parsed.take_profit,
                "stop_loss": cand.parsed.stop,
                "position_pct": cand.parsed.pos_size_pct,
            }
        )
    df = pd.DataFrame(rows)
    stype = args.sort_by
    if stype == "score":
        df = df.sort_values("score", ascending=False)
    elif stype == "symbol":
        df = df.sort_values("symbol")
    print(df.to_string(index=False))


def handle_open(args: argparse.Namespace) -> None:
    cfg = load_config()
    text = Path(args.finder_output).read_text(encoding="utf-8")
    candidates = gather_candidates(text)
    if not candidates:
        raise SystemExit("No finder candidates detected.")

    candidates = _filter_supported_candidates(candidates)
    if not candidates:
        raise SystemExit("No supported perps found in finder output.")

    symbols = []
    if args.symbols:
        for chunk in args.symbols:
            symbols.extend([part for part in chunk.split(",") if part.strip()])
    selected = _select_candidates(
        candidates=candidates,
        symbols=symbols,
        picks=args.pick,
        top=args.top,
        balanced_top=args.balanced_top,
    )
    if not selected:
        raise SystemExit("No candidates selected (use --symbols/--pick/--top).")

    portfolio_usd = args.portfolio_usd or cfg["initial_capital"]
    leverage = args.leverage or cfg["default_leverage"]
    expiry_hours = args.expiry_hours or cfg["default_expiry_hours"]
    default_pct = args.default_position_pct or cfg["default_position_pct"]
    note = args.notes or ""
    tag = args.tag or ""

    _open_selected_trades(
        selected,
        portfolio_usd=portfolio_usd,
        leverage=leverage,
        expiry_hours=expiry_hours,
        default_pct=default_pct,
        tag=tag,
        note=note,
        fixed_position_usd=args.fixed_position_usd,
        partial_tp_rr=float(args.partial_tp_rr or 0.0),
        partial_tp_pct=float(args.partial_tp_pct or 0.0),
        partial_tp_move_sl=bool(args.partial_tp_move_sl),
        dry_run=args.dry_run,
    )


def handle_update(args: argparse.Namespace) -> None:
    open_df = _load_csv(OPEN_CSV, OPEN_COLUMNS)
    if open_df.empty:
        logger.info("No open paper trades.")
        return

    overrides: Dict[str, float] = {}
    for item in args.override or []:
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        key = key.strip().upper()
        try:
            price = float(raw_value)
        except Exception:
            logger.warning("Invalid override price %s", raw_value)
            continue
        if "-" not in key:
            pid = _product_id(key)
        else:
            pid = key
        if not pid:
            logger.warning("Unable to resolve override symbol %s", key)
            continue
        overrides[pid] = price

    price_lookup = _price_provider(overrides)
    open_rows = open_df.to_dict("records")
    now = datetime.now(tz=UTC)
    updated_rows, closed_rows = _close_and_update_rows(open_rows, price_lookup, now)
    _save_csv(pd.DataFrame(updated_rows), OPEN_CSV, OPEN_COLUMNS)

    if closed_rows:
        closed_df = _load_csv(CLOSED_CSV, CLOSED_COLUMNS)
        rows_df = pd.DataFrame(closed_rows)[CLOSED_COLUMNS]
        if closed_df.empty:
            closed_df = rows_df
        else:
            closed_df = pd.concat([closed_df, rows_df], ignore_index=True)
        _save_csv(closed_df, CLOSED_CSV, CLOSED_COLUMNS)
        logger.info("Closed %s trades.", len(closed_rows))
        for row in closed_rows:
            logger.info(
                "%s %s %s P/L %+.2f (%+.4f%%) via %s",
                row["product_id"],
                row["position_side"],
                row.get("closed_at"),
                row["profit_loss"],
                row["profit_loss_pct"],
                row["closure_reason"],
            )
    else:
        logger.info("Updated %s trades; none hit targets/stops yet.", len(updated_rows))

    open_pnl_total = sum(_safe_float(row.get("unrealized_pnl"), 0.0) for row in updated_rows)
    logger.info(
        "Open paper trades: %d | unrealized P/L %+.2f",
        len(updated_rows),
        open_pnl_total,
    )
    if updated_rows:
        for row in sorted(updated_rows, key=lambda r: str(r.get("product_id", ""))):
            logger.info(
                "  %s %-5s last %.4f | P/L %+.2f (%+.4f%%) | expires in %s",
                row.get("product_id"),
                (row.get("position_side") or "").upper(),
                _safe_float(row.get("last_price"), 0.0),
                _safe_float(row.get("unrealized_pnl"), 0.0),
                _safe_float(row.get("unrealized_pct"), 0.0),
                _format_time_left(row.get("expires_at"), now),
            )


def handle_close(args: argparse.Namespace) -> None:
    open_df = _load_csv(OPEN_CSV, OPEN_COLUMNS)
    if open_df.empty:
        logger.info("No open paper trades to close.")
        return

    def _split(values: Optional[Sequence[str]]) -> List[str]:
        items: List[str] = []
        for chunk in values or []:
            for part in chunk.split(","):
                val = part.strip()
                if val:
                    items.append(val)
        return items

    trade_ids = {tid.strip() for tid in _split(args.trade_id) if tid.strip()}
    product_ids = {pid.strip().upper() for pid in _split(args.product_id) if pid.strip()}
    symbols = {sym.strip().upper() for sym in _split(args.symbol) if sym.strip()}
    if not (trade_ids or product_ids or symbols or args.all):
        raise SystemExit("Provide --trade-id/--product-id/--symbol or --all to choose trades to close.")

    open_rows = open_df.to_dict("records")
    selected: List[Dict[str, object]] = []
    remaining: List[Dict[str, object]] = []
    for row in open_rows:
        tid = str(row.get("trade_id") or "")
        pid = str(row.get("product_id") or "").upper()
        sym = str(row.get("symbol") or "").upper()
        match = False
        if trade_ids and tid in trade_ids:
            match = True
        if product_ids and pid in product_ids:
            match = True
        if symbols and sym in symbols:
            match = True
        if args.all:
            match = True
        (selected if match else remaining).append(row)

    if not selected:
        raise SystemExit("No matching open trades found for the provided filters.")

    exit_price_override = args.price
    reason = args.reason or "manual_close"
    now = datetime.now(tz=UTC)
    closed_rows: List[Dict[str, object]] = []
    for row in selected:
        exit_price = exit_price_override
        if exit_price is None:
            exit_price = _safe_float(row.get("last_price"), _safe_float(row.get("entry_price"), 0.0))
        mae, mfe = _compute_mae_mfe(row, float(exit_price), now)
        closed_rows.append(_build_closed_record(row, float(exit_price), reason, now, mae=mae, mfe=mfe))

    if args.dry_run:
        logger.info("Dry run: would close %s trades.", len(selected))
        for row in closed_rows:
            logger.info(
                "  %s %s exit %.4f P/L %+.2f (%+.4f%%)",
                row["product_id"],
                row["position_side"],
                row["exit_price"],
                row["profit_loss"],
                row["profit_loss_pct"],
            )
        return

    _save_csv(pd.DataFrame(remaining), OPEN_CSV, OPEN_COLUMNS)
    closed_df = _load_csv(CLOSED_CSV, CLOSED_COLUMNS)
    rows_df = pd.DataFrame(closed_rows)[CLOSED_COLUMNS]
    if closed_df.empty:
        closed_df = rows_df
    else:
        closed_df = pd.concat([closed_df, rows_df], ignore_index=True)
    _save_csv(closed_df, CLOSED_CSV, CLOSED_COLUMNS)

    logger.info("Closed %s trades via %s.", len(closed_rows), reason)
    for row in closed_rows:
        logger.info(
            "  %s %s exit %.4f P/L %+.2f (%+.4f%%)",
            row["product_id"],
            row["position_side"],
            row["exit_price"],
            row["profit_loss"],
            row["profit_loss_pct"],
        )


def handle_summary(args: argparse.Namespace) -> None:
    cfg = load_config()
    open_df = _load_csv(OPEN_CSV, OPEN_COLUMNS)
    closed_df = _load_csv(CLOSED_CSV, CLOSED_COLUMNS)
    realized = float(closed_df["profit_loss"].sum()) if "profit_loss" in closed_df else 0.0
    unrealized = float(open_df["unrealized_pnl"].sum()) if "unrealized_pnl" in open_df else 0.0
    equity = cfg["initial_capital"] + realized + unrealized
    print(f"Initial capital : {cfg['initial_capital']:.2f}")
    print(f"Realized P/L    : {realized:+.2f}")
    print(f"Unrealized P/L  : {unrealized:+.2f}")
    print(f"Equity (paper)  : {equity:.2f}")
    print(f"Open trades     : {len(open_df)}")
    print(f"Closed trades   : {len(closed_df)}")


def handle_open_single(args: argparse.Namespace) -> None:
    cfg = load_config()
    text = Path(args.finder_output).read_text(encoding="utf-8")
    candidates = gather_candidates(text)
    if not candidates:
        raise SystemExit("No finder candidates detected.")
    index = max(1, int(args.block_index or 1))
    if index > len(candidates):
        raise SystemExit(f"Block index {index} exceeds total candidates ({len(candidates)}).")

    selected = [candidates[index - 1]]
    cand = selected[0]
    product_id = cand.product_id or _product_id(cand.parsed.symbol)
    if not product_id or not _is_supported_product(product_id):
        raise SystemExit(f"{cand.parsed.symbol} perp is not supported on Coinbase.")

    portfolio_usd = args.portfolio_usd or cfg["initial_capital"]
    leverage = args.leverage or cfg["default_leverage"]
    expiry_hours = args.expiry_hours or cfg["default_expiry_hours"]
    default_pct = args.default_position_pct or cfg["default_position_pct"]
    tag = args.tag or ""
    note = args.notes or ""

    _open_selected_trades(
        selected,
        portfolio_usd=portfolio_usd,
        leverage=leverage,
        expiry_hours=expiry_hours,
        default_pct=default_pct,
        tag=tag,
        note=note,
        fixed_position_usd=args.fixed_position_usd,
        partial_tp_rr=float(args.partial_tp_rr or 0.0),
        partial_tp_pct=float(args.partial_tp_pct or 0.0),
        partial_tp_move_sl=bool(args.partial_tp_move_sl),
        dry_run=args.dry_run,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Paper trading helper for short_term_crypto_finder output.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialise/update default config.")
    p_init.add_argument("--initial-capital", type=float)
    p_init.add_argument("--default-leverage", type=float)
    p_init.add_argument("--default-expiry-hours", type=int)
    p_init.add_argument("--default-position-pct", type=float)
    p_init.set_defaults(func=handle_init)

    p_candidates = sub.add_parser("candidates", help="List parsed finder candidates.")
    p_candidates.add_argument("--finder-output", required=True, help="Path to finder text output.")
    p_candidates.add_argument("--sort-by", choices=("rank", "score", "symbol"), default="rank")
    p_candidates.set_defaults(func=handle_candidates)

    p_open = sub.add_parser("open", help="Open paper trades from finder output.")
    p_open.add_argument("--finder-output", required=True, help="Path to finder text file.")
    p_open.add_argument("--symbols", nargs="*", help="Symbols to pick (comma-separated chunks allowed).")
    p_open.add_argument("--pick", type=int, nargs="*", help="1-based finder ranks to pick.")
    p_open.add_argument("--top", type=int, default=0, help="Automatically pick the top-N candidates by score.")
    p_open.add_argument(
        "--balanced-top",
        action="store_true",
        help="When using --top, pick 2 longs, 2 shorts, and the next best remaining (mirrors add_top5).",
    )
    p_open.add_argument("--portfolio-usd", type=float, help="Portfolio size used for sizing (defaults to initial capital).")
    p_open.add_argument("--fixed-position-usd", type=float, help="Override absolute USD per trade.")
    p_open.add_argument("--default-position-pct", type=float, help="Fallback %% of portfolio when finder lacks a recommendation.")
    p_open.add_argument("--leverage", type=float, help="Stored leverage hint (for display only).")
    p_open.add_argument("--expiry-hours", type=float, help="Expiry horizon in hours (default config).")
    p_open.add_argument("--partial-tp-rr", type=float, default=0.0, help="Partial take-profit RR (default: 0 disabled).")
    p_open.add_argument("--partial-tp-pct", type=float, default=0.0, help="Percent to close at partial TP (default: 0).")
    p_open.add_argument("--partial-tp-move-sl", action="store_true", help="Move SL to entry after partial TP.")
    p_open.add_argument("--tag", help="Optional tag stored with the trade.")
    p_open.add_argument("--notes", help="Optional freeform note.")
    p_open.add_argument("--dry-run", action="store_true", help="Parse and display without writing CSVs.")
    p_open.set_defaults(func=handle_open)

    p_single = sub.add_parser(
        "open-single",
        help="Open a single finder block (mirrors running add_position_from_finder.py for one trade).",
    )
    p_single.add_argument("--finder-output", required=True, help="Path to finder block text.")
    p_single.add_argument(
        "--block-index",
        type=int,
        default=1,
        help="When finder-output contains multiple blocks, pick this 1-based index (default 1).",
    )
    p_single.add_argument("--portfolio-usd", type=float)
    p_single.add_argument("--fixed-position-usd", type=float)
    p_single.add_argument("--default-position-pct", type=float)
    p_single.add_argument("--leverage", type=float)
    p_single.add_argument("--expiry-hours", type=float)
    p_single.add_argument("--partial-tp-rr", type=float, default=0.0)
    p_single.add_argument("--partial-tp-pct", type=float, default=0.0)
    p_single.add_argument("--partial-tp-move-sl", action="store_true")
    p_single.add_argument("--tag")
    p_single.add_argument("--notes")
    p_single.add_argument("--dry-run", action="store_true")
    p_single.set_defaults(func=handle_open_single)

    p_update = sub.add_parser("update", help="Refresh prices and close trades that hit TP/SL/expiry.")
    p_update.add_argument(
        "--override",
        action="append",
        help="Manual price overrides product=price (symbol or product id). Repeat for multiple symbols.",
    )
    p_update.set_defaults(func=handle_update)

    p_summary = sub.add_parser("summary", help="Display equity snapshot for paper trades.")
    p_summary.set_defaults(func=handle_summary)

    p_close = sub.add_parser("close", help="Manually close open paper trades.")
    p_close.add_argument("--trade-id", action="append", help="Trade IDs to close (comma-separated chunks allowed).")
    p_close.add_argument("--product-id", action="append", help="Product IDs to close (comma-separated).")
    p_close.add_argument("--symbol", action="append", help="Symbols to close (comma-separated).")
    p_close.add_argument("--all", action="store_true", help="Close all open trades.")
    p_close.add_argument("--price", type=float, help="Exit price to apply (defaults to last mark or entry).")
    p_close.add_argument("--reason", default="manual_close", help="Closure reason stored in the CSV.")
    p_close.add_argument("--dry-run", action="store_true", help="Preview the closure without modifying CSVs.")
    p_close.set_defaults(func=handle_close)

    args = parser.parse_args(argv)
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    args.func(args)


if __name__ == "__main__":
    main()
