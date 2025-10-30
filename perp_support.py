from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

from coinbaseservice import CoinbaseService

_SUPPORT_CACHE: Dict[str, bool] = {}
THOUSAND_UNIT_BASES = {"SHIB", "BONK", "PEPE", "FLOKI"}
_KNOWN_PERPS = set()


def _load_known_perps() -> set[str]:
    path = Path(__file__).resolve().with_name("derived_perps_intx.txt")
    if not path.exists():
        return set()
    entries = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip().upper()
            if value:
                entries.add(value)
    return entries


if not _KNOWN_PERPS:
    try:
        _KNOWN_PERPS = _load_known_perps()
    except Exception:
        _KNOWN_PERPS = set()


def _is_not_found_error(exc: Exception) -> bool:
    message = str(exc).upper()
    return "NOT_FOUND" in message or "COULD NOT BE FOUND" in message

def _is_thousand_perp(product_id: str) -> bool:
    match = re.match(r"^1000([A-Z0-9]+)-PERP-INTX$", product_id)
    return bool(match and match.group(1) in THOUSAND_UNIT_BASES)


def is_perp_supported(
    product_id: str,
    cb_service: Optional[CoinbaseService],
    logger: Optional[object] = None,
) -> bool:
    """Return True when Coinbase INTX perps list supports the product.

    When ``cb_service`` is None (no credentials available) support cannot be
    verified, so the function returns True to avoid false negatives but logs a
    warning if a logger is provided.
    """
    if not product_id:
        return False

    product_id = product_id.upper()

    if product_id in _KNOWN_PERPS or _is_thousand_perp(product_id):
        _SUPPORT_CACHE[product_id] = True
        return True

    if product_id in _SUPPORT_CACHE:
        return _SUPPORT_CACHE[product_id]

    if cb_service is None:
        if logger is not None:
            try:
                logger.warning(
                    "Skipping support verification for %s (missing INTX credentials).",
                    product_id,
                )
            except Exception:
                pass
        _SUPPORT_CACHE[product_id] = True
        return True


def canonical_perp_symbol(symbol: str) -> str:
    """Return canonical perp base (inject 1000- prefixes where applicable)."""
    s = (symbol or "").upper().strip()
    if not s:
        return ""
    if s.startswith("1000") and s[4:] in THOUSAND_UNIT_BASES:
        return s
    base_match = re.match(r"^1000([A-Z0-9]+)$", s)
    if base_match and base_match.group(1) in THOUSAND_UNIT_BASES:
        return s
    if s in THOUSAND_UNIT_BASES:
        return f"1000{s}"
    return s


def perp_price_multiplier(symbol: str) -> float:
    """Return the unit multiplier between spot quotes and perp contract."""
    canonical = canonical_perp_symbol(symbol)
    if canonical.startswith("1000"):
        base = canonical[4:]
        if base in THOUSAND_UNIT_BASES:
            return 1000.0
    return 1.0

    try:
        cb_service.client.get_market_trades(product_id=product_id, limit=1)
        _SUPPORT_CACHE[product_id] = True
        return True
    except Exception as exc:  # pragma: no cover - network conditions may vary
        if _is_not_found_error(exc):
            if product_id in _KNOWN_PERPS or _is_thousand_perp(product_id):
                _SUPPORT_CACHE[product_id] = True
                return True
            if logger is not None:
                try:
                    logger.info("Perp product %s is not supported on Coinbase.", product_id)
                except Exception:
                    pass
            _SUPPORT_CACHE[product_id] = False
            return False
        if logger is not None:
            try:
                logger.warning("Error checking support for %s: %s", product_id, exc)
            except Exception:
                pass
        _SUPPORT_CACHE[product_id] = True
        return True
