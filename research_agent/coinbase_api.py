from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


COINBASE_EXCHANGE_BASE_URL = os.getenv("COINBASE_EXCHANGE_BASE_URL", "https://api.exchange.coinbase.com").rstrip("/")
DEFAULT_PUBLIC_CACHE_TTL_SEC = int(os.getenv("COINBASE_PUBLIC_CACHE_TTL_SEC", "300") or "300")
DEFAULT_PUBLIC_TIMEOUT_SEC = float(os.getenv("COINBASE_PUBLIC_TIMEOUT_SEC", "10") or "10")
DEFAULT_PUBLIC_CACHE_DIR = Path(os.getenv("COINBASE_PUBLIC_CACHE_DIR", "cache/coinbase_public"))


def _json_dumps_canonical(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


def _hash_key(parts: Dict[str, Any]) -> str:
    payload = _json_dumps_canonical(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_cache(path: Path, ttl_sec: int) -> Optional[Tuple[float, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    ts = payload.get("ts")
    if not isinstance(ts, (int, float)):
        return None
    ts_val = float(ts)
    if ttl_sec > 0 and (time.time() - ts_val) > ttl_sec:
        return None
    return ts_val, payload.get("data")


def _write_cache(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ts": time.time(), "data": data}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(_json_dumps_canonical(payload), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        return


_MEM_CACHE: Dict[str, Tuple[float, Any]] = {}


def cached_get_json(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    ttl_sec: int = DEFAULT_PUBLIC_CACHE_TTL_SEC,
    timeout_sec: float = DEFAULT_PUBLIC_TIMEOUT_SEC,
) -> Any:
    """GET a JSON endpoint with a simple TTL cache (memory + disk)."""

    try:
        import requests  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("Missing dependency: requests (pip install requests)") from exc

    key = _hash_key({"url": url, "params": params or {}})
    now = time.time()
    if ttl_sec > 0:
        entry = _MEM_CACHE.get(key)
        if entry is not None:
            ts, data = entry
            if (now - ts) <= ttl_sec:
                return data

        cache_path = DEFAULT_PUBLIC_CACHE_DIR / f"{key}.json"
        cached = _read_cache(cache_path, ttl_sec)
        if cached is not None:
            cached_ts, cached_data = cached
            _MEM_CACHE[key] = (cached_ts, cached_data)
            return cached_data
    else:
        cache_path = None

    req_headers = {
        "Accept": "application/json",
        "User-Agent": os.getenv("COINBASE_PUBLIC_USER_AGENT", "crypto-finance/coinbase-public-client"),
    }
    if headers:
        req_headers.update(headers)

    resp = requests.get(url, params=params, headers=req_headers, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()

    if ttl_sec > 0 and cache_path is not None:
        _MEM_CACHE[key] = (now, data)
        _write_cache(cache_path, data)
    return data


def get_coinbase_candles(
    product_id: str,
    *,
    granularity: int = 86400,
    start: Optional[str] = None,
    end: Optional[str] = None,
    ttl_sec: int = DEFAULT_PUBLIC_CACHE_TTL_SEC,
) -> List[list]:
    """Fetch Coinbase Exchange candles.

    Returns the raw Coinbase response list: [time, low, high, open, close, volume].
    """

    pid = (product_id or "").strip().upper()
    if not pid:
        raise ValueError("product_id is required")
    params: Dict[str, Any] = {"granularity": int(granularity)}
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    url = f"{COINBASE_EXCHANGE_BASE_URL}/products/{pid}/candles"
    data = cached_get_json(url, params=params, ttl_sec=ttl_sec)
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected candles payload for {pid}: {type(data).__name__}")


def get_coinbase_stats(product_id: str, *, ttl_sec: int = DEFAULT_PUBLIC_CACHE_TTL_SEC) -> Dict[str, Any]:
    """Fetch Coinbase Exchange 24h stats for a product."""

    pid = (product_id or "").strip().upper()
    if not pid:
        raise ValueError("product_id is required")
    url = f"{COINBASE_EXCHANGE_BASE_URL}/products/{pid}/stats"
    data = cached_get_json(url, ttl_sec=ttl_sec)
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unexpected stats payload for {pid}: {type(data).__name__}")


def get_coinbase_products(*, ttl_sec: int = DEFAULT_PUBLIC_CACHE_TTL_SEC) -> List[Dict[str, Any]]:
    """Fetch Coinbase Exchange product list (public)."""

    url = f"{COINBASE_EXCHANGE_BASE_URL}/products"
    data = cached_get_json(url, ttl_sec=ttl_sec)
    if isinstance(data, list):
        products: List[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                products.append(item)
        return products
    raise ValueError(f"Unexpected products payload: {type(data).__name__}")


def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Candle:
    time: int
    low: float
    high: float
    open: float
    close: float
    volume: float


def normalize_candles(raw_candles: List[list]) -> List[Candle]:
    """Normalize raw Coinbase candle rows into Candle dataclasses, sorted ascending by time."""

    candles: List[Candle] = []
    for row in raw_candles or []:
        if not isinstance(row, list) or len(row) < 6:
            continue
        ts = coerce_float(row[0])
        low = coerce_float(row[1])
        high = coerce_float(row[2])
        opn = coerce_float(row[3])
        close = coerce_float(row[4])
        vol = coerce_float(row[5])
        if ts is None or low is None or high is None or opn is None or close is None or vol is None:
            continue
        candles.append(
            Candle(
                time=int(ts),
                low=float(low),
                high=float(high),
                open=float(opn),
                close=float(close),
                volume=float(vol),
            )
        )
    candles.sort(key=lambda c: c.time)
    return candles
