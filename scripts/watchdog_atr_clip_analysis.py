#!/usr/bin/env python3
"""
Analyze ATR clip ratio vs outcomes for watchdog closed trades.

Buckets trades by ATR_bps / cap_bps and summarizes win rate + PnL.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import ccxt  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("ccxt is required. Install with `pip install -r requirements.txt`.") from exc


LOGGER = logging.getLogger("watchdog_atr_clip_analysis")


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def _parse_ts(text: str) -> datetime:
    text = (text or "").strip()
    if not text:
        raise ValueError("Missing timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _ccxt_symbol(product_id: str) -> str:
    product_id = (product_id or "").strip().upper()
    if not product_id:
        raise ValueError("Missing product id")
    if product_id.endswith("-PERP-INTX"):
        base = product_id.split("-")[0]
        return f"{base}/USDC:USDC"
    if "/" in product_id:
        return product_id
    if "-" in product_id:
        base, quote = product_id.split("-", 1)
        return f"{base}/{quote}"
    raise ValueError(f"Unrecognised product id: {product_id}")


def _dynamic_atr_bps(price: float) -> float:
    if price >= 20000:
        return 325.0
    if price >= 2000:
        return 350.0
    if price >= 200:
        return 400.0
    return 450.0


def _effective_atr_cap(price: float, max_atr_usd: Optional[float], max_atr_bps: Optional[float]) -> Optional[float]:
    caps: List[float] = []
    if max_atr_usd and max_atr_usd > 0:
        caps.append(float(max_atr_usd))
    if price > 0:
        tier_bps = _dynamic_atr_bps(price)
        eff_bps = tier_bps
        if max_atr_bps and max_atr_bps > 0:
            eff_bps = min(float(max_atr_bps), tier_bps)
        caps.append(price * eff_bps / 10000.0)
    return min(caps) if caps else None


def calculate_atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
    """ATR using Wilder smoothing, matching long_term_crypto_finder._calculate_atr."""
    if period <= 0 or high.size < period + 1:
        return 0.0
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    tr = np.r_[0.0, np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])])]
    atr = np.empty_like(tr)
    atr[:period] = np.nan
    atr[period] = tr[1 : period + 1].sum()
    for idx in range(period + 1, len(tr)):
        atr[idx] = atr[idx - 1] - atr[idx - 1] / period + tr[idx]
    val = float(np.nan_to_num(atr[-1] / period, nan=0.0, posinf=0.0, neginf=0.0)) if np.isfinite(atr[-1]) else 0.0
    return max(0.0, val)


@dataclass(frozen=True)
class AnalysisConfig:
    exchange_id: str
    timeframe: str
    atr_period: int
    lookback_days: int
    timeout_ms: int
    retries: int
    retry_base_delay_s: float
    max_atr_usd: Optional[float]
    max_atr_bps: Optional[float]


class CcxtOhlcvClient:
    def __init__(self, cfg: AnalysisConfig) -> None:
        self.cfg = cfg
        self.exchange = self._build_exchange(cfg.exchange_id)
        self.exchange.load_markets()

    def _build_exchange(self, exchange_id: str):
        params: Dict[str, object] = {"enableRateLimit": True, "timeout": int(self.cfg.timeout_ms)}
        api_key = os.getenv("COINBASE_PERP_API_KEY") or os.getenv("API_KEY_PERPS") or ""
        api_secret = os.getenv("COINBASE_PERP_API_SECRET") or os.getenv("API_SECRET_PERPS") or ""
        if api_key and api_secret:
            params.update({"apiKey": api_key, "secret": api_secret})
        return getattr(ccxt, exchange_id)(params)

    def fetch_ohlcv_range(self, symbol: str, start: datetime, end: datetime) -> List[Dict[str, object]]:
        timeframe = self.cfg.timeframe
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        out: List[Dict[str, object]] = []
        while since <= end_ms:
            data = self._fetch_with_retries(symbol, timeframe, since, limit=200)
            if not data:
                break
            for row in data:
                ts_ms = int(row[0])
                ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
                out.append(
                    {
                        "timestamp": ts,
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                    }
                )
            last_ms = int(data[-1][0])
            if last_ms <= since:
                break
            since = last_ms + 1
        # Dedup / sort just in case.
        out.sort(key=lambda r: r["timestamp"])
        return out

    def _fetch_with_retries(self, symbol: str, timeframe: str, since: int, limit: int) -> List[List[object]]:
        for attempt in range(1, self.cfg.retries + 1):
            try:
                return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            except Exception as exc:
                if attempt >= self.cfg.retries:
                    raise
                sleep_for = self.cfg.retry_base_delay_s * (2 ** (attempt - 1))
                LOGGER.warning("Retry %s/%s for %s (%s): %s", attempt, self.cfg.retries, symbol, timeframe, exc)
                time.sleep(sleep_for)
        return []


def _bucket_ratio(ratio: Optional[float]) -> str:
    if ratio is None:
        return "n/a"
    if ratio <= 1.0:
        return "<=1.0"
    if ratio <= 1.25:
        return "1.0-1.25"
    if ratio <= 1.5:
        return "1.25-1.5"
    if ratio <= 2.0:
        return "1.5-2.0"
    return ">2.0"


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals))


def _median(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.median(vals))


def analyze(path: Path, cfg: AnalysisConfig, out_path: Optional[Path]) -> None:
    rows = _load_rows(path)
    if not rows:
        raise SystemExit(f"No rows found in {path}")

    client = CcxtOhlcvClient(cfg)

    symbols: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        product = row.get("product_id", "")
        if not product:
            continue
        symbols.setdefault(product, []).append(row)

    history: Dict[str, List[Dict[str, object]]] = {}
    for product, items in symbols.items():
        try:
            symbol = _ccxt_symbol(product)
        except ValueError:
            continue
        entry_times = []
        for row in items:
            try:
                entry_times.append(_parse_ts(row.get("opened_at", "")))
            except Exception:
                continue
        if not entry_times:
            continue
        start = min(entry_times) - timedelta(days=cfg.lookback_days)
        end = max(entry_times) + timedelta(days=1)
        history[product] = client.fetch_ohlcv_range(symbol, start, end)

    enriched: List[Dict[str, object]] = []
    for row in rows:
        product = row.get("product_id", "")
        opened_at_raw = row.get("opened_at", "")
        try:
            opened_at = _parse_ts(opened_at_raw)
        except Exception:
            continue
        try:
            entry_price = float(row.get("entry_price") or 0.0)
        except Exception:
            entry_price = 0.0
        if entry_price <= 0:
            continue
        candles = history.get(product, [])
        if not candles:
            continue
        usable = [c for c in candles if c["timestamp"] <= opened_at]
        if len(usable) < cfg.atr_period + 1:
            continue
        highs = np.array([c["high"] for c in usable], dtype=float)
        lows = np.array([c["low"] for c in usable], dtype=float)
        closes = np.array([c["close"] for c in usable], dtype=float)
        atr = calculate_atr_wilder(highs, lows, closes, cfg.atr_period)

        cap_usd = _effective_atr_cap(entry_price, cfg.max_atr_usd, cfg.max_atr_bps)
        if cap_usd and cap_usd > 0:
            cap_bps = cap_usd / entry_price * 10_000
        else:
            cap_bps = None
        atr_bps = atr / entry_price * 10_000
        ratio = (atr_bps / cap_bps) if (cap_bps and cap_bps > 0) else None
        bucket = _bucket_ratio(ratio)

        try:
            pnl_pct = float(row.get("profit_loss_pct") or 0.0)
        except Exception:
            pnl_pct = 0.0
        try:
            mae = float(row.get("mae")) if row.get("mae") not in (None, "") else None
        except Exception:
            mae = None
        try:
            mfe = float(row.get("mfe")) if row.get("mfe") not in (None, "") else None
        except Exception:
            mfe = None

        enriched.append(
            {
                **row,
                "opened_at": opened_at_raw,
                "atr": atr,
                "atr_bps": atr_bps,
                "cap_bps": cap_bps,
                "ratio": ratio,
                "bucket": bucket,
                "profit_loss_pct": pnl_pct,
                "mae": mae,
                "mfe": mfe,
            }
        )

    if not enriched:
        raise SystemExit("No trades could be analyzed (missing candles or timestamps).")

    buckets = {}
    for row in enriched:
        buckets.setdefault(row["bucket"], []).append(row)

    print("ATR cap ratio buckets (ATR_bps / cap_bps):")
    header = f"{'bucket':<10} {'n':>4} {'win%':>6} {'avg%':>7} {'med%':>7} {'avg_mae':>8} {'avg_mfe':>8} {'avg_ratio':>10}"
    print(header)
    print("-" * len(header))
    for bucket in sorted(buckets.keys()):
        group = buckets[bucket]
        n = len(group)
        wins = sum(1 for g in group if g["profit_loss_pct"] > 0)
        win_rate = (wins / n * 100.0) if n else 0.0
        avg_pct = _mean(g["profit_loss_pct"] for g in group)
        med_pct = _median(g["profit_loss_pct"] for g in group)
        avg_mae = _mean(g["mae"] for g in group if g["mae"] is not None)
        avg_mfe = _mean(g["mfe"] for g in group if g["mfe"] is not None)
        avg_ratio = _mean(g["ratio"] for g in group if g["ratio"] is not None)
        print(
            f"{bucket:<10} {n:>4} {win_rate:>6.1f} "
            f"{(avg_pct if avg_pct is not None else 0.0):>7.2f} "
            f"{(med_pct if med_pct is not None else 0.0):>7.2f} "
            f"{(avg_mae if avg_mae is not None else 0.0):>8.2f} "
            f"{(avg_mfe if avg_mfe is not None else 0.0):>8.2f} "
            f"{(avg_ratio if avg_ratio is not None else 0.0):>10.2f}"
        )

    if out_path:
        fieldnames = list(enriched[0].keys())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enriched)
        print(f"\nWrote per-trade detail to {out_path}")


def main() -> None:
    _load_dotenv_if_available()
    parser = argparse.ArgumentParser(description="Analyze ATR clip ratio buckets vs outcomes.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("trade_logs/watchdog_closed_positions.csv"),
        help="Path to watchdog closed-positions CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV output for per-trade detail.",
    )
    parser.add_argument(
        "--exchange",
        default="coinbaseadvanced",
        help="CCXT exchange id for OHLCV (default: coinbaseadvanced).",
    )
    parser.add_argument(
        "--atr-period",
        type=int,
        default=7,
        help="ATR period in days (default: 7).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="Daily lookback window per symbol (default: 90).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=30000,
        help="CCXT timeout in ms (default: 30000).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts for OHLCV fetches (default: 3).",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=5.0,
        help="Base delay in seconds for retry backoff (default: 5.0).",
    )
    parser.add_argument(
        "--max-atr-usd",
        type=float,
        default=float(os.getenv("SHORT_MAX_ATR_USD", "3000")),
        help="Override ATR USD cap (default from SHORT_MAX_ATR_USD or 3000).",
    )
    parser.add_argument(
        "--max-atr-bps",
        type=float,
        default=float(os.getenv("SHORT_MAX_ATR_BPS", "400")),
        help="Override ATR bps cap (default from SHORT_MAX_ATR_BPS or 400).",
    )
    args = parser.parse_args()

    cfg = AnalysisConfig(
        exchange_id=args.exchange,
        timeframe="1d",
        atr_period=args.atr_period,
        lookback_days=args.lookback_days,
        timeout_ms=args.timeout_ms,
        retries=args.retries,
        retry_base_delay_s=args.retry_base_delay,
        max_atr_usd=args.max_atr_usd,
        max_atr_bps=args.max_atr_bps,
    )
    analyze(args.input, cfg, args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
