#!/usr/bin/env python3
"""
Baseline backtest generator for watchdog closed-trade logs.

This script replays *your existing entries* from `trade_logs/watchdog_closed_positions.csv`
and simulates an alternative, simple exit model:
  - Stop-loss = ATR(7) * atr_mult
  - Take-profit = rr * (entry - stop)
  - Time stop = expiry-hours (default 24h)

It writes a CSV compatible with `watchdog_dashboard.py` so you can compare equity curves
by selecting the output file in the Streamlit UI.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import ccxt  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("ccxt is required. Install with `pip install -r requirements.txt`.") from exc


LOGGER = logging.getLogger("watchdog_baseline_backtest")


WATCHDOG_FIELDS = [
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


def _fmt_ts(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _dynamic_atr_bps(price: float) -> float:
    if price >= 20000:
        return 325.0
    if price >= 2000:
        return 350.0
    if price >= 200:
        return 400.0
    return 450.0


def _effective_atr_cap_usd(price: float, *, max_atr_usd: Optional[float], max_atr_bps: Optional[float]) -> Optional[float]:
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
    """ATR using Wilder smoothing, matching `long_term_crypto_finder._calculate_atr`."""
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


def _compute_excursions(side: str, entry: float, candles: Sequence[Dict[str, object]]) -> Tuple[Optional[float], Optional[float]]:
    if entry <= 0 or not candles:
        return None, None
    side = side.upper()
    mae: Optional[float] = None
    mfe: Optional[float] = None
    for bar in candles:
        high = float(bar["high"])
        low = float(bar["low"])
        if side == "LONG":
            adverse = (low - entry) / entry * 100.0
            favorable = (high - entry) / entry * 100.0
        else:
            adverse = (entry - high) / entry * 100.0
            favorable = (entry - low) / entry * 100.0
        mae = adverse if mae is None else min(mae, adverse)
        mfe = favorable if mfe is None else max(mfe, favorable)
    return mae, mfe


def simulate_bracket_exit(
    *,
    candles: Sequence[Dict[str, object]],
    opened_at: datetime,
    expiry_at: datetime,
    timeframe_seconds: int,
    side: str,
    entry: float,
    stop: float,
    take_profit: float,
) -> Tuple[float, str, datetime, Optional[float], Optional[float]]:
    if not candles:
        return entry, "no_market_data", opened_at, None, None

    interval = timedelta(seconds=max(1, int(timeframe_seconds)))
    usable = [bar for bar in candles if isinstance(bar.get("timestamp"), datetime) and bar["timestamp"] >= opened_at - interval]
    if not usable:
        usable = list(candles)

    mae_pct, mfe_pct = _compute_excursions(side, entry, usable)
    side = side.upper()

    for bar in usable:
        ts = bar.get("timestamp")
        if not isinstance(ts, datetime):
            continue
        if ts >= expiry_at:
            break
        high = float(bar["high"])
        low = float(bar["low"])
        if side == "LONG":
            hit_stop = stop > 0 and low <= stop
            hit_tp = take_profit > 0 and high >= take_profit
        else:
            hit_stop = stop > 0 and high >= stop
            hit_tp = take_profit > 0 and low <= take_profit
        if hit_stop:
            return stop, "stop_loss", ts, mae_pct, mfe_pct
        if hit_tp:
            return take_profit, "take_profit", ts, mae_pct, mfe_pct

    # Expired: close at last close observed before expiry (or the last candle if none).
    last_before_expiry = None
    for bar in reversed(usable):
        ts = bar.get("timestamp")
        if isinstance(ts, datetime) and ts < expiry_at:
            last_before_expiry = bar
            break
    if last_before_expiry is None:
        last_before_expiry = usable[-1]
    ts = last_before_expiry.get("timestamp")
    last_close = float(last_before_expiry["close"])
    closed_at = ts if isinstance(ts, datetime) else expiry_at
    if closed_at > expiry_at:
        closed_at = expiry_at
    return last_close, "expired", closed_at, mae_pct, mfe_pct


@dataclass(frozen=True)
class BaselineConfig:
    exchange_id: str
    intraday_timeframe: str
    daily_timeframe: str
    expiry_hours: float
    atr_period: int
    atr_mult: float
    rr: float
    mode: str  # atr_raw | atr_clipped
    timeout_ms: int
    retries: int
    retry_base_delay_s: float
    max_atr_usd: Optional[float]
    max_atr_bps: Optional[float]


class CcxtOhlcvClient:
    def __init__(self, cfg: BaselineConfig) -> None:
        self.cfg = cfg
        self.exchange = self._build_exchange(cfg.exchange_id)
        self.exchange.load_markets()

    def _build_exchange(self, exchange_id: str):
        params: Dict[str, object] = {"enableRateLimit": True, "timeout": int(self.cfg.timeout_ms)}

        api_key = os.getenv("COINBASE_PERP_API_KEY") or os.getenv("API_KEY_PERPS") or ""
        api_secret = os.getenv("COINBASE_PERP_API_SECRET") or os.getenv("API_SECRET_PERPS") or ""
        if api_key and api_secret:
            params.update({"apiKey": api_key, "secret": api_secret})

        try:
            exchange_cls = getattr(ccxt, exchange_id)
        except AttributeError as exc:
            raise ValueError(f"Unsupported CCXT exchange '{exchange_id}'") from exc
        exchange = exchange_cls(params)
        return exchange

    def _sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, float(seconds)))

    def _with_retries(self, fn, *args, **kwargs):
        attempts = max(1, int(self.cfg.retries))
        base_delay = max(0.1, float(self.cfg.retry_base_delay_s))
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                return fn(*args, **kwargs)
            except ccxt.NetworkError as exc:
                last_exc = exc
                delay = base_delay * (2**attempt)
                LOGGER.warning("Network error (attempt %d/%d): %s; sleeping %.1fs", attempt + 1, attempts, exc, delay)
                self._sleep(delay)
            except ccxt.ExchangeError as exc:
                raise exc
        assert last_exc is not None
        raise last_exc

    def fetch_ohlcv_range(
        self, *, symbol: str, timeframe: str, start: datetime, end: datetime, limit: int = 450
    ) -> List[Dict[str, object]]:
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        interval_ms = int(self.exchange.parse_timeframe(timeframe) * 1000)
        if interval_ms <= 0:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        bars: List[Dict[str, object]] = []
        since = max(0, start_ms - interval_ms)
        safety = 0
        while since <= end_ms and safety < 1000:
            safety += 1
            batch = self._with_retries(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, since=since, limit=limit)
            if not batch:
                break
            for row in batch:
                ts_ms = int(row[0])
                if ts_ms < start_ms:
                    continue
                if ts_ms > end_ms:
                    break
                bars.append(
                    {
                        "timestamp": datetime.fromtimestamp(ts_ms / 1000, tz=UTC),
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                    }
                )
            last_ts = int(batch[-1][0])
            if last_ts >= end_ms or len(batch) < limit:
                break
            next_since = last_ts + interval_ms
            if next_since <= since:
                next_since = since + interval_ms
            since = next_since
        return bars


def _read_watchdog_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def _write_watchdog_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WATCHDOG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _day_start_utc(dt: datetime) -> datetime:
    dt = dt.astimezone(UTC)
    return datetime(dt.year, dt.month, dt.day, tzinfo=UTC)


def _compute_baseline_levels(
    *,
    side: str,
    entry: float,
    atr: float,
    cfg: BaselineConfig,
) -> Tuple[float, float, float]:
    atr_eff = float(atr)
    if cfg.mode == "atr_clipped":
        cap = _effective_atr_cap_usd(entry, max_atr_usd=cfg.max_atr_usd, max_atr_bps=cfg.max_atr_bps)
        if cap is not None and cap > 0:
            atr_eff = min(atr_eff, float(cap))

    risk = float(cfg.atr_mult) * float(atr_eff)
    if risk <= 0:
        return atr_eff, 0.0, 0.0
    side = side.upper()
    if side == "LONG":
        stop = entry - risk
        tp = entry + cfg.rr * risk
    else:
        stop = entry + risk
        tp = entry - cfg.rr * risk
    return atr_eff, float(stop), float(tp)


def _pnl_and_pct(*, side: str, net_size: float, entry: float, exit_price: float) -> Tuple[float, float]:
    qty = abs(float(net_size or 0.0))
    if qty <= 0 or entry <= 0:
        return 0.0, 0.0
    side = side.upper()
    if side == "LONG":
        pnl = qty * (exit_price - entry)
    else:
        pnl = qty * (entry - exit_price)
    pct = pnl / (qty * entry) * 100.0
    return pnl, pct


def run_baseline_backtest(
    *,
    cfg: BaselineConfig,
    in_path: Path,
    out_path: Path,
    start_date: Optional[str],
    end_date: Optional[str],
    max_trades: Optional[int],
) -> None:
    rows = _read_watchdog_rows(in_path)
    if not rows:
        raise SystemExit(f"No rows found in {in_path}")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC) if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1) if end_date else None

    client = CcxtOhlcvClient(cfg)
    trades_out: List[Dict[str, object]] = []

    processed = 0
    skipped = 0
    for row in rows:
        product_id = row.get("product_id", "")
        side = (row.get("position_side") or "").upper()
        if side not in {"LONG", "SHORT"}:
            skipped += 1
            continue
        try:
            opened_at = _parse_ts(row.get("opened_at", ""))
        except Exception:
            skipped += 1
            continue
        if start_dt and opened_at < start_dt:
            continue
        if end_dt and opened_at >= end_dt:
            continue

        try:
            entry = float(row.get("entry_price") or 0.0)
            net_size = float(row.get("net_size") or 0.0)
        except Exception:
            skipped += 1
            continue
        if entry <= 0 or net_size == 0:
            skipped += 1
            continue

        try:
            symbol = _ccxt_symbol(product_id)
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", product_id, exc)
            skipped += 1
            continue

        entry_day_start = _day_start_utc(opened_at)
        daily_start = entry_day_start - timedelta(days=max(60, cfg.atr_period * 10))
        daily_end = entry_day_start - timedelta(seconds=1)
        daily = client.fetch_ohlcv_range(symbol=symbol, timeframe=cfg.daily_timeframe, start=daily_start, end=daily_end, limit=365)
        if len(daily) < cfg.atr_period + 1:
            LOGGER.warning("Insufficient daily candles for %s at %s (got %d)", product_id, opened_at, len(daily))
            skipped += 1
            continue

        highs = np.array([bar["high"] for bar in daily], dtype=float)
        lows = np.array([bar["low"] for bar in daily], dtype=float)
        closes = np.array([bar["close"] for bar in daily], dtype=float)
        atr_raw = calculate_atr_wilder(highs, lows, closes, cfg.atr_period)
        atr_eff, stop, tp = _compute_baseline_levels(side=side, entry=entry, atr=atr_raw, cfg=cfg)
        if stop <= 0 or tp <= 0:
            skipped += 1
            continue

        expiry_at = opened_at + timedelta(hours=float(cfg.expiry_hours))
        intraday_start = opened_at - timedelta(hours=2)
        intraday_end = expiry_at + timedelta(hours=2)
        intraday = client.fetch_ohlcv_range(
            symbol=symbol,
            timeframe=cfg.intraday_timeframe,
            start=intraday_start,
            end=intraday_end,
            limit=500,
        )

        tf_seconds = int(client.exchange.parse_timeframe(cfg.intraday_timeframe))
        exit_price, reason, closed_at, mae_pct, mfe_pct = simulate_bracket_exit(
            candles=intraday,
            opened_at=opened_at,
            expiry_at=expiry_at,
            timeframe_seconds=tf_seconds,
            side=side,
            entry=entry,
            stop=stop,
            take_profit=tp,
        )

        pnl, pnl_pct = _pnl_and_pct(side=side, net_size=net_size, entry=entry, exit_price=exit_price)
        duration_seconds = max(0, int((closed_at - opened_at).total_seconds()))

        out_row: Dict[str, object] = {
            "closed_at": _fmt_ts(closed_at),
            "product_id": product_id,
            "position_side": side,
            "net_size": row.get("net_size", ""),
            "leverage": row.get("leverage", ""),
            "opened_at": _fmt_ts(opened_at),
            "closure_reason": reason,
            "entry_price": f"{entry:.8g}",
            "exit_price": f"{exit_price:.8g}",
            "profit_loss": round(float(pnl), 2),
            "profit_loss_pct": round(float(pnl_pct), 4),
            "mae": round(float(mae_pct), 2) if mae_pct is not None else "",
            "mfe": round(float(mfe_pct), 2) if mfe_pct is not None else "",
            "duration_seconds": duration_seconds,
        }
        trades_out.append(out_row)
        processed += 1
        if max_trades and processed >= max_trades:
            break

        if processed % 10 == 0:
            LOGGER.info("Processed %d trades...", processed)

    if not trades_out:
        raise SystemExit("No baseline trades produced (all rows skipped).")
    _write_watchdog_rows(out_path, trades_out)
    LOGGER.info("Baseline written: %s (trades=%d, skipped=%d)", out_path, len(trades_out), skipped)


def _default_out_path(mode: str) -> Path:
    suffix = "atr_raw" if mode == "atr_raw" else "atr_clipped"
    return Path("trade_logs") / f"watchdog_closed_positions_baseline_{suffix}.csv"


def main() -> None:
    _load_dotenv_if_available()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Generate a baseline CSV from watchdog closed trades.")
    parser.add_argument("--in", dest="in_path", type=Path, default=Path("trade_logs/watchdog_closed_positions.csv"))
    parser.add_argument("--out", dest="out_path", type=Path, default=None)
    parser.add_argument("--mode", choices=["atr_raw", "atr_clipped"], default="atr_raw")
    parser.add_argument("--expiry-hours", type=float, default=24.0)
    parser.add_argument("--intraday-timeframe", type=str, default="1h")
    parser.add_argument("--daily-timeframe", type=str, default="1d")
    parser.add_argument("--atr-period", type=int, default=7)
    parser.add_argument("--atr-mult", type=float, default=1.3)
    parser.add_argument("--rr", type=float, default=2.0)
    parser.add_argument("--exchange", type=str, default="coinbaseadvanced")
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=5.0)
    parser.add_argument("--start-opened", type=str, default=None, help="Filter trades opened on/after YYYY-MM-DD (UTC).")
    parser.add_argument("--end-opened", type=str, default=None, help="Filter trades opened on/before YYYY-MM-DD (UTC).")
    parser.add_argument("--max-trades", type=int, default=None)
    parser.add_argument(
        "--max-atr-usd",
        type=float,
        default=float(os.getenv("SHORT_MAX_ATR_USD", "3000") or 3000.0),
        help="Only used in atr_clipped mode; defaults to env SHORT_MAX_ATR_USD or 3000.",
    )
    parser.add_argument(
        "--max-atr-bps",
        type=float,
        default=float(os.getenv("SHORT_MAX_ATR_BPS", "400") or 400.0),
        help="Only used in atr_clipped mode; defaults to env SHORT_MAX_ATR_BPS or 400.",
    )
    args = parser.parse_args()

    out_path = args.out_path or _default_out_path(args.mode)
    cfg = BaselineConfig(
        exchange_id=args.exchange,
        intraday_timeframe=args.intraday_timeframe,
        daily_timeframe=args.daily_timeframe,
        expiry_hours=args.expiry_hours,
        atr_period=args.atr_period,
        atr_mult=args.atr_mult,
        rr=args.rr,
        mode=args.mode,
        timeout_ms=args.timeout_ms,
        retries=args.retries,
        retry_base_delay_s=args.retry_base_delay,
        max_atr_usd=args.max_atr_usd,
        max_atr_bps=args.max_atr_bps,
    )

    run_baseline_backtest(
        cfg=cfg,
        in_path=args.in_path,
        out_path=out_path,
        start_date=args.start_opened,
        end_date=args.end_opened,
        max_trades=args.max_trades,
    )


if __name__ == "__main__":
    main()

