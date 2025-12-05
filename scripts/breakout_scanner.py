#!/usr/bin/env python
"""
Simple breakout scanner for liquid majors.

Finds breakouts above recent swing highs/lows with volume thrust, computes
structure-based stops and 2R targets, and prints a ranked list.
"""
import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import ccxt  # type: ignore
import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------


def load_ohlcv(exchange: ccxt.Exchange, symbol: str, tf: str, limit: int = 200) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = np.maximum.reduce([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ])
    atr = pd.Series(tr).rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()


def latest_breakout(df: pd.DataFrame, lookback: int = 50, side: str = "long") -> Optional[Tuple[float, float]]:
    if len(df) < lookback + 1:
        return None
    last_close = float(df["close"].iloc[-1])
    window = df.iloc[-lookback:]
    if side == "long":
        swing_high = float(window["high"].max())
        swing_low = float(window["low"].tail(10).min())
        if last_close <= swing_high:
            return None
        return swing_high, swing_low
    swing_low = float(window["low"].min())
    swing_high = float(window["high"].tail(10).max())
    if last_close >= swing_low:
        return None
    return swing_low, swing_high


# ----------------------------
# Scanner
# ----------------------------


@dataclass
class BreakoutCandidate:
    symbol: str
    side: str
    entry: float
    stop: float
    tp: float
    rr: float
    vol_thrust: float
    trend: float
    adx_val: float
    ts: pd.Timestamp


def scan_symbols(exchange: ccxt.Exchange, symbols: List[str], tf: str, lookback: int) -> List[BreakoutCandidate]:
    out: List[BreakoutCandidate] = []
    for sym in symbols:
        try:
            df = load_ohlcv(exchange, sym, tf, limit=max(lookback + 20, 120))
        except Exception as exc:
            print(f"[warn] failed to load {sym}: {exc}", file=sys.stderr)
            continue
        if df.empty:
            continue

        df["ema50"] = ema(df["close"], 50)
        df["trend"] = (df["ema50"].iloc[-1] / df["ema50"].iloc[-20] - 1.0) if len(df) >= 50 else 0.0
        df["adx"] = adx(df, period=14)
        vol = df["volume"]
        vol_thrust = float(vol.tail(3).mean() / max(vol.tail(20).mean(), 1e-8))
        last_ts = df["ts"].iloc[-1]

        for side in ("long", "short"):
            br = latest_breakout(df, lookback=lookback, side=side)
            if not br:
                continue
            level, opp_level = br
            entry = float(df["close"].iloc[-1])
            if side == "long":
                stop = max(opp_level, entry * 0.99)
                risk = entry - stop
                tp = entry + 2 * risk
                rr = (tp - entry) / risk if risk > 0 else 0.0
            else:
                stop = min(opp_level, entry * 1.01)
                risk = stop - entry
                tp = entry - 2 * risk
                rr = (entry - tp) / risk if risk > 0 else 0.0

            adx_val = float(df["adx"].iloc[-1] or 0.0)
            out.append(
                BreakoutCandidate(
                    symbol=sym,
                    side=side,
                    entry=entry,
                    stop=stop,
                    tp=tp,
                    rr=rr,
                    vol_thrust=vol_thrust,
                    trend=float(df["trend"].iloc[-1]),
                    adx_val=adx_val,
                    ts=last_ts,
                )
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Trend breakout scanner")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC,ETH,SOL,XRP,ADA,DOT,AVAX,MATIC,LINK,LTC,DOGE,USDT,USDC",
        help="Comma-separated symbols (e.g., BTC,ETH or BTC/USDC)",
    )
    parser.add_argument("--timeframe", type=str, default="4h", help="CCXT timeframe (e.g., 1h,4h,1d)")
    parser.add_argument("--lookback", type=int, default=50, help="Lookback bars for swing high/low")
    parser.add_argument("--exchange", type=str, default="coinbaseadvanced", help="Primary CCXT exchange id")
    parser.add_argument("--out", type=str, default="finder_breakout.txt", help="Output file path (finder format)")
    args = parser.parse_args()

    base_symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    # Default to USDC quote if not specified
    symbols = [sym if "/" in sym else f"{sym}/USDC" for sym in base_symbols]

    import os

    def init_exchange(primary: str) -> ccxt.Exchange:
        fallbacks = ["kraken"]
        tried = []
        last_exc = None
        for ex_id in [primary] + [fx for fx in fallbacks if fx != primary]:
            tried.append(ex_id)
            params = {"enableRateLimit": True}
            if ex_id == "coinbaseadvanced":
                api = os.getenv("API_KEY")
                secret = os.getenv("API_SECRET")
                if api and secret:
                    params["apiKey"] = api
                    params["secret"] = secret
            if ex_id == "kraken":
                api = os.getenv("KRAKEN_API_KEY")
                secret = os.getenv("KRAKEN_API_SECRET")
                if api and secret:
                    params["apiKey"] = api
                    params["secret"] = secret
            try:
                exc = getattr(ccxt, ex_id)(params)
                exc.timeout = 30000
                exc.load_markets()
                print(f"[info] using exchange={ex_id}")
                return exc
            except Exception as exc:  # type: ignore
                last_exc = exc
                print(f"[warn] init failed for {ex_id}: {exc}", file=sys.stderr)
                continue
        raise RuntimeError(f"Failed to init exchanges {tried}: {last_exc}")

    exc = init_exchange(args.exchange.lower())
    tf = args.timeframe.lower()
    if hasattr(exc, "timeframes") and exc.timeframes:
        if tf not in exc.timeframes:
            fallback_tf = "1h" if "1h" in exc.timeframes else list(exc.timeframes.keys())[0]
            print(f"[warn] timeframe {tf} not supported on {exc.id}; using {fallback_tf}")
            tf = fallback_tf
    else:
        tf = args.timeframe

    # Filter out self-quotes like USDC/USDC
    symbols = [s for s in symbols if s.split("/")[0] != s.split("/")[-1]]

    cands = scan_symbols(exc, symbols, tf, args.lookback)
    if not cands:
        print("No breakouts found.")
        return

    # Rank: higher RR, then higher vol_thrust, then higher ADX
    cands.sort(key=lambda c: (c.rr, c.vol_thrust, c.adx_val), reverse=True)

    def _fmt(v: float) -> str:
        if v >= 1000:
            return f"{v:,.2f}"
        if v >= 10:
            return f"{v:,.3f}"
        return f"{v:,.5f}"

    lines: List[str] = []
    lines.append("================================================================")
    lines.append(f"Breakout candidates (tf={tf}, lookback={args.lookback})")
    lines.append("================================================================")
    for idx, c in enumerate(cands, start=1):
        lines.append(f"{idx}. {c.symbol} — {c.side.upper()}")
        lines.append(f"Data Timestamp (UTC): {c.ts}")
        lines.append(f"TRADING LEVELS ({c.side.upper()})")
        lines.append(f"Entry Price: ${_fmt(c.entry)}")
        lines.append(f"Stop Loss: ${_fmt(c.stop)}")
        lines.append(f"Take Profit: ${_fmt(c.tp)}")
        lines.append("Recommended Position Size: 0.0%")
        lines.append(
            f"RR={c.rr:.2f}  vol_thrust={c.vol_thrust:.2f}  "
            f"trend={c.trend:.3f}  adx={c.adx_val:.1f}"
        )
        lines.append("----------------------------------------------------------------")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
