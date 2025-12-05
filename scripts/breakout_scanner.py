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
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT", help="Comma-separated symbols")
    parser.add_argument("--timeframe", type=str, default="4h", help="CCXT timeframe (e.g., 1h,4h,1d)")
    parser.add_argument("--lookback", type=int, default=50, help="Lookback bars for swing high/low")
    args = parser.parse_args()

    symbols = [s.strip().upper().replace("-", "/") for s in args.symbols.split(",") if s.strip()]

    exc = ccxt.kraken({"enableRateLimit": True})
    kr_key = exc.safe_value({}, "apiKey")
    # If env vars exist, ccxt picks them up via constructor params; pass explicitly:
    if exc.apiKey == "" and exc.secret == "":
        import os

        k = os.getenv("KRAKEN_API_KEY")
        s = os.getenv("KRAKEN_API_SECRET")
        if k and s:
            exc.apiKey = k
            exc.secret = s
    exc.timeout = 30000

    cands = scan_symbols(exc, symbols, args.timeframe, args.lookback)
    if not cands:
        print("No breakouts found.")
        return

    # Rank: higher RR, then higher vol_thrust, then higher ADX
    cands.sort(key=lambda c: (c.rr, c.vol_thrust, c.adx_val), reverse=True)

    print("================================================================")
    print(f"Breakout candidates (tf={args.timeframe}, lookback={args.lookback})")
    print("================================================================")
    for c in cands:
        print(f"{c.symbol} {c.side.upper()}  ts={c.ts}  RR={c.rr:.2f}")
        print(
            f" entry={c.entry:.4f}  stop={c.stop:.4f}  tp={c.tp:.4f}  "
            f"vol_thrust={c.vol_thrust:.2f}  trend={c.trend:.3f}  adx={c.adx_val:.1f}"
        )
        print("----------------------------------------------------------------")


if __name__ == "__main__":
    main()
