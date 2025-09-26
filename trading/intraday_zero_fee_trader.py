"""Intraday zero-fee perpetual trading framework.

This module provides building blocks for running a frequent-trading bot on
zero-fee INTX perpetual products (for example BTC-PERP-INTX). The primary goal
is to harvest small intraday edges made viable by the absence of taker fees.

The design favours safe defaults:
- Dry-run mode unless explicitly enabled via configuration
- Risk and cooldown guards to avoid runaway position churn
- Signal engine tuned for 1–5 minute candles with both momentum continuation
  and mean-reversion scalp entries

The high-level flow is:
    1. Fetch historical candles for each configured product
    2. Compute technical features suited for microstructure-aware trading
    3. Evaluate entry signals and apply risk/cooldown filters
    4. Dispatch orders through either a paper executor or CoinbaseService
    5. Update open positions and exit when targets/guards trigger

Usage: instantiate :class:`ZeroFeePerpTrader` and call ``run_once`` inside a
scheduler or ``run_forever`` for a blocking loop. A CLI wrapper is provided in
``auto_zero_fee_trader.py``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from config import API_KEY_PERPS, API_SECRET_PERPS  # type: ignore
except ImportError:  # pragma: no cover - tests can monkeypatch
    API_KEY_PERPS = None
    API_SECRET_PERPS = None

from coinbaseservice import CoinbaseService

LOGGER = logging.getLogger(__name__)

GRANULARITY_SECONDS = {
    "ONE_MINUTE": 60,
    "TWO_MINUTE": 120,
    "THREE_MINUTE": 180,
    "FIVE_MINUTE": 300,
}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _env_list(key: str, default: List[str]) -> List[str]:
    raw = os.getenv(key)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        LOGGER.warning("Invalid float for %s: %s; using %s", key, raw, default)
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        LOGGER.warning("Invalid int for %s: %s; using %s", key, raw, default)
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y"}


DEFAULT_LIQUID_PRODUCTS = [
    "BTC-PERP-INTX",
    "ETH-PERP-INTX",
    "SOL-PERP-INTX",
    "XRP-PERP-INTX",
    "DOGE-PERP-INTX",
    "ADA-PERP-INTX",
    "BCH-PERP-INTX",
    "LTC-PERP-INTX",
    "AVAX-PERP-INTX",
    "DOT-PERP-INTX",
    "LINK-PERP-INTX",
    "ATOM-PERP-INTX",
    "1000SHIB-PERP-INTX",
    "APT-PERP-INTX",
]


@dataclass
class IntradayTraderConfig:
    """Configuration for the intraday zero-fee trading system."""

    product_ids: List[str] = field(default_factory=lambda: list(DEFAULT_LIQUID_PRODUCTS))
    granularity: str = "ONE_MINUTE"
    lookback_bars: int = 720  # 12 hours on 1m bars
    poll_seconds: int = 60
    cooldown_seconds: int = 180
    min_volume_ratio: float = 1.1
    volume_lookback: int = 60
    ema_fast: int = 12
    ema_slow: int = 26
    ema_trend: int = 55
    rsi_period: int = 14
    rsi_buy_threshold: float = 34.0
    rsi_sell_threshold: float = 66.0
    pullback_pct: float = 0.0015  # 0.15%
    breakout_zscore: float = 1.2
    atr_period: int = 14
    stop_loss_pct: float = 0.002  # 0.2%
    take_profit_pct: float = 0.0035  # 0.35%
    trailing_atr_multiple: float = 1.2
    max_open_positions: int = 2
    max_exposure_usd: float = 5000.0
    base_position_usd: float = 5.0
    leverage: float = 50.0
    max_position_minutes: int = 45
    dry_run: bool = True
    log_dir: Path = Path("trade_logs")
    state_dir: Path = Path("cache/zero_fee_trader")

    @classmethod
    def from_env(cls) -> "IntradayTraderConfig":
        product_ids = _env_list("AUTO_ZERO_FEE_PRODUCTS", DEFAULT_LIQUID_PRODUCTS)
        granularity = os.getenv("AUTO_ZERO_FEE_GRANULARITY", "ONE_MINUTE").upper()
        lookback_bars = _env_int("AUTO_ZERO_FEE_LOOKBACK_BARS", 720)
        poll_seconds = _env_int("AUTO_ZERO_FEE_POLL_SECONDS", 60)
        cooldown_seconds = _env_int("AUTO_ZERO_FEE_COOLDOWN", 180)
        min_volume_ratio = _env_float("AUTO_ZERO_FEE_MIN_VOL_RATIO", 1.1)
        volume_lookback = _env_int("AUTO_ZERO_FEE_VOL_LOOKBACK", 60)
        ema_fast = _env_int("AUTO_ZERO_FEE_EMA_FAST", 12)
        ema_slow = _env_int("AUTO_ZERO_FEE_EMA_SLOW", 26)
        ema_trend = _env_int("AUTO_ZERO_FEE_EMA_TREND", 55)
        rsi_period = _env_int("AUTO_ZERO_FEE_RSI_PERIOD", 14)
        rsi_buy_threshold = _env_float("AUTO_ZERO_FEE_RSI_BUY", 34.0)
        rsi_sell_threshold = _env_float("AUTO_ZERO_FEE_RSI_SELL", 66.0)
        pullback_pct = _env_float("AUTO_ZERO_FEE_PULLBACK_PCT", 0.0015)
        breakout_zscore = _env_float("AUTO_ZERO_FEE_BREAKOUT_Z", 1.2)
        atr_period = _env_int("AUTO_ZERO_FEE_ATR_PERIOD", 14)
        stop_loss_pct = _env_float("AUTO_ZERO_FEE_STOP_PCT", 0.002)
        take_profit_pct = _env_float("AUTO_ZERO_FEE_TP_PCT", 0.0035)
        trailing_atr_multiple = _env_float("AUTO_ZERO_FEE_TRAIL_ATR", 1.2)
        max_open_positions = _env_int("AUTO_ZERO_FEE_MAX_POSITIONS", 2)
        max_exposure_usd = _env_float("AUTO_ZERO_FEE_MAX_EXPOSURE", 5000.0)
        base_position_usd = _env_float("AUTO_ZERO_FEE_BASE_USD", 5.0)
        leverage = _env_float("AUTO_ZERO_FEE_LEVERAGE", 50.0)
        max_position_minutes = _env_int("AUTO_ZERO_FEE_MAX_MINUTES", 45)
        dry_run = not _env_bool("AUTO_ZERO_FEE_LIVE", False)

        cfg = cls(
            product_ids=product_ids,
            granularity=granularity,
            lookback_bars=lookback_bars,
            poll_seconds=poll_seconds,
            cooldown_seconds=cooldown_seconds,
            min_volume_ratio=min_volume_ratio,
            volume_lookback=volume_lookback,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_trend=ema_trend,
            rsi_period=rsi_period,
            rsi_buy_threshold=rsi_buy_threshold,
            rsi_sell_threshold=rsi_sell_threshold,
            pullback_pct=pullback_pct,
            breakout_zscore=breakout_zscore,
            atr_period=atr_period,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_atr_multiple=trailing_atr_multiple,
            max_open_positions=max_open_positions,
            max_exposure_usd=max_exposure_usd,
            base_position_usd=base_position_usd,
            leverage=leverage,
            max_position_minutes=max_position_minutes,
            dry_run=dry_run,
        )
        cfg.log_dir = Path(os.getenv("AUTO_ZERO_FEE_LOG_DIR", str(cfg.log_dir)))
        cfg.state_dir = Path(os.getenv("AUTO_ZERO_FEE_STATE_DIR", str(cfg.state_dir)))
        return cfg

    @property
    def bar_seconds(self) -> int:
        if self.granularity not in GRANULARITY_SECONDS:
            raise ValueError(f"Unsupported granularity: {self.granularity}")
        return GRANULARITY_SECONDS[self.granularity]


# ---------------------------------------------------------------------------
# Data engineering
# ---------------------------------------------------------------------------


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / (roll_down + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr


def build_feature_frame(df: pd.DataFrame, config: IntradayTraderConfig) -> pd.DataFrame:
    minimum_required = max(
        config.ema_trend + 5,
        config.volume_lookback + 5,
        config.rsi_period + 5,
        config.atr_period + 5,
    )
    if len(df) < minimum_required:
        raise ValueError("Insufficient data for feature computation")

    features = df.sort_index().copy()
    features["ema_fast"] = features["close"].ewm(span=config.ema_fast, adjust=False).mean()
    features["ema_slow"] = features["close"].ewm(span=config.ema_slow, adjust=False).mean()
    features["ema_trend"] = features["close"].ewm(span=config.ema_trend, adjust=False).mean()
    features["ema_fast_slope"] = features["ema_fast"].diff()
    features["returns"] = features["close"].pct_change().fillna(0.0)
    features["return_z"] = (features["returns"] - features["returns"].rolling(30).mean()) / (
        features["returns"].rolling(30).std() + 1e-9
    )
    features["volume"] = features["volume"].astype(float)
    features["volume_ma"] = features["volume"].rolling(config.volume_lookback).mean()
    features["volume_ratio"] = features["volume"] / (features["volume_ma"] + 1e-9)
    features["rsi"] = compute_rsi(features["close"], config.rsi_period)
    features["atr"] = compute_atr(features, config.atr_period)
    features["atr_pct"] = features["atr"] / features["close"].abs()
    features.dropna(inplace=True)
    return features


# ---------------------------------------------------------------------------
# Trading primitives
# ---------------------------------------------------------------------------


@dataclass
class SignalDecision:
    product_id: str
    side: str  # "buy" or "sell"
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    rationale: str
    leverage: float = 1.0


@dataclass
class Position:
    product_id: str
    side: str
    size: float
    entry_price: float
    opened_at: datetime
    stop_loss: float
    take_profit: float
    leverage: float

    def age_minutes(self) -> float:
        return (datetime.now(UTC) - self.opened_at).total_seconds() / 60.0


class RiskManager:
    def __init__(self, config: IntradayTraderConfig):
        self.config = config

    def remaining_capacity(self, positions: List[Position]) -> float:
        exposure = sum(pos.entry_price * pos.size for pos in positions)
        return max(0.0, self.config.max_exposure_usd - exposure)

    def approve(self, decision: SignalDecision, positions: List[Position], cooldown_ok: bool) -> bool:
        if not cooldown_ok:
            return False
        if len(positions) >= self.config.max_open_positions:
            return False
        exposure = sum(pos.entry_price * pos.size for pos in positions)
        exposure += decision.entry_price * decision.size
        if exposure > self.config.max_exposure_usd:
            return False
        return True


class CooldownTracker:
    def __init__(self, cooldown_seconds: int):
        self.cooldown_seconds = cooldown_seconds
        self._last_signal: Dict[str, datetime] = {}

    def mark(self, key: str) -> None:
        self._last_signal[key] = datetime.now(UTC)

    def ready(self, key: str) -> bool:
        last = self._last_signal.get(key)
        if last is None:
            return True
        return (datetime.now(UTC) - last).total_seconds() >= self.cooldown_seconds


class SignalEngine:
    def __init__(self, config: IntradayTraderConfig):
        self.config = config

    def _position_size(self, price: float) -> float:
        notional = self.config.base_position_usd * self.config.leverage
        return max(0.0, notional / max(price, 1e-8))

    def _calc_levels(self, side: str, price: float, atr: float) -> tuple[float, float]:
        if side == "buy":
            stop = price * (1 - self.config.stop_loss_pct)
            tp = price * (1 + self.config.take_profit_pct)
            if atr > 0:
                stop = min(stop, price - self.config.trailing_atr_multiple * atr)
                tp = max(tp, price + self.config.trailing_atr_multiple * atr)
        else:
            stop = price * (1 + self.config.stop_loss_pct)
            tp = price * (1 - self.config.take_profit_pct)
            if atr > 0:
                stop = max(stop, price + self.config.trailing_atr_multiple * atr)
                tp = min(tp, price - self.config.trailing_atr_multiple * atr)
        return stop, tp

    def evaluate(self, product_id: str, features: pd.DataFrame) -> Optional[SignalDecision]:
        last = features.iloc[-1]
        prev = features.iloc[-2]

        price = float(last["close"])
        atr = float(last.get("atr", 0.0))
        volume_ratio = float(last.get("volume_ratio", 1.0))

        trend_up = last["ema_fast"] > last["ema_slow"] > last["ema_trend"]
        trend_down = last["ema_fast"] < last["ema_slow"] < last["ema_trend"]
        momentum_up = last["ema_fast_slope"] > 0 and prev["ema_fast_slope"] <= 0
        momentum_down = last["ema_fast_slope"] < 0 and prev["ema_fast_slope"] >= 0

        pullback = (last["close"] <= last["ema_fast"] * (1 - self.config.pullback_pct)) and last["returns"] < 0
        snapback = (last["close"] >= last["ema_fast"] * (1 + self.config.pullback_pct)) and last["returns"] > 0

        breakout_long = last["return_z"] > self.config.breakout_zscore
        breakout_short = last["return_z"] < -self.config.breakout_zscore

        rsi = float(last["rsi"])
        rationales: List[str] = []

        if volume_ratio < self.config.min_volume_ratio:
            return None

        # Long bias: uptrend pullback or breakout continuation
        if trend_up and (pullback or breakout_long) and rsi <= self.config.rsi_buy_threshold:
            if pullback:
                rationales.append("trend_up_pullback")
            if breakout_long:
                rationales.append("momentum_breakout")
            if momentum_up:
                rationales.append("ema_accelerating")
            size = self._position_size(price)
            if size <= 0:
                return None
            stop, tp = self._calc_levels("buy", price, atr)
            rationale = ",".join(rationales) or "trend_up"
            return SignalDecision(
                product_id=product_id,
                side="buy",
                size=size,
                entry_price=price,
                stop_loss=stop,
                take_profit=tp,
                rationale=rationale,
                leverage=self.config.leverage,
            )

        # Short bias: downtrend snapback or breakdown continuation
        rationales.clear()
        if trend_down and (snapback or breakout_short) and rsi >= self.config.rsi_sell_threshold:
            if snapback:
                rationales.append("trend_down_snapback")
            if breakout_short:
                rationales.append("momentum_breakdown")
            if momentum_down:
                rationales.append("ema_decaying")
            size = self._position_size(price)
            if size <= 0:
                return None
            stop, tp = self._calc_levels("sell", price, atr)
            rationale = ",".join(rationales) or "trend_down"
            return SignalDecision(
                product_id=product_id,
                side="sell",
                size=size,
                entry_price=price,
                stop_loss=stop,
                take_profit=tp,
                rationale=rationale,
                leverage=self.config.leverage,
            )

        return None


class ExecutionEngine:
    def enter(self, decision: SignalDecision) -> Optional[Position]:  # pragma: no cover - interface
        raise NotImplementedError

    def exit(self, position: Position, reason: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class PaperExecutionEngine(ExecutionEngine):
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_dir / "zero_fee_paper_trades.csv"
        if not self._log_file.exists():
            self._log_file.write_text("timestamp,product_id,side,price,size,stop_loss,take_profit,rationale\n")

    def enter(self, decision: SignalDecision) -> Position:
        timestamp = datetime.now(UTC).isoformat()
        line = (
            f"{timestamp},{decision.product_id},{decision.side},{decision.entry_price:.2f},{decision.size:.6f},"
            f"{decision.stop_loss:.2f},{decision.take_profit:.2f},{decision.rationale}\n"
        )
        with self._log_file.open("a") as fh:
            fh.write(line)
        LOGGER.info("[PAPER] %s %s size=%.6f price=%.2f rationale=%s", decision.side.upper(), decision.product_id, decision.size, decision.entry_price, decision.rationale)
        return Position(
            product_id=decision.product_id,
            side=decision.side,
            size=decision.size,
            entry_price=decision.entry_price,
            opened_at=datetime.now(UTC),
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            leverage=decision.leverage,
        )

    def exit(self, position: Position, reason: str) -> bool:
        LOGGER.info(
            "[PAPER EXIT] %s %s size=%.6f price=%.2f reason=%s",
            position.side.upper(),
            position.product_id,
            position.size,
            position.entry_price,
            reason,
        )
        return True


class CoinbaseExecutionEngine(ExecutionEngine):
    def __init__(self, service: CoinbaseService, log_dir: Path):
        self.service = service
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_dir / "zero_fee_live_trades.csv"
        if not self._log_file.exists():
            header = (
                "timestamp,product_id,side,price,size,stop_loss,take_profit," \
                "rationale,leverage,order_id,event,reason\n"
            )
            self._log_file.write_text(header)

    @staticmethod
    def _extract_order_id(response: Optional[dict]) -> str:
        if not isinstance(response, dict):
            return ""
        for key in ("order_id", "id", "orderId", "client_order_id", "clientOrderId"):
            if key in response:
                return str(response[key])
        return ""

    def _write_live_log(
        self,
        *,
        product_id: str,
        side: str,
        price: float,
        size: float,
        stop_loss: float,
        take_profit: float,
        rationale: str,
        leverage: float,
        order_id: str,
        event: str,
        reason: str = "",
    ) -> None:
        try:
            timestamp = datetime.now(UTC).isoformat()
            line = (
                f"{timestamp},{product_id},{side},{price:.2f},{size:.6f},"
                f"{stop_loss:.2f},{take_profit:.2f},{rationale},{leverage:.2f},{order_id},{event},{reason}\n"
            )
            with self._log_file.open("a") as fh:
                fh.write(line)
        except Exception:  # pragma: no cover - logging best-effort only
            LOGGER.exception("Failed to record live trade to %s", self._log_file)

    def _record_trade(self, decision: SignalDecision, response: Optional[dict]) -> str:
        order_id = self._extract_order_id(response)
        self._write_live_log(
            product_id=decision.product_id,
            side=decision.side,
            price=decision.entry_price,
            size=decision.size,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            rationale=decision.rationale,
            leverage=decision.leverage,
            order_id=order_id,
            event="entry",
        )
        return order_id

    def _record_exit(self, position: Position, response: Optional[dict], reason: str) -> None:
        order_id = self._extract_order_id(response)
        self._write_live_log(
            product_id=position.product_id,
            side=position.side,
            price=position.entry_price,
            size=position.size,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            rationale="",
            leverage=position.leverage,
            order_id=order_id,
            event="exit",
            reason=reason,
        )

    def _record_bracket(self, position: Position, order_id: str, result: dict) -> None:
        outcome = "success" if result.get("status") == "success" else result.get("error", "error")
        self._write_live_log(
            product_id=position.product_id,
            side=position.side,
            price=position.entry_price,
            size=position.size,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            rationale="",
            leverage=position.leverage,
            order_id=order_id,
            event="bracket",
            reason=outcome,
        )

    def _maybe_place_brackets(self, decision: SignalDecision, position: Position, order_id: str) -> None:
        if not order_id:
            LOGGER.warning("Cannot place bracket orders without an order_id")
            return
        try:
            leverage_arg = str(decision.leverage) if decision.leverage else None
            result = self.service.place_bracket_after_fill(
                product_id=decision.product_id,
                order_id=order_id,
                size=decision.size,
                take_profit_price=decision.take_profit,
                stop_loss_price=decision.stop_loss,
                leverage=leverage_arg,
            )
            if isinstance(result, dict) and result.get("status") == "success":
                LOGGER.info(
                    "Placed bracket orders for %s order_id=%s tp=%s sl=%s",
                    decision.product_id,
                    order_id,
                    result.get("tp_price"),
                    result.get("sl_price"),
                )
            else:
                LOGGER.warning("Bracket placement warning for %s: %s", decision.product_id, result)
            if isinstance(result, dict):
                self._record_bracket(position, order_id, result)
            else:
                self._record_bracket(position, order_id, {"error": str(result)})
        except Exception as exc:  # pragma: no cover - network/API errors
            LOGGER.exception("Failed to place bracket orders for %s: %s", decision.product_id, exc)
            self._record_bracket(position, order_id, {"error": str(exc)})

    def enter(self, decision: SignalDecision) -> Optional[Position]:  # pragma: no cover - requires live trading
        response = self.service.place_order(
            product_id=decision.product_id,
            side="BUY" if decision.side == "buy" else "SELL",
            size=decision.size,
            order_type="MARKET",
            leverage=decision.leverage,
            margin_type="CROSS" if decision.leverage and decision.leverage > 1 else None,
        )
        LOGGER.info("Placed live order: %s", response)
        order_id = self._record_trade(decision, response if isinstance(response, dict) else None)
        position = Position(
            product_id=decision.product_id,
            side=decision.side,
            size=decision.size,
            entry_price=decision.entry_price,
            opened_at=datetime.now(UTC),
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            leverage=decision.leverage,
        )
        self._maybe_place_brackets(decision, position, order_id)
        return position

    def exit(self, position: Position, reason: str) -> bool:  # pragma: no cover - requires live trading
        side = "SELL" if position.side == "buy" else "BUY"
        response = self.service.place_order(
            product_id=position.product_id,
            side=side,
            size=position.size,
            order_type="MARKET",
            leverage=position.leverage,
            margin_type="CROSS" if position.leverage and position.leverage > 1 else None,
        )
        if response is None:
            LOGGER.error(
                "Failed to close %s %s size=%.6f leverage=%.2f reason=%s",
                side,
                position.product_id,
                position.size,
                position.leverage,
                reason,
            )
            return False

        LOGGER.info("Closed live position: %s reason=%s", response, reason)
        self._record_exit(position, response if isinstance(response, dict) else None, reason)
        return True


# ---------------------------------------------------------------------------
# Trader orchestrator
# ---------------------------------------------------------------------------


class ZeroFeePerpTrader:
    def __init__(self, config: IntradayTraderConfig, execution: Optional[ExecutionEngine] = None, service: Optional[CoinbaseService] = None):
        self.config = config
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.service = service or self._build_service()
        self.execution = execution or self._default_execution()
        self.signal_engine = SignalEngine(config)
        self.risk_manager = RiskManager(config)
        self.cooldown = CooldownTracker(config.cooldown_seconds)
        self.positions: List[Position] = []
        self.state_dir = config.state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _build_service(self) -> CoinbaseService:
        if API_KEY_PERPS and API_SECRET_PERPS:
            return CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
        if self.config.dry_run:
            LOGGER.info("Using unauthenticated CoinbaseService for public data (dry run)")
            return CoinbaseService(api_key="", api_secret="")
        raise RuntimeError("API credentials are required for live trading")

    def _default_execution(self) -> ExecutionEngine:
        if self.config.dry_run:
            LOGGER.info("Using paper execution engine (dry run)")
            return PaperExecutionEngine(self.config.log_dir)
        return CoinbaseExecutionEngine(self.service, self.config.log_dir)

    # ----------------------------
    # Data fetch & preprocessing
    # ----------------------------

    def _fetch_candles(self, product_id: str) -> pd.DataFrame:
        if self.service is None:
            self.service = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS) if (API_KEY_PERPS and API_SECRET_PERPS) else CoinbaseService("", "")
        now = datetime.now(UTC)
        duration = timedelta(seconds=self.config.bar_seconds * self.config.lookback_bars)
        start = now - duration
        candles = self.service.historical_data.get_historical_data(
            product_id=product_id,
            start_date=start,
            end_date=now,
            granularity=self.config.granularity,
            force_refresh=True,
        )
        if not candles:
            raise ValueError(f"No candle data returned for {product_id}")
        df = pd.DataFrame(candles)
        for column in ("open", "high", "low", "close", "volume"):
            df[column] = pd.to_numeric(df[column], errors="coerce")
        timestamp_column = None
        for candidate in ("start", "time", "timestamp"):
            if candidate in df.columns:
                timestamp_column = candidate
                break
        if timestamp_column is None:
            raise ValueError("Candle data missing timestamp column")
        df[timestamp_column] = pd.to_numeric(df[timestamp_column], errors="coerce")
        df.index = pd.to_datetime(df[timestamp_column], unit="s", utc=True)
        df = df.sort_index()
        return df.tail(self.config.lookback_bars)

    # ----------------------------
    # Position management
    # ----------------------------

    def _prune_positions(self) -> None:
        alive: List[Position] = []
        for pos in self.positions:
            age = pos.age_minutes()
            if age >= self.config.max_position_minutes:
                LOGGER.info(
                    "Closing %s %s for max_age %.2f>=%.2f",
                    pos.side,
                    pos.product_id,
                    age,
                    self.config.max_position_minutes,
                )
                if not self.execution.exit(pos, reason="max_age"):
                    LOGGER.warning(
                        "Exit failed for %s %s (max_age). Will retry next loop.",
                        pos.side,
                        pos.product_id,
                    )
                    alive.append(pos)
                continue
            alive.append(pos)
        self.positions = alive

    def _check_stops(self, product_id: str, features: pd.DataFrame) -> None:
        last = features.iloc[-1]
        high = float(last.get("high", last["close"]))
        low = float(last.get("low", last["close"]))
        updated_positions: List[Position] = []
        for pos in self.positions:
            if pos.product_id != product_id:
                updated_positions.append(pos)
                continue
            exit_reason: Optional[str] = None
            if pos.side == "buy":
                if low <= pos.stop_loss:
                    exit_reason = "stop_hit"
                elif high >= pos.take_profit:
                    exit_reason = "target_hit"
            else:
                if high >= pos.stop_loss:
                    exit_reason = "stop_hit"
                elif low <= pos.take_profit:
                    exit_reason = "target_hit"
            if exit_reason:
                if not self.execution.exit(pos, exit_reason):
                    LOGGER.warning(
                        "Exit failed for %s %s (reason=%s). Will retry.",
                        pos.side,
                        pos.product_id,
                        exit_reason,
                    )
                    updated_positions.append(pos)
            else:
                updated_positions.append(pos)
        self.positions = updated_positions

    # ----------------------------
    # Main loop
    # ----------------------------

    def run_once(self) -> None:
        self._prune_positions()
        for product_id in self.config.product_ids:
            try:
                df = self._fetch_candles(product_id)
            except Exception as exc:  # pragma: no cover - network heavy
                LOGGER.error("Failed to fetch candles for %s: %s", product_id, exc)
                continue
            try:
                features = build_feature_frame(df, self.config)
            except Exception as exc:
                LOGGER.error("Failed to build features for %s: %s", product_id, exc)
                continue
            self._check_stops(product_id, features)
            decision = self.signal_engine.evaluate(product_id, features)
            if decision is None:
                continue
            cooldown_key = f"{product_id}:{decision.side}"
            if not self.cooldown.ready(cooldown_key):
                continue
            if not self.risk_manager.approve(decision, self.positions, cooldown_ok=True):
                continue
            position = self.execution.enter(decision)
            if position:
                self.positions.append(position)
                self.cooldown.mark(cooldown_key)

    def run_forever(self) -> None:  # pragma: no cover - long running
        LOGGER.info("Starting zero-fee perp trader loop; dry_run=%s", self.config.dry_run)
        while True:
            start = time.monotonic()
            self.run_once()
            elapsed = time.monotonic() - start
            sleep_seconds = max(0, self.config.poll_seconds - int(elapsed))
            if sleep_seconds:
                time.sleep(sleep_seconds)


__all__ = [
    "IntradayTraderConfig",
    "SignalDecision",
    "Position",
    "ZeroFeePerpTrader",
    "PaperExecutionEngine",
    "CoinbaseExecutionEngine",
    "build_feature_frame",
    "compute_rsi",
    "compute_atr",
]
