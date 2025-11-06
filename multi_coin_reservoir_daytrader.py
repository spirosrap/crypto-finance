#!/usr/bin/env python3
"""Multi-Coin Reservoir Day Trader

This script builds on the short-term finder architecture to generate
24-hour long/short trade ideas across multiple crypto assets using a
shared echo state network (reservoir computing) readout. Trade targets
and stops are derived from each asset's current volatility regime.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from coinbaseservice import CoinbaseService
from historicaldata import HistoricalData
from perp_support import canonical_perp_symbol, is_perp_supported, perp_price_multiplier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from llm_scoring import LLMScorer

try:
    from credentials import get_primary_credentials
except ModuleNotFoundError:  # pragma: no cover - fallback for tests
    def _config_value(name: str) -> str:
        try:
            import config as _cfg  # type: ignore
            return getattr(_cfg, name, "") or ""
        except Exception:
            return ""

    def get_primary_credentials() -> Tuple[str, str]:
        return _config_value("API_KEY"), _config_value("API_SECRET")


def _seed_credentials() -> Tuple[str, str]:
    api_key, api_secret = get_primary_credentials()
    if api_key and not os.getenv("API_KEY"):
        os.environ["API_KEY"] = api_key
    if api_secret and not os.getenv("API_SECRET"):
        os.environ["API_SECRET"] = api_secret
    return api_key, api_secret


LOGGER_NAME = "multi_coin_reservoir_daytrader"
DEFAULT_SYMBOLS = ["BTC-USDC", "ETH-USDC", "SOL-USDC", "AVAX-USDC"]
DEFAULT_TIMEFRAME = "ONE_HOUR"
DEFAULT_LOOKBACK = 720
EXPIRY_HOURS = 24
RESERVOIR_PROFILES: Dict[str, Dict[str, Any]] = {
    "default": {
        "max_products": 40,
        "quotes": ["USDC", "USD"],
        "min_volume": 2_000_000.0,
    },
    "wide": {
        "max_products": 150,
        "quotes": ["USDC", "USD", "USDT"],
        "min_volume": 500_000.0,
    },
    "focused": {
        "max_products": 20,
        "quotes": ["USDC"],
        "min_volume": 5_000_000.0,
    },
    "focused_llm_100": {
        "max_products": 100,
        "quotes": ["USDC"],
        "min_volume": 5_000_000.0,
        "use_llm_scoring": True,
        "llm_weight": 0.25,
        "llm_model": "gpt-5-mini",
        "llm_max_candidates": 20,
        "llm_temperature": None,
        "llm_sleep_seconds": 0.0,
    },
    "focused_reservoir_100": {
        "max_products": 100,
        "quotes": ["USDC"],
        "min_volume": 5_000_000.0,
        "use_llm_scoring": False,
    },
    "oos_15m_focus": {
        "max_products": 100,
        "quotes": ["USDC"],
        "min_volume": 2_000_000.0,
        "use_llm_scoring": False,
    },
}

PROFILE_DESCRIPTIONS: Dict[str, str] = {
    "default": "Balanced coverage of the most liquid USDC/USD pairs.",
    "wide": "Broader scan across top-volume quotes (USDC/USD/USDT) up to 150 products.",
    "focused": "Tight list of high-volume USDC majors for faster execution.",
    "focused_llm_100": "Focused USDC basket with OpenAI opinion blended into rankings (100 product scan).",
    "focused_reservoir_100": "Same universe as focused_llm_100 but ranked purely by reservoir scores (no LLM opinion).",
    "oos_15m_focus": "Walk-forward friendly 15m preset (LLM disabled) for faster regime checks.",
}

COINBASE_GRANULARITIES: Dict[str, float] = {
    "ONE_MINUTE": 1 / 60,
    "FIVE_MINUTE": 5 / 60,
    "TEN_MINUTE": 10 / 60,
    "FIFTEEN_MINUTE": 15 / 60,
    "THIRTY_MINUTE": 0.5,
    "ONE_HOUR": 1.0,
    "SIX_HOUR": 6.0,
    "ONE_DAY": 24.0,
}

GRANULARITY_ALIASES: Dict[str, str] = {
    "1M": "ONE_MINUTE",
    "ONE_MINUTE": "ONE_MINUTE",
    "ONE_MIN": "ONE_MINUTE",
    "5M": "FIVE_MINUTE",
    "FIVE_MINUTE": "FIVE_MINUTE",
    "10M": "TEN_MINUTE",
    "TEN_MINUTE": "TEN_MINUTE",
    "15M": "FIFTEEN_MINUTE",
    "FIFTEEN_MINUTE": "FIFTEEN_MINUTE",
    "30M": "THIRTY_MINUTE",
    "THIRTY_MINUTE": "THIRTY_MINUTE",
    "1H": "ONE_HOUR",
    "60M": "ONE_HOUR",
    "ONE_HOUR": "ONE_HOUR",
    "6H": "SIX_HOUR",
    "SIX_HOUR": "SIX_HOUR",
    "1D": "ONE_DAY",
    "24H": "ONE_DAY",
    "ONE_DAY": "ONE_DAY",
}

TIMEFRAME_TO_HOURS: Dict[str, float] = {}
for _key, _value in COINBASE_GRANULARITIES.items():
    TIMEFRAME_TO_HOURS[_key] = _value
    TIMEFRAME_TO_HOURS[_key.lower()] = _value

TIMEFRAME_TO_HOURS.update(
    {
        "1m": COINBASE_GRANULARITIES["ONE_MINUTE"],
        "5m": COINBASE_GRANULARITIES["FIVE_MINUTE"],
        "10m": COINBASE_GRANULARITIES["TEN_MINUTE"],
        "15m": COINBASE_GRANULARITIES["FIFTEEN_MINUTE"],
        "30m": COINBASE_GRANULARITIES["THIRTY_MINUTE"],
        "1h": COINBASE_GRANULARITIES["ONE_HOUR"],
        "6h": COINBASE_GRANULARITIES["SIX_HOUR"],
        "1d": COINBASE_GRANULARITIES["ONE_DAY"],
        "24h": COINBASE_GRANULARITIES["ONE_DAY"],
    }
)


def _setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logs_dir = Path("logs") / LOGGER_NAME
    logs_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )

    file_handler = TimedRotatingFileHandler(
        logs_dir / f"{LOGGER_NAME}.log",
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Logger initialised")
    return logger


logger = _setup_logger(os.getenv("MC_RESERVOIR_LOG_LEVEL", "INFO"))


def timeframe_to_hours(timeframe: str) -> float:
    key = timeframe.strip()
    if key in TIMEFRAME_TO_HOURS:
        return TIMEFRAME_TO_HOURS[key]
    key_lower = key.lower()
    if key_lower in TIMEFRAME_TO_HOURS:
        return TIMEFRAME_TO_HOURS[key_lower]
    key_upper = key.upper()
    if key_upper in TIMEFRAME_TO_HOURS:
        return TIMEFRAME_TO_HOURS[key_upper]
    try:
        unit = key_lower[-1]
        value = float(key_lower[:-1])
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc
    if unit == "m":
        return value / 60.0
    if unit == "h":
        return value
    if unit == "d":
        return value * 24.0
    if unit == "w":
        return value * 24.0 * 7.0
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def normalize_granularity(value: str) -> str:
    raw = value.strip().upper().replace(" ", "_").replace("-", "_")
    normalized = GRANULARITY_ALIASES.get(raw, raw)
    if normalized not in COINBASE_GRANULARITIES:
        raise ValueError(
            f"Unsupported Coinbase granularity {value!r}. "
            f"Supported values: {', '.join(sorted(COINBASE_GRANULARITIES))}"
        )
    return normalized


def normalize_product_id(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "-").replace(":", "-")

def spot_to_perp_id(product_id: str, prefer: str = "PERP-INTX") -> str:
    base = product_id.split("-")[0]
    base = canonical_perp_symbol(base)
    suffix = "INTX-PERP" if prefer == "INTX-PERP" else "PERP-INTX"
    return f"{base}-{suffix}"


def _pget(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def profile_summary() -> str:
    lines = ["Available profiles:"]
    for name in sorted(RESERVOIR_PROFILES):
        cfg = RESERVOIR_PROFILES[name]
        desc = PROFILE_DESCRIPTIONS.get(name, "")
        parts = [
            f"  - {name}",
            f"max_products={cfg.get('max_products', 'all')}",
            f"quotes={','.join(cfg.get('quotes', [])) or 'ANY'}",
            f"min_volume={cfg.get('min_volume', 0)}",
        ]
        if cfg.get("use_llm_scoring"):
            parts.append("llm=on")
        if desc:
            parts.append(desc)
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _parse_quotes(quotes: Optional[str], fallback: Iterable[str]) -> List[str]:
    if quotes:
        return [q.strip().upper() for q in quotes.split(",") if q.strip()]
    return [q.strip().upper() for q in fallback if q]


def discover_product_ids(
    coinbase_service: CoinbaseService,
    quotes: List[str],
    max_products: Optional[int],
    min_volume: float,
) -> List[str]:
    products: List[Any] = []
    try:
        from coinbase.rest import products as cb_products  # type: ignore

        resp = cb_products.get_products(coinbase_service.client)
        if isinstance(resp, dict):
            products = resp.get("products", []) or []
        else:
            products = getattr(resp, "products", []) or []
    except Exception as exc:  # pragma: no cover - network/API failure
        logger.warning("Failed to fetch Coinbase products via REST API: %s", exc)
        products = []

    if not products:
        logger.warning("Coinbase product discovery returned no entries.")
        return []

    quotes_set = set(q.upper() for q in quotes if q) if quotes else None
    entries: List[Dict[str, Any]] = []
    for prod in products:
        product_id = str(_pget(prod, "product_id") or "")
        if not product_id or "-" not in product_id:
            continue
        status = str(_pget(prod, "status") or "").lower()
        trading_disabled = _pget(prod, "trading_disabled")
        cancel_only = _pget(prod, "cancel_only")
        if status and status != "online":
            continue
        if trading_disabled is True or cancel_only is True:
            continue
        quote = str(
            _pget(prod, "quote_currency")
            or _pget(prod, "quote_currency_id")
            or ""
        ).upper()
        if quotes_set and quote not in quotes_set:
            continue
        try:
            volume = float(_pget(prod, "volume_24h") or 0.0)
        except (TypeError, ValueError):
            volume = 0.0
        if volume < min_volume:
            continue
        entries.append(
            {
                "product_id": product_id,
                "quote": quote,
                "volume": volume,
            }
        )

    if not entries:
        logger.warning(
            "No Coinbase products met volume/quote filters (quotes=%s, min_volume=%s).",
            quotes or "ANY",
            min_volume,
        )
        return []

    entries.sort(key=lambda row: row["volume"], reverse=True)
    if max_products:
        entries = entries[:max_products]

    logger.info(
        "Discovered %s Coinbase products after filtering (quotes=%s, min_volume=%.0f).",
        len(entries),
        ",".join(quotes) if quotes else "ANY",
        min_volume,
    )
    return [row["product_id"] for row in entries]


def _parse_coinbase_timestamp(value: object) -> Optional[datetime]:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except ValueError:
            try:
                iso = value.replace("Z", "+00:00")
                dt = datetime.fromisoformat(iso)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                return None
    return None


def fetch_coinbase_ohlcv(
    historical_data: HistoricalData,
    product_id: str,
    lookback: int,
    granularity: str,
    bar_hours: float,
    force_refresh: bool,
) -> Optional[pd.DataFrame]:
    end_time = datetime.now(tz=timezone.utc)
    if lookback:
        span_hours = max(bar_hours * lookback, bar_hours * 5)
        start_time = end_time - timedelta(hours=span_hours)
    else:
        start_time = end_time - timedelta(days=365)

    candles = historical_data.get_historical_data(
        product_id=product_id,
        start_date=start_time,
        end_date=end_time,
        granularity=granularity,
        force_refresh=force_refresh,
    )
    if not candles:
        logger.warning("No Coinbase candles returned for %s (%s)", product_id, granularity)
        return None

    records: List[Dict[str, float]] = []
    for candle in candles:
        ts = _parse_coinbase_timestamp(candle.get("start") or candle.get("time"))
        if ts is None:
            continue
        try:
            record = {
                "timestamp": ts,
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": float(candle.get("volume", 0.0)),
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Skipping malformed candle for %s: %s", product_id, exc)
            continue
        records.append(record)

    if not records:
        logger.warning("No valid Coinbase records parsed for %s", product_id)
        return None

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="timestamp").dropna()
    df = df.sort_values("timestamp").set_index("timestamp")

    if lookback:
        df = df.tail(lookback)

    logger.info("Loaded %s Coinbase candles for %s (%s)", len(df), product_id, granularity)
    return df


def compute_indicators(df: pd.DataFrame, rsi_period: int = 14, atr_period: int = 14) -> pd.DataFrame:
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - close.shift(1)).abs()
    low_close = (df["low"] - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / atr_period, min_periods=atr_period, adjust=False).mean()
    atr_pct = (atr / close).replace([np.inf, -np.inf], np.nan)

    ema_fast = close.ewm(span=21, adjust=False).mean()
    ema_slope = ema_fast.diff()

    rel_volume = df["volume"] / df["volume"].rolling(window=20, min_periods=5).mean()

    log_return = np.log(close / close.shift(1))

    out = df.copy()
    out["rsi"] = rsi
    out["atr_pct"] = atr_pct
    out["ema_slope"] = ema_slope
    out["relative_volume"] = rel_volume
    out["log_return"] = log_return
    out["target_return"] = log_return.shift(-1)
    out = out.dropna()
    return out


def signal_from_prediction(prediction: float, threshold: float) -> int:
    if prediction > threshold:
        return 1
    if prediction < -threshold:
        return -1
    return 0


@dataclass
class CoinReadoutHistory:
    timestamps: pd.Index
    predicted_returns: np.ndarray
    actual_returns: np.ndarray
    log_returns: np.ndarray
    ema_slope: np.ndarray
    relative_volume: np.ndarray
    atr_pct: np.ndarray


class SharedReservoir:
    def __init__(
        self,
        n_inputs: int,
        n_reservoir: int = 1000,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        leak_rate: float = 0.3,
        seed: int = 1337,
    ) -> None:
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        rng = np.random.default_rng(seed)
        self._win = (rng.random((n_reservoir, n_inputs)) - 0.5) * 2 * input_scaling
        w = (rng.random((n_reservoir, n_reservoir)) - 0.5) * 2
        eigenvalues = np.linalg.eigvals(w)
        current_radius = max(abs(eigenvalues))
        if current_radius == 0:
            raise ValueError("Reservoir matrix has zero spectral radius.")
        self._w = w * (spectral_radius / current_radius)

    def compute_states(
        self,
        inputs: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if inputs.ndim != 2 or inputs.shape[1] != self.n_inputs:
            raise ValueError(f"Expected inputs of shape (n_samples, {self.n_inputs})")
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        state = np.zeros(self.n_reservoir) if initial_state is None else initial_state.copy()
        for t, u in enumerate(inputs):
            pre_activation = self._win @ u + self._w @ state
            state = (1.0 - self.leak_rate) * state + self.leak_rate * np.tanh(pre_activation)
            states[t] = state
        return states, state


def train_readout_for_coin(
    symbol: str,
    reservoir: SharedReservoir,
    df_scaled: pd.DataFrame,
    df_raw: pd.DataFrame,
    feature_cols: List[str],
    threshold: float,
    short_threshold: float,
    ridge_alpha: float,
    washout: int,
    timeframe_hours: float,
) -> Optional[Tuple[Dict[str, object], CoinReadoutHistory]]:
    if len(df_scaled) < washout + 60:
        logger.warning("%s insufficient history (%s rows)", symbol, len(df_scaled))
        return None

    step = max(1, int(round(EXPIRY_HOURS / max(timeframe_hours, 1e-6))))
    predicted_returns, actual_returns = walk_forward_train_predict(
        df_raw=df_raw,
        feature_cols=feature_cols,
        reservoir=reservoir,
        ridge_alpha=ridge_alpha,
        washout=washout,
        step=step,
    )

    valid_idx = np.where(~np.isnan(predicted_returns))[0]
    if valid_idx.size == 0:
        logger.warning("%s produced no walk-forward predictions", symbol)
        return None

    latest_idx = int(valid_idx[-1])
    final_prediction = float(predicted_returns[latest_idx])
    latest_row = df_raw.iloc[latest_idx]
    ema_slope = float(latest_row["ema_slope"])
    rel_vol = float(latest_row["relative_volume"])
    atr_pct = float(latest_row["atr_pct"])
    latest_close = float(latest_row["close"])
    latest_rsi = float(latest_row["rsi"])
    timestamp = df_raw.index[latest_idx].to_pydatetime()

    final_signal = signal_from_prediction_side(final_prediction, threshold, short_threshold)
    if not regime_gate(final_signal, ema_slope, rel_vol, atr_pct):
        final_signal = 0

    uptick_tp, downtick_tp = calibrate_tp_sl(df_raw, timeframe_hours=timeframe_hours)
    if final_signal > 0:
        tp_pct = min(2.5 * atr_pct, uptick_tp)
        sl_pct = max(0.6 * atr_pct, 0.5 * tp_pct)
    elif final_signal < 0:
        tp_pct = min(2.5 * atr_pct, downtick_tp)
        sl_pct = max(0.6 * atr_pct, 0.5 * tp_pct)
    else:
        tp_pct = max(atr_pct * 1.5, 0.005)
        sl_pct = max(0.6 * atr_pct, 0.5 * tp_pct)

    result = {
        "timestamp": timestamp.isoformat(),
        "coin": symbol,
        "signal": final_signal,
        "tp_pct": float(tp_pct),
        "sl_pct": float(sl_pct),
        "expiry_h": EXPIRY_HOURS,
        "predicted_return": float(final_prediction),
        "last_price": latest_close,
        "atr_pct": atr_pct,
        "rsi": latest_rsi,
        "relative_volume": rel_vol,
        "perp_product_id": spot_to_perp_id(symbol),
    }

    history = CoinReadoutHistory(
        timestamps=df_raw.index,
        predicted_returns=predicted_returns,
        actual_returns=actual_returns,
        log_returns=df_raw["log_return"].to_numpy(),
        ema_slope=df_raw["ema_slope"].to_numpy(),
        relative_volume=df_raw["relative_volume"].to_numpy(),
        atr_pct=df_raw["atr_pct"].to_numpy(),
    )
    return result, history


def evaluate_signals(
    histories: Dict[str, CoinReadoutHistory],
    threshold: float,
    short_threshold: float,
    timeframe_hours: float,
) -> pd.DataFrame:
    rows = []
    for symbol, history in histories.items():
        metrics = oos_metrics(history, threshold, short_threshold, timeframe_hours)
        metrics["coin"] = symbol
        rows.append(metrics)
    return pd.DataFrame(rows)


def _safe_float(value: object) -> float:
    try:
        val = float(value)
        if math.isfinite(val):
            return val
    except Exception:
        pass
    return float("nan")


def _format_usd(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    if abs(value) >= 1:
        return f"${value:,.2f}"
    return f"${value:,.6f}"


def _format_percent(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def fit_scaler_past_only(df: pd.DataFrame, feature_cols: List[str], cutoff_idx: int) -> StandardScaler:
    scaler = StandardScaler()
    cutoff = max(1, int(cutoff_idx))
    scaler.fit(df[feature_cols].iloc[:cutoff].values)
    return scaler


def walk_forward_train_predict(
    df_raw: pd.DataFrame,
    feature_cols: List[str],
    reservoir: SharedReservoir,
    ridge_alpha: float,
    washout: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_rows = len(df_raw)
    preds = np.full(n_rows, np.nan, dtype=float)
    targets = df_raw["target_return"].to_numpy(copy=True)
    if n_rows < washout + 60:
        return preds, targets

    for end in range(washout + 50, n_rows - 1, step):
        scaler = fit_scaler_past_only(df_raw, feature_cols, end)
        scaled_values = scaler.transform(df_raw[feature_cols].values)
        states, final_state = reservoir.compute_states(scaled_values[:end], initial_state=None)
        y = targets[:end]
        valid = ~np.isnan(y)
        idx = np.where(valid)[0]
        idx = idx[idx >= washout]
        if idx.size < 5:
            continue

        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        model.fit(states[idx], y[idx])

        future_end = min(end + step, n_rows)
        future_inputs = scaled_values[end:future_end]
        future_states, _ = reservoir.compute_states(future_inputs, initial_state=final_state)
        preds[end:future_end] = model.predict(future_states)

    return preds, targets


def signal_from_prediction_side(pred: float, thr_long: float, thr_short: float) -> int:
    if not math.isfinite(pred):
        return 0
    if pred > thr_long:
        return 1
    if pred < -thr_short:
        return -1
    return 0


def regime_gate(side: int, ema_slope: float, rel_vol: float, atr_pct: float) -> bool:
    if side == 0:
        return False
    if atr_pct < 0.002:
        return False
    if side < 0 and (ema_slope > 0 or rel_vol < 1.0):
        return False
    return True


def calibrate_tp_sl(
    df: pd.DataFrame,
    horizon: int = EXPIRY_HOURS,
    timeframe_hours: float = 1.0,
    quantile: float = 0.6,
) -> Tuple[float, float]:
    k = max(1, int(round(horizon / max(timeframe_hours, 1e-6))))
    forward_high = df["high"].rolling(k).max().shift(-k)
    forward_low = df["low"].rolling(k).min().shift(-k)
    close = df["close"]

    up_moves = (forward_high / close - 1.0).dropna()
    down_moves = ((close / forward_low) - 1.0).dropna()

    up_q = float(np.nanquantile(up_moves, quantile)) if not up_moves.empty else 0.01
    down_q = float(np.nanquantile(down_moves, quantile)) if not down_moves.empty else 0.01
    return max(0.005, up_q), max(0.005, down_q)


def oos_metrics(
    history: CoinReadoutHistory,
    thr_long: float,
    thr_short: float,
    timeframe_hours: float,
) -> Dict[str, float]:
    preds = history.predicted_returns
    signals = np.zeros_like(preds, dtype=int)
    valid = ~np.isnan(preds)
    signals[valid & (preds > thr_long)] = 1
    signals[valid & (preds < -thr_short)] = -1

    for idx in np.where(signals != 0)[0]:
        if not regime_gate(
            signals[idx],
            history.ema_slope[idx],
            history.relative_volume[idx],
            history.atr_pct[idx],
        ):
            signals[idx] = 0

    k = max(1, int(round(EXPIRY_HOURS / max(timeframe_hours, 1e-6))))
    log_returns = history.log_returns
    future_sum = pd.Series(log_returns).rolling(window=k, min_periods=k).sum().shift(-k + 1)
    future_arr = future_sum.to_numpy()

    active_mask = (signals != 0) & ~np.isnan(future_arr)
    if not np.any(active_mask):
        return {"n": 0, "win": float("nan"), "mean": float("nan"), "std": float("nan")}

    realised_side = future_arr[active_mask] * signals[active_mask]
    win_rate = float((realised_side > 0).mean()) if realised_side.size else float("nan")
    return {
        "n": int(active_mask.sum()),
        "win": win_rate,
        "mean": float(np.nanmean(realised_side)) if realised_side.size else float("nan"),
        "std": float(np.nanstd(realised_side)) if realised_side.size else float("nan"),
    }


def _prepare_llm_candidates(df: pd.DataFrame, threshold: float) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if df.empty:
        return candidates

    safe_threshold = max(abs(threshold), 1e-6)

    for _, row in df.iterrows():
        coin = str(row.get("coin", "") or "")
        if not coin:
            continue
        predicted_return = _safe_float(row.get("predicted_return"))
        atr_pct = _safe_float(row.get("atr_pct"))
        rsi = _safe_float(row.get("rsi"))
        rel_vol = _safe_float(row.get("relative_volume"))
        tp_pct = _safe_float(row.get("tp_pct"))
        sl_pct = _safe_float(row.get("sl_pct"))
        signal = int(row.get("signal", 0) or 0)
        side = "LONG" if signal > 0 else "SHORT" if signal < 0 else "FLAT"

        base_score = min(100.0, max(0.0, abs(predicted_return) / safe_threshold * 50.0))

        candidates.append(
            {
                "candidate_id": coin,
                "symbol": coin,
                "position_side": side,
                "base_score": base_score,
                "predicted_return_pct": predicted_return * 100.0,
                "atr_pct": atr_pct * 100.0,
                "rsi": rsi,
                "relative_volume": rel_vol,
                "tp_pct": tp_pct * 100.0,
                "sl_pct": sl_pct * 100.0,
                "threshold_pct": safe_threshold * 100.0,
            }
        )

    return candidates


def prefer_usdc_products(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate results by base asset, preferring USDC-quoted products when available."""
    if df.empty or "coin" not in df.columns:
        return df
    df = df.copy()
    df["_order_idx"] = np.arange(len(df))
    split = df["coin"].str.split("-", n=1, expand=True)
    df["_base"] = split[0]
    df["_quote"] = split[1].fillna("")

    keep_indices: List[int] = []
    selection: Dict[str, Tuple[int, str]] = {}
    for _, row in df.iterrows():
        base = row["_base"]
        quote = row["_quote"]
        idx = int(row["_order_idx"])
        if base not in selection:
            selection[base] = (idx, quote)
            keep_indices.append(idx)
            continue
        prev_idx, prev_quote = selection[base]
        if quote == "USDC" and prev_quote != "USDC":
            if prev_idx in keep_indices:
                keep_indices.remove(prev_idx)
            keep_indices.append(idx)
            selection[base] = (idx, quote)

    kept = df[df["_order_idx"].isin(keep_indices)].copy()
    kept = kept.sort_values("_order_idx").drop(columns=["_order_idx", "_base", "_quote"])
    return kept.reset_index(drop=True)


def _save_plain_report(path: Path, content: str) -> None:
    tmp_path = Path(f"{path}.tmp.{os.getpid()}.{int(datetime.now().timestamp() * 1000)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(tmp_path, path)
    logger.info("Plain finder report saved to %s", path)


def build_plain_report(
    results_df: pd.DataFrame,
    raw_feature_frames: Dict[str, pd.DataFrame],
    threshold: float,
    timeframe: str,
    expiry_hours: int = EXPIRY_HOURS,
    recommended_position_pct: float = 5.0,
    compact: bool = False,
) -> str:
    lines: List[str] = []
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    active = results_df[results_df["signal"] != 0].copy()

    if compact:
        lines.append(
            f"{len(active)} signal(s) | threshold={threshold:.4f} | timeframe={timeframe} | expiry={expiry_hours}h | generated_utc={generated_utc}"
        )
    else:
        header_rule = "=" * 100
        lines.append(header_rule)
        lines.append("MULTI-COIN RESERVOIR DAYTRADER SIGNALS")
        lines.append(header_rule)
        lines.append(f"Generated on (UTC): {generated_utc}")
        lines.append(f"Signal threshold: {threshold:.4f} | Timeframe: {timeframe} | Expiry: {expiry_hours}h")
        lines.append(f"Total opportunities listed: {len(active)}")
        lines.append(header_rule)
        lines.append("")

    if active.empty:
        msg = "No actionable signals crossed the threshold today. Consider lowering the threshold or extending lookback."
        lines.append(msg)
        return "\n".join(lines) + "\n"

    active.reset_index(drop=True, inplace=True)

    for idx, row in active.iterrows():
        product_id = str(row["coin"])
        raw_base = product_id.split("-")[0]
        display_symbol = canonical_perp_symbol(raw_base)
        side = "LONG" if int(row["signal"]) > 0 else "SHORT"
        raw_df = raw_feature_frames.get(product_id)
        if raw_df is None or raw_df.empty:
            logger.warning("Missing raw feature frame for %s; skipping plain report block", product_id)
            continue
        latest = raw_df.iloc[-1]
        price = _safe_float(row.get("last_price", latest.get("close")))
        atr_pct = _safe_float(row.get("atr_pct", latest.get("atr_pct")))
        rsi_val = _safe_float(row.get("rsi", latest.get("rsi")))
        rel_vol = _safe_float(row.get("relative_volume", latest.get("relative_volume")))
        predicted_return = _safe_float(row.get("predicted_return"))
        tp_pct = max(_safe_float(row.get("tp_pct")), 0.0)
        sl_pct = max(_safe_float(row.get("sl_pct")), 0.0)
        timestamp_iso = str(row.get("timestamp", ""))
        try:
            data_timestamp = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%SZ")
        except Exception:
            data_timestamp = timestamp_iso or "N/A"

        if side == "LONG":
            take_profit_price = price * (1 + tp_pct)
            stop_loss_price = price * (1 - sl_pct)
        else:
            take_profit_price = price * (1 - tp_pct)
            stop_loss_price = price * (1 + sl_pct)

        multiplier = perp_price_multiplier(raw_base)
        display_price = price * multiplier
        display_tp = take_profit_price * multiplier
        display_sl = stop_loss_price * multiplier

        pred_str = _format_percent(predicted_return)
        atr_str = _format_percent(atr_pct)
        rel_vol_str = f"{rel_vol:.2f}" if math.isfinite(rel_vol) else "N/A"

        rr_ratio = "N/A"
        if sl_pct > 0 and math.isfinite(tp_pct):
            rr_ratio_value = tp_pct / sl_pct if sl_pct else float("inf")
            if math.isfinite(rr_ratio_value):
                rr_ratio = f"{rr_ratio_value:.2f}:1"

        if compact:
            lines.append(
                f"{idx + 1}. {display_symbol:<12} {side:<5} entry={display_price:.6f} tp={display_tp:.6f} sl={display_sl:.6f} "
                f"pred={pred_str} atr={atr_str} rv={rel_vol_str} "
                f"timestamp={data_timestamp}"
                + (
                    f" llm={row['llm_score']:.1f}/{row.get('llm_confidence', 'N/A')}"
                    if "llm_score" in row and math.isfinite(_safe_float(row.get("llm_score")))
                    else ""
                )
            )
        else:
            lines.append(f"{idx + 1}. {display_symbol} ({product_id}) — {side}")
            lines.append("-" * 50)
            lines.append(f"Data Timestamp (UTC): {data_timestamp}")
            lines.append(f"Price: {_format_usd(display_price)}")
            lines.append(f"Predicted Return (next period): {pred_str}")
            lines.append(f"RSI: {rsi_val:.2f}" if math.isfinite(rsi_val) else "RSI: N/A")
            lines.append(f"ATR %: {atr_str}" if math.isfinite(atr_pct) else "ATR %: N/A")
            lines.append(f"Relative Volume: {rel_vol_str}")
            lines.append("")
            lines.append(f"💼 TRADING LEVELS ({side}):")
            lines.append(f"Entry Price: {_format_usd(display_price)}")
            lines.append(f"Stop Loss: {_format_usd(display_sl)}")
            lines.append(f"Take Profit: {_format_usd(display_tp)}")
            lines.append(f"Risk:Reward Ratio: {rr_ratio}")
            lines.append(f"Recommended Position Size: {recommended_position_pct:.1f}% of portfolio")
            lines.append(f"Take-Profit Distance: {_format_percent(tp_pct)} | Stop-Loss Distance: {_format_percent(sl_pct)}")
            lines.append(f"Signal Expires In: {expiry_hours} hours")
            if "llm_score" in row and math.isfinite(_safe_float(row.get("llm_score"))):
                confidence = row.get("llm_confidence") or "N/A"
                lines.append(f"LLM Score: {row['llm_score']:.2f} (confidence: {confidence})")
                reason = row.get("llm_reason")
                if isinstance(reason, str) and reason.strip():
                    lines.append(f"LLM Insight: {reason.strip()}")
            lines.append("")

    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 24h day-trade signals with a shared reservoir model.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Explicit Coinbase product IDs to evaluate (skip automatic discovery).",
    )
    parser.add_argument(
        "--timeframe",
        default=DEFAULT_TIMEFRAME,
        help="Coinbase candle granularity or alias (e.g. ONE_HOUR, 1h, 15m).",
    )
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="Number of candles to use per product.")
    parser.add_argument(
        "--profile",
        choices=sorted(RESERVOIR_PROFILES),
        default="default",
        help="Discovery preset for Coinbase product list.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Display profile presets and exit.",
    )
    parser.add_argument(
        "--quotes",
        type=str,
        help="Comma-separated quote currencies for discovery (e.g., USDC,USD).",
    )
    parser.add_argument(
        "--max-products",
        type=int,
        help="Maximum number of products to evaluate (after filtering).",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        help="Minimum 24h volume (in quote currency units) required for inclusion.",
    )
    parser.add_argument("--reservoir-size", type=int, default=400, help="Number of reservoir units.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Reservoir leak rate.")
    parser.add_argument("--spectral-radius", type=float, default=0.9, help="Reservoir spectral radius.")
    parser.add_argument("--input-scaling", type=float, default=0.1, help="Reservoir input scaling.")
    parser.add_argument("--threshold", type=float, default=0.003, help="Return threshold for long/short signals.")
    parser.add_argument("--short-threshold", type=float, help="Return threshold for short signals (default 1.5x long threshold).")
    parser.add_argument("--ridge-alpha", type=float, default=5e-2, help="Ridge regression strength.")
    parser.add_argument("--washout", type=int, help="Discarded initial states per coin (auto if omitted).")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reservoir initialisation.")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass cached Coinbase candles when fetching OHLCV data.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("signals") / "multi_coin_reservoir_daytrader_signals.csv",
        help="Target CSV path for execution engine.",
    )
    parser.add_argument(
        "--plain-output",
        type=Path,
        default=Path("finder_short.txt"),
        help="Finder-style plain-text output for add_position_from_finder.py.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("MC_RESERVOIR_LOG_LEVEL", "INFO"),
        help="Logging level.",
    )
    parser.add_argument(
        "--use-llm-scoring",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Blend reservoir ranking with an OpenAI LLM opinion.",
    )
    parser.add_argument(
        "--llm-weight",
        type=float,
        help="Weight (0-1) applied when combining reservoir and LLM scores.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        help="OpenAI model name for LLM scoring (default: profile setting or gpt-5-mini).",
    )
    parser.add_argument(
        "--llm-max-candidates",
        type=int,
        help="Maximum number of candidates to score with the LLM.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        help="Optional temperature for the LLM call (<=0 disables).",
    )
    parser.add_argument(
        "--llm-sleep-seconds",
        type=float,
        help="Optional sleep between LLM calls (seconds).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    if args.list_profiles:
        print(profile_summary())
        return 0

    try:
        granularity = normalize_granularity(args.timeframe)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    args.timeframe = granularity
    timeframe_hours = timeframe_to_hours(granularity)

    api_key, api_secret = _seed_credentials()
    if not api_key or not api_secret:
        logger.warning("Coinbase API credentials not found in environment; attempting unauthenticated access.")
    coinbase_service = CoinbaseService(api_key, api_secret)
    historical_data = HistoricalData(coinbase_service.client)
    historical_data.set_force_refresh(bool(args.force_refresh))

    profile_cfg = RESERVOIR_PROFILES.get(args.profile, {})
    quotes = _parse_quotes(args.quotes, profile_cfg.get("quotes", []))
    max_products = args.max_products if args.max_products is not None else profile_cfg.get("max_products")
    min_volume = args.min_volume if args.min_volume is not None else float(profile_cfg.get("min_volume", 0.0))
    if args.washout is None:
        args.washout = int(max(60, 3.0 / max(args.alpha, 1e-6)))
    short_threshold = args.short_threshold if args.short_threshold is not None else max(args.threshold * 1.5, args.threshold + 0.001)
    use_llm_scoring = (
        bool(profile_cfg.get("use_llm_scoring", False))
        if args.use_llm_scoring is None
        else bool(args.use_llm_scoring)
    )
    llm_weight = args.llm_weight if args.llm_weight is not None else float(profile_cfg.get("llm_weight", 0.25))
    llm_model = args.llm_model or profile_cfg.get("llm_model", "gpt-5-mini")
    llm_max_candidates = (
        args.llm_max_candidates if args.llm_max_candidates is not None else int(profile_cfg.get("llm_max_candidates", 12))
    )
    llm_temperature: Optional[float]
    if args.llm_temperature is not None:
        llm_temperature = args.llm_temperature
    else:
        llm_temperature = profile_cfg.get("llm_temperature")
    llm_sleep_seconds = (
        args.llm_sleep_seconds if args.llm_sleep_seconds is not None else float(profile_cfg.get("llm_sleep_seconds", 0.0))
    )

    llm_scorer: Optional[LLMScorer] = None
    if use_llm_scoring:
        try:
            llm_scorer = LLMScorer(
                model=str(llm_model),
                weight=float(np.clip(llm_weight, 0.0, 1.0)),
                max_candidates=int(max(1, llm_max_candidates)),
                temperature=llm_temperature,
                sleep_seconds=float(max(0.0, llm_sleep_seconds)),
            )
            if not getattr(llm_scorer, "enabled", False):
                logger.warning("LLM scoring disabled: OpenAI client unavailable.")
                llm_scorer = None
            else:
                logger.info("LLM scoring enabled (model=%s, weight=%.2f).", llm_scorer.model, llm_scorer.weight)
        except Exception as exc:
            logger.warning("Failed to initialise LLM scoring: %s", exc)
            llm_scorer = None

    os.makedirs(args.output_csv.parent, exist_ok=True)

    if args.symbols:
        selected_products = [normalize_product_id(s) for s in args.symbols]
    else:
        selected_products = discover_product_ids(
            coinbase_service=coinbase_service,
            quotes=quotes,
            max_products=max_products,
            min_volume=min_volume,
        )
        if not selected_products:
            selected_products = [normalize_product_id(s) for s in DEFAULT_SYMBOLS]
            logger.warning(
                "Falling back to default product list because discovery returned nothing: %s",
                ", ".join(selected_products),
            )

    selected_products = list(dict.fromkeys(selected_products))  # preserve order, drop duplicates
    logger.info(
        "Evaluating %s Coinbase product(s) using profile '%s'.",
        len(selected_products),
        args.profile,
    )

    raw_frames: Dict[str, pd.DataFrame] = {}
    for product_id in selected_products:
        df = fetch_coinbase_ohlcv(
            historical_data=historical_data,
            product_id=product_id,
            lookback=args.lookback,
            granularity=granularity,
            bar_hours=timeframe_hours,
            force_refresh=bool(args.force_refresh),
        )
        if df is None or df.empty:
            logger.warning("Skipping %s due to missing Coinbase data", product_id)
            continue
        raw_frames[product_id] = df

    if not raw_frames:
        logger.error("No market data loaded; exiting.")
        return 1

    feature_frames_raw: Dict[str, pd.DataFrame] = {}
    for symbol, frame in raw_frames.items():
        feature_df = compute_indicators(frame)
        if feature_df.empty:
            logger.warning("%s produced empty feature set", symbol)
            continue
        feature_frames_raw[symbol] = feature_df

    if not feature_frames_raw:
        logger.error("No features computed; exiting.")
        return 1

    feature_cols = ["rsi", "atr_pct", "ema_slope", "relative_volume", "log_return"]
    scaler = StandardScaler()
    stacked_features = np.vstack([df[feature_cols].values for df in feature_frames_raw.values()])
    scaler.fit(stacked_features)

    feature_frames_scaled: Dict[str, pd.DataFrame] = {}
    for symbol, df in feature_frames_raw.items():
        scaled_df = df.copy()
        scaled_df[feature_cols] = scaler.transform(df[feature_cols].values)
        feature_frames_scaled[symbol] = scaled_df

    reservoir = SharedReservoir(
        n_inputs=len(feature_cols),
        n_reservoir=args.reservoir_size,
        spectral_radius=args.spectral_radius,
        input_scaling=args.input_scaling,
        leak_rate=args.alpha,
        seed=args.seed,
    )

    results: List[Dict[str, object]] = []
    histories: Dict[str, CoinReadoutHistory] = {}
    skipped_perps: List[str] = []

    for symbol, df_scaled in feature_frames_scaled.items():
        df_raw = feature_frames_raw[symbol]
        trained = train_readout_for_coin(
            symbol=symbol,
            reservoir=reservoir,
            df_scaled=df_scaled,
            df_raw=df_raw,
            feature_cols=feature_cols,
            threshold=args.threshold,
            short_threshold=short_threshold,
            ridge_alpha=args.ridge_alpha,
            washout=args.washout,
            timeframe_hours=timeframe_hours,
        )
        if trained is None:
            logger.warning("Skipping %s due to training failure", symbol)
            continue
        result, history = trained
        perp_id = result.get("perp_product_id")
        if perp_id and not is_perp_supported(perp_id, coinbase_service, logger=logger):
            skipped_perps.append(f"{symbol}→{perp_id}")
            continue
        results.append(result)
        histories[symbol] = history
        logger.info(
            "%s prediction %.5f signal %s tp %.4f sl %.4f",
            symbol,
            result["predicted_return"],
            result["signal"],
            result["tp_pct"],
            result["sl_pct"],
        )

    if not results:
        logger.error("No signals generated.")
        return 1
    if skipped_perps:
        logger.info("Skipped unsupported perps: %s", ", ".join(skipped_perps))

    results_df = pd.DataFrame(results)

    base_scores: Dict[str, float] = {}
    if not results_df.empty:
        candidates = _prepare_llm_candidates(results_df, args.threshold)
        for cand in candidates:
            base_scores[str(cand.get("candidate_id"))] = float(cand.get("base_score", 0.0))
        if base_scores:
            results_df["base_score"] = results_df["coin"].map(base_scores).fillna(0.0)
        if llm_scorer and candidates:
            llm_results = llm_scorer.score_candidates(candidates)
            if llm_results:
                llm_rows = []
                for idx, row in results_df.iterrows():
                    coin = str(row.get("coin", "") or "")
                    base = base_scores.get(coin, 0.0)
                    outcome = llm_results.get(coin)
                    if outcome:
                        llm_score = float(outcome.get("llm_score", base))
                        combined = llm_scorer.combine_scores(base, llm_score)
                        results_df.at[idx, "llm_score"] = llm_score
                        results_df.at[idx, "llm_confidence"] = outcome.get("confidence", "")
                        results_df.at[idx, "llm_reason"] = outcome.get("reason", "")
                        results_df.at[idx, "combined_score"] = combined
                    else:
                        results_df.at[idx, "combined_score"] = base
                logger.info("LLM scoring refined %s candidate(s).", len(llm_results))
            else:
                logger.info("LLM scoring returned no adjustments; falling back to base ordering.")
        if "combined_score" not in results_df.columns:
            # Use base score fallback derived from predicted returns
            for idx, row in results_df.iterrows():
                coin = str(row.get("coin", "") or "")
                results_df.at[idx, "combined_score"] = base_scores.get(coin, abs(row.get("predicted_return", 0.0)))

    results_df = results_df.sort_values(
        by="combined_score",
        ascending=False,
    ).reset_index(drop=True)
    preferred_results = prefer_usdc_products(results_df)
    if len(preferred_results) != len(results_df):
        logger.info(
            "Filtered duplicate quote variants: kept %s of %s (preferring USDC).",
            len(preferred_results),
            len(results_df),
        )
    results_df = preferred_results

    ranked_path = args.output_csv.with_name(args.output_csv.stem + "_ranked.csv")
    results_df.to_csv(ranked_path, index=False)
    logger.info("Ranked diagnostics saved to %s", ranked_path)

    execution_df = results_df[["timestamp", "coin", "signal", "tp_pct", "sl_pct", "expiry_h"]]
    execution_df.to_csv(args.output_csv, index=False)
    logger.info("Execution signals saved to %s", args.output_csv)

    active_coins = set(results_df["coin"])
    filtered_histories = {k: v for k, v in histories.items() if k in active_coins}

    metrics_df = evaluate_signals(filtered_histories, args.threshold, short_threshold, timeframe_hours)
    if not metrics_df.empty:
        metrics_path = args.output_csv.with_name(args.output_csv.stem + "_evaluation.csv")
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("Evaluation metrics saved to %s", metrics_path)
    else:
        logger.info("Insufficient history to compute evaluation metrics.")

    compact_report = build_plain_report(
        results_df=results_df,
        raw_feature_frames=feature_frames_raw,
        threshold=args.threshold,
        timeframe=args.timeframe,
        expiry_hours=EXPIRY_HOURS,
        compact=True,
    )
    if compact_report:
        print(compact_report, end="")

    if args.plain_output:
        detailed_report = build_plain_report(
            results_df=results_df,
            raw_feature_frames=feature_frames_raw,
            threshold=args.threshold,
            timeframe=args.timeframe,
            expiry_hours=EXPIRY_HOURS,
            compact=False,
        )
        _save_plain_report(Path(args.plain_output), detailed_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
