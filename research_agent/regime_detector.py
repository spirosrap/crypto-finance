from __future__ import annotations

import csv
import json
import os
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

from research_agent.coinbase_api import (
    Candle,
    coerce_float,
    get_coinbase_candles,
    get_coinbase_stats,
    normalize_candles,
)


DEFAULT_EMA_PERIOD = 20
DEFAULT_NEUTRAL_BAND_PCT = 2.0
DEFAULT_ATR_PERIOD = 7
DEFAULT_ATR_MODE = (os.getenv("BASELINE_ATR_MODE", "clipped") or "clipped").strip().lower()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


DEFAULT_MAX_ATR_USD = _env_float("SHORT_MAX_ATR_USD", 3000.0)
DEFAULT_MAX_ATR_BPS = _env_float("SHORT_MAX_ATR_BPS", 400.0)
DEFAULT_SETUP_SL_ATR_MULT = _env_float("BASELINE_SETUP_SL_ATR_MULT", 0.8)
DEFAULT_SETUP_TP1_RR = _env_float("BASELINE_PARTIAL_TP_RR", 0.8)
DEFAULT_SETUP_TP2_RR = _env_float("BASELINE_TP2_RR", 1.5)
DEFAULT_SETUP_ENTRY_BUFFER_PCT = _env_float("BASELINE_SETUP_ENTRY_BUFFER_PCT", 0.3)


def calculate_ema(prices: Sequence[float], period: int) -> float:
    """Calculate EMA using a simple iterative method seeded with the first price.

    This intentionally matches the spec-provided approach (seed = first element).
    """

    if period <= 0:
        raise ValueError("period must be positive")
    if not prices:
        raise ValueError("prices cannot be empty")
    multiplier = 2.0 / (float(period) + 1.0)
    ema = float(prices[0])
    for price in prices[1:]:
        ema = (float(price) - ema) * multiplier + ema
    return float(ema)


def _dynamic_atr_bps(price: float) -> float:
    if price >= 20000:
        return 325.0
    if price >= 2000:
        return 350.0
    if price >= 200:
        return 400.0
    return 450.0


def _effective_atr_cap_usd(
    price: float,
    *,
    max_atr_usd: Optional[float],
    max_atr_bps: Optional[float],
) -> Optional[float]:
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


def calculate_atr_wilder(candles: Sequence[Candle], period: int = DEFAULT_ATR_PERIOD) -> float:
    """ATR using Wilder smoothing from normalized candle data."""

    if period <= 0:
        raise ValueError("period must be positive")
    if len(candles) < period + 1:
        return 0.0

    prev_close = float(candles[0].close)
    true_ranges: List[float] = []
    for candle in candles[1:]:
        high = float(candle.high)
        low = float(candle.low)
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(float(tr))
        prev_close = float(candle.close)

    if len(true_ranges) < period:
        return 0.0

    atr = sum(true_ranges[:period]) / float(period)
    for tr in true_ranges[period:]:
        atr = ((atr * float(period - 1)) + float(tr)) / float(period)
    return max(0.0, float(atr))


def _format_regime(distance_pct: float, neutral_band_pct: float) -> Tuple[str, str]:
    if distance_pct > neutral_band_pct:
        return "BULLISH", "70% long / 30% short"
    if distance_pct < -neutral_band_pct:
        return "BEARISH", "70% short / 30% long"
    return "NEUTRAL", "50/50 or wait"


def _slice_last_closes(candles: List[Candle], period: int) -> List[float]:
    if len(candles) < period:
        raise ValueError(f"Need at least {period} candles to compute EMA (got {len(candles)})")
    closes = [float(c.close) for c in candles[-period:]]
    return closes


def get_regime_status(
    product_id: str = "BTC-USD",
    *,
    ema_period: int = DEFAULT_EMA_PERIOD,
    neutral_band_pct: float = DEFAULT_NEUTRAL_BAND_PCT,
    candles_granularity: int = 86400,
    atr_period: int = DEFAULT_ATR_PERIOD,
    atr_mode: str = DEFAULT_ATR_MODE,
    max_atr_usd: Optional[float] = DEFAULT_MAX_ATR_USD,
    max_atr_bps: Optional[float] = DEFAULT_MAX_ATR_BPS,
) -> Dict[str, Any]:
    """Return regime status based on price vs daily EMA."""

    raw_candles = get_coinbase_candles(product_id, granularity=candles_granularity)
    candles = normalize_candles(raw_candles)
    closes = _slice_last_closes(candles, int(ema_period))
    ema_20 = calculate_ema(closes, int(ema_period))

    stats = get_coinbase_stats(product_id)
    current_price = coerce_float(stats.get("last"))
    if current_price is None:
        raise ValueError(f"Missing 'last' in stats for {product_id}")

    atr_raw = calculate_atr_wilder(candles, period=int(atr_period))
    atr_cap = _effective_atr_cap_usd(float(current_price), max_atr_usd=max_atr_usd, max_atr_bps=max_atr_bps)
    atr_mode_norm = (atr_mode or "raw").strip().lower()
    atr_used = float(atr_raw)
    if atr_mode_norm == "clipped" and atr_cap is not None:
        atr_used = min(float(atr_raw), float(atr_cap))

    distance_pct = ((float(current_price) - float(ema_20)) / float(ema_20)) * 100.0 if ema_20 else 0.0
    regime, recommendation = _format_regime(distance_pct, float(neutral_band_pct))

    high_24h = coerce_float(stats.get("high"))
    low_24h = coerce_float(stats.get("low"))
    open_24h = coerce_float(stats.get("open"))
    volume_24h_base = coerce_float(stats.get("volume"))
    volume_24h_usd = None
    if volume_24h_base is not None:
        volume_24h_usd = float(volume_24h_base) * float(current_price)

    return {
        "product": str(product_id).upper(),
        "current_price": float(current_price),
        "ema_20": float(round(ema_20, 2)),
        "distance_pct": float(round(distance_pct, 2)),
        "regime": regime,
        "recommendation": recommendation,
        "24h_open": float(open_24h) if open_24h is not None else None,
        "24h_high": float(high_24h) if high_24h is not None else None,
        "24h_low": float(low_24h) if low_24h is not None else None,
        "24h_volume_base": float(volume_24h_base) if volume_24h_base is not None else None,
        "24h_volume_usd": float(volume_24h_usd) if volume_24h_usd is not None else None,
        "ema_period": int(ema_period),
        "neutral_band_pct": float(neutral_band_pct),
        "atr_period": int(atr_period),
        "atr_mode": atr_mode_norm,
        "atr_raw": float(round(atr_raw, 2)),
        "atr_used": float(round(atr_used, 2)),
        "atr_cap_usd": float(round(atr_cap, 2)) if atr_cap is not None else None,
    }


def build_market_overview(
    product_ids: Sequence[str],
    *,
    ema_period: int = DEFAULT_EMA_PERIOD,
    neutral_band_pct: float = DEFAULT_NEUTRAL_BAND_PCT,
) -> "pd.DataFrame":
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for pid in product_ids:
        status = get_regime_status(
            pid,
            ema_period=ema_period,
            neutral_band_pct=neutral_band_pct,
            candles_granularity=86400,
        )
        price = float(status["current_price"])
        open_24h = status.get("24h_open")
        change_pct = None
        if open_24h not in (None, 0, 0.0):
            change_pct = (price - float(open_24h)) / float(open_24h) * 100.0

        signal = "Above EMA" if price > float(status["ema_20"]) else "Below EMA"
        rows.append(
            {
                "Asset": status["product"],
                "Price": price,
                "24h Change %": round(change_pct, 2) if change_pct is not None else None,
                "24h Volume (USD)": status.get("24h_volume_usd"),
                "Signal": signal,
                "EMA": float(status["ema_20"]),
                "Distance %": float(status["distance_pct"]),
                "Regime": status["regime"],
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Asset")
    return df


def build_key_levels(product_status: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "product": product_status.get("product"),
        "current": product_status.get("current_price"),
        "ema_20": product_status.get("ema_20"),
        "daily_open": product_status.get("24h_open"),
        "24h_high": product_status.get("24h_high"),
        "24h_low": product_status.get("24h_low"),
        "24h_volume_usd": product_status.get("24h_volume_usd"),
        "atr_used": product_status.get("atr_used"),
    }


def scan_opportunities(
    statuses: Dict[str, Dict[str, Any]],
    *,
    pullback_band_pct: float = 0.5,
    breakout_band_pct: float = 0.5,
    sl_atr_mult: float = DEFAULT_SETUP_SL_ATR_MULT,
    tp1_rr: float = DEFAULT_SETUP_TP1_RR,
    tp2_rr: float = DEFAULT_SETUP_TP2_RR,
    entry_buffer_pct: float = DEFAULT_SETUP_ENTRY_BUFFER_PCT,
) -> List[str]:
    """Create lightweight trade opportunity notes from regime + 24h levels."""

    opportunities: List[str] = []
    for pid, status in statuses.items():
        price = float(status.get("current_price") or 0.0)
        ema = float(status.get("ema_20") or 0.0)
        atr_used = float(status.get("atr_used") or status.get("atr_raw") or 0.0)
        high = status.get("24h_high")

        if ema:
            dist_to_ema_pct = ((price - ema) / ema) * 100.0
        else:
            dist_to_ema_pct = 0.0

        risk = max(0.0, atr_used * float(sl_atr_mult))
        if ema and risk > 0 and price > 0:
            entry_buf = max(0.0, float(entry_buffer_pct)) / 100.0
            long_entry = ema * (1.0 + entry_buf)
            short_entry = ema * (1.0 - entry_buf)
            long_sl = ema - risk
            long_tp1 = ema + (risk * float(tp1_rr))
            long_tp2 = ema + (risk * float(tp2_rr))
            short_sl = ema + risk
            short_tp1 = ema - (risk * float(tp1_rr))
            short_tp2 = ema - (risk * float(tp2_rr))
            entries_txt = (
                f"Entries: LONG reclaim close >= {long_entry:,.2f} "
                f"(EMA {ema:,.2f} +{entry_buffer_pct:.2f}%) | "
                f"SHORT rejection close <= {short_entry:,.2f} "
                f"(EMA {ema:,.2f} -{entry_buffer_pct:.2f}%)"
            )
            levels_txt = (
                f"SL={sl_atr_mult:.2f}xATR, TP1={tp1_rr:.2f}R, TP2={tp2_rr:.2f}R "
                f"(ATR{int(status.get('atr_period') or DEFAULT_ATR_PERIOD)} used={atr_used:,.2f})"
            )
            if abs(dist_to_ema_pct) <= pullback_band_pct:
                opportunities.append(
                    f"{pid}: At EMA zone {ema:,.2f}. Long hold/reclaim above EMA or short rejection below EMA. "
                    f"{entries_txt}. {levels_txt}. LONG SL {long_sl:,.2f} TP1 {long_tp1:,.2f} TP2 {long_tp2:,.2f} | "
                    f"SHORT SL {short_sl:,.2f} TP1 {short_tp1:,.2f} TP2 {short_tp2:,.2f}"
                )
            elif price > ema:
                opportunities.append(
                    f"{pid}: Above EMA {ema:,.2f}. Suggested LONG on pullback/hold above EMA; SHORT only on rejection back below EMA. "
                    f"{entries_txt}. {levels_txt}. LONG SL {long_sl:,.2f} TP1 {long_tp1:,.2f} TP2 {long_tp2:,.2f} | "
                    f"SHORT SL {short_sl:,.2f} TP1 {short_tp1:,.2f} TP2 {short_tp2:,.2f}"
                )
            else:
                opportunities.append(
                    f"{pid}: Below EMA {ema:,.2f}. Suggested SHORT on rejection near EMA; LONG only after reclaim close above EMA. "
                    f"{entries_txt}. {levels_txt}. SHORT SL {short_sl:,.2f} TP1 {short_tp1:,.2f} TP2 {short_tp2:,.2f} | "
                    f"LONG SL {long_sl:,.2f} TP1 {long_tp1:,.2f} TP2 {long_tp2:,.2f}"
                )
        elif ema and abs(dist_to_ema_pct) <= pullback_band_pct and price > 0:
            opportunities.append(f"{pid}: Pullback to EMA around {ema:,.2f} (in zone)")
        elif ema and price > ema:
            opportunities.append(f"{pid}: Prefer pullback entries near EMA {ema:,.2f}")
        elif ema and price < ema:
            opportunities.append(f"{pid}: Below EMA; wait for reclaim or short rejection near EMA {ema:,.2f}")
        else:
            opportunities.append(f"{pid}: Insufficient data for setup scan")

        if isinstance(high, (int, float)) and price > 0:
            dist_to_high_pct = (float(high) - price) / price * 100.0
            if 0.0 <= dist_to_high_pct <= breakout_band_pct:
                opportunities.append(f"{pid}: Watch breakout above 24h high {float(high):,.2f}")

    # Keep the list compact for the dashboard.
    return opportunities[:10]


def _knowledge_vault_root() -> Path:
    return Path(os.getenv("KNOWLEDGE_VAULT_ROOT", "research_agent/knowledge_vault"))


def _vault_date_dir(run_date: date) -> Path:
    return _knowledge_vault_root() / run_date.isoformat()


def write_daily_json_outputs(
    run_date: date,
    *,
    levels: Dict[str, Any],
    regimes: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Write daily JSON artifacts into the knowledge vault date directory."""

    out_dir = _vault_date_dir(run_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels_path = out_dir / "coinbase_levels.json"
    regimes_path = out_dir / "regime_status.json"

    def _atomic_write(path: Path, payload: Any) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    _atomic_write(levels_path, levels)
    _atomic_write(regimes_path, regimes)
    return levels_path, regimes_path


def upsert_regime_history_csv(
    path: Path,
    *,
    run_date: date,
    price: float,
    ema: float,
    regime: str,
) -> None:
    """Upsert one row per day into the regime history CSV (idempotent for Streamlit reruns)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    if path.exists():
        try:
            with path.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if not row:
                        continue
                    rows.append({k: (v or "") for k, v in row.items()})
        except Exception:
            rows = []

    date_str = run_date.isoformat()
    new_row = {
        "date": date_str,
        "price": f"{float(price):.8f}",
        "ema": f"{float(ema):.8f}",
        "regime": str(regime),
    }

    replaced = False
    for idx, row in enumerate(rows):
        if str(row.get("date", "")).strip() == date_str:
            rows[idx] = new_row
            replaced = True
            break
    if not replaced:
        rows.append(new_row)

    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "price", "ema", "regime"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def load_trading_alpha_markdown(run_date: Optional[date] = None) -> Optional[str]:
    """Load trading alpha notes from the knowledge vault if present."""

    vault = _knowledge_vault_root()
    candidates: List[Path] = []
    if run_date is not None:
        candidates.append(_vault_date_dir(run_date) / "1_trading_alpha.md")
    candidates.append(vault / "1_trading_alpha.md")
    for path in candidates:
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
        except Exception:
            continue
    return None
