#!/usr/bin/env python3
"""
Generate lightweight health metrics for the watchdog pipeline.

Primary capabilities:
  * Summarise closed-trade performance using the existing watchdog CSV
  * Optionally pull live unrealised PnL from Coinbase INTX perp positions
  * Emit metrics to stdout/JSON/Prometheus text format
  * Serve a minimal HTTP endpoint for Prometheus scraping (no extra deps required)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from coinbaseservice import CoinbaseService
from watchdog_close_old_positions import _get_portfolio_uuid
from watchdog_stats import StatsResult, compute_metrics

LOGGER = logging.getLogger(__name__)

UTC = timezone.utc
DEFAULT_CSV = Path("trade_logs/watchdog_closed_positions.csv")


try:  # pragma: no cover - optional local module
    from credentials import get_perps_credentials  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - local dev fallback
    try:
        from config import API_KEY_PERPS, API_SECRET_PERPS  # type: ignore
    except ModuleNotFoundError:
        API_KEY_PERPS = ""  # type: ignore
        API_SECRET_PERPS = ""  # type: ignore

    def get_perps_credentials() -> Tuple[str, str]:
        return (API_KEY_PERPS or "", API_SECRET_PERPS or "")


def _load_closed_positions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "closed_at" not in df.columns:
        raise ValueError("CSV missing required column 'closed_at'")
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
    df["profit_loss"] = pd.to_numeric(df["profit_loss"], errors="coerce")
    df["profit_loss_pct"] = pd.to_numeric(df.get("profit_loss_pct"), errors="coerce")
    df = df.dropna(subset=["closed_at"]).sort_values("closed_at")
    return df


def _stats_for(df: pd.DataFrame) -> Optional[StatsResult]:
    if df.empty:
        return None
    try:
        return compute_metrics(df, starting_equity=1000.0)
    except Exception as exc:
        LOGGER.debug("Failed to compute stats: %s", exc)
        return None


def _recent_slice(df: pd.DataFrame, *, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = datetime.now(tz=UTC) - timedelta(days=days)
    return df.loc[df["closed_at"] >= cutoff]


def _from_currency(container: object, default: float = 0.0) -> float:
    if container is None:
        return default
    if isinstance(container, (int, float)):
        return float(container)
    if isinstance(container, str):
        try:
            return float(container)
        except ValueError:
            return default
    if isinstance(container, dict):
        for key in ("value", "amount", "rawCurrency", "userNativeCurrency"):
            if key in container:
                try:
                    return float(container[key])
                except (TypeError, ValueError):
                    continue
        return default
    try:
        return float(container)
    except (TypeError, ValueError):
        return default


def fetch_unrealized_pnl() -> Tuple[float, int]:
    key, secret = get_perps_credentials()
    if not key or not secret:
        return float("nan"), 0
    try:
        cb = CoinbaseService(key, secret)
    except Exception as exc:
        LOGGER.debug("Failed to initialise CoinbaseService: %s", exc)
        return float("nan"), 0

    portfolio_uuid = _get_portfolio_uuid(cb)
    if not portfolio_uuid:
        return float("nan"), 0

    try:
        response = cb.client.list_perps_positions(portfolio_uuid=portfolio_uuid)
    except Exception as exc:
        LOGGER.debug("Failed to list positions: %s", exc)
        return float("nan"), 0

    positions = []
    if isinstance(response, dict):
        positions = response.get("positions", []) or []
        summary_obj = response.get("summary")
    else:
        positions = getattr(response, "positions", []) or []
        summary_obj = getattr(response, "summary", None)

    aggregated = float("nan")
    if summary_obj is not None:
        aggregated = _from_currency(summary_obj.get("aggregated_pnl"))
    if np.isnan(aggregated):
        pnl_values = []
        for pos in positions:
            if isinstance(pos, dict):
                pnl_values.append(_from_currency(pos.get("aggregated_pnl")))
            else:
                pnl_values.append(_from_currency(getattr(pos, "aggregated_pnl", None)))
        if pnl_values:
            aggregated = float(np.nansum(pnl_values))

    active_positions = 0
    for pos in positions:
        if isinstance(pos, dict):
            size = pos.get("net_size") or 0
        else:
            size = getattr(pos, "net_size", 0)
        try:
            size = float(size)
        except Exception:
            size = 0.0
        if abs(size) > 0:
            active_positions += 1

    return aggregated, active_positions


@dataclass
class MetricsSnapshot:
    generated_at: datetime
    total_trades: int
    win_rate_pct: float
    expectancy_currency: float
    max_drawdown_pct: float
    recent_win_rate_pct_7d: float
    recent_expectancy_currency_7d: float
    recent_trades_7d: int
    recent_expectancy_currency_30d: float
    recent_trades_30d: int
    trades_last_24h: int
    latest_closed_at: Optional[datetime]
    latest_closed_age_seconds: float
    unrealized_pnl: float
    open_positions: int
    health_level: str
    health_reasons: Tuple[str, ...]


def _determine_health(snapshot: MetricsSnapshot) -> Tuple[str, Tuple[str, ...]]:
    reasons = []
    level = "ok"

    if snapshot.total_trades == 0:
        return "warn", ("No closed trades recorded yet.",)

    if snapshot.latest_closed_age_seconds > 12 * 3600:
        level = "warn"
        reasons.append("Logs older than 12h.")

    if snapshot.recent_trades_7d >= 5 and snapshot.recent_expectancy_currency_7d < 0:
        level = "warn"
        reasons.append("Negative expectancy over last 7 days.")

    if snapshot.trades_last_24h == 0 and snapshot.latest_closed_age_seconds > 24 * 3600:
        level = "warn"
        reasons.append("No trades in last 24h.")

    return level, tuple(reasons)


def build_snapshot(df: pd.DataFrame, include_unrealized: bool = True) -> MetricsSnapshot:
    main_stats = _stats_for(df) or StatsResult(
        total_trades=0,
        wins=0,
        losses=0,
        breakevens=0,
        win_rate_pct=0.0,
        expectancy_currency=0.0,
        max_drawdown_pct=0.0,
        average_r=None,
        expectancy_r=None,
        starting_equity_used=1000.0,
        ending_equity=None,
        std_dev_currency=0.0,
    )

    recent_7 = _stats_for(_recent_slice(df, days=7))
    recent_30 = _stats_for(_recent_slice(df, days=30))
    last_24 = _recent_slice(df, days=1)

    latest_closed_at: Optional[datetime] = None
    latest_age = float("inf")
    if not df.empty:
        latest_closed_at = df["closed_at"].max()
        if pd.notna(latest_closed_at):
            latest_closed_at = latest_closed_at.to_pydatetime()
            latest_age = (datetime.now(tz=UTC) - latest_closed_at).total_seconds()

    unrealized = float("nan")
    open_positions = 0
    if include_unrealized:
        unrealized, open_positions = fetch_unrealized_pnl()

    snapshot = MetricsSnapshot(
        generated_at=datetime.now(tz=UTC),
        total_trades=main_stats.total_trades,
        win_rate_pct=main_stats.win_rate_pct,
        expectancy_currency=main_stats.expectancy_currency,
        max_drawdown_pct=main_stats.max_drawdown_pct,
        recent_win_rate_pct_7d=recent_7.win_rate_pct if recent_7 else float("nan"),
        recent_expectancy_currency_7d=recent_7.expectancy_currency if recent_7 else float("nan"),
        recent_trades_7d=recent_7.total_trades if recent_7 else 0,
        recent_expectancy_currency_30d=recent_30.expectancy_currency if recent_30 else float("nan"),
        recent_trades_30d=recent_30.total_trades if recent_30 else 0,
        trades_last_24h=len(last_24),
        latest_closed_at=latest_closed_at,
        latest_closed_age_seconds=latest_age if np.isfinite(latest_age) else float("inf"),
        unrealized_pnl=unrealized,
        open_positions=open_positions,
        health_level="ok",
        health_reasons=(),
    )

    level, reasons = _determine_health(snapshot)
    snapshot.health_level = level
    snapshot.health_reasons = reasons
    return snapshot


def snapshot_to_prometheus(snapshot: MetricsSnapshot) -> str:
    metrics = {
        "watchdog_total_trades": snapshot.total_trades,
        "watchdog_win_rate_pct": snapshot.win_rate_pct,
        "watchdog_expectancy_currency": snapshot.expectancy_currency,
        "watchdog_max_drawdown_pct": snapshot.max_drawdown_pct,
        "watchdog_recent_win_rate_pct_7d": snapshot.recent_win_rate_pct_7d,
        "watchdog_recent_expectancy_currency_7d": snapshot.recent_expectancy_currency_7d,
        "watchdog_recent_trades_7d": snapshot.recent_trades_7d,
        "watchdog_recent_expectancy_currency_30d": snapshot.recent_expectancy_currency_30d,
        "watchdog_recent_trades_30d": snapshot.recent_trades_30d,
        "watchdog_trades_last_24h": snapshot.trades_last_24h,
        "watchdog_latest_closed_age_seconds": snapshot.latest_closed_age_seconds,
        "watchdog_unrealized_pnl": snapshot.unrealized_pnl,
        "watchdog_open_positions": snapshot.open_positions,
        "watchdog_health": 1 if snapshot.health_level == "ok" else 0,
    }
    lines = [
        "# HELP watchdog_health 1 indicates healthy pipeline, 0 indicates warnings.",
        "# TYPE watchdog_health gauge",
        f"watchdog_health {metrics.pop('watchdog_health')}",
    ]
    for name, value in metrics.items():
        lines.append(f"# TYPE {name} gauge")
        val = "nan" if value is None or (isinstance(value, float) and np.isnan(value)) else value
        lines.append(f"{name} {val}")
    return "\n".join(lines) + "\n"


def snapshot_to_json(snapshot: MetricsSnapshot) -> str:
    payload = asdict(snapshot)
    if snapshot.latest_closed_at:
        payload["latest_closed_at"] = snapshot.latest_closed_at.isoformat()
    payload["generated_at"] = snapshot.generated_at.isoformat()
    payload["health_reasons"] = list(snapshot.health_reasons)
    return json.dumps(payload, indent=2)


def snapshot_to_table(snapshot: MetricsSnapshot) -> str:
    lines = [
        f"Generated: {snapshot.generated_at.isoformat()}",
        f"Total trades: {snapshot.total_trades}",
        f"Win rate (%): {snapshot.win_rate_pct:.2f}",
        f"Expectancy ($): {snapshot.expectancy_currency:.2f}",
        f"Max drawdown (%): {snapshot.max_drawdown_pct:.2f}",
        f"Recent win rate 7d (%): {snapshot.recent_win_rate_pct_7d:.2f}",
        f"Recent expectancy 7d ($): {snapshot.recent_expectancy_currency_7d:.2f}",
        f"Recent trades 7d: {snapshot.recent_trades_7d}",
        f"Recent expectancy 30d ($): {snapshot.recent_expectancy_currency_30d:.2f}",
        f"Recent trades 30d: {snapshot.recent_trades_30d}",
        f"Trades last 24h: {snapshot.trades_last_24h}",
        f"Latest closed at: {snapshot.latest_closed_at.isoformat() if snapshot.latest_closed_at else 'n/a'}",
        f"Latest closed age (s): {snapshot.latest_closed_age_seconds:.0f}",
        f"Unrealised PnL ($): {snapshot.unrealized_pnl:.2f}",
        f"Open positions: {snapshot.open_positions}",
        f"Health level: {snapshot.health_level}",
    ]
    if snapshot.health_reasons:
        lines.append("Health notes: " + "; ".join(snapshot.health_reasons))
    return "\n".join(lines)


class _MetricsHTTPHandler(BaseHTTPRequestHandler):
    snapshot_provider: Optional[callable] = None  # type: ignore[misc]

    def do_GET(self) -> None:  # pragma: no cover - simple IO
        if not self.snapshot_provider:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"No snapshot provider configured.")
            return
        try:
            snapshot = self.snapshot_provider()
            payload = snapshot_to_prometheus(snapshot)
        except Exception as exc:
            LOGGER.error("Failed to produce metrics: %s", exc)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(exc).encode("utf-8"))
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))


def _serve_http(snapshot_provider, port: int) -> None:  # pragma: no cover - simple server
    handler = _MetricsHTTPHandler
    handler.snapshot_provider = snapshot_provider
    with HTTPServer(("0.0.0.0", port), handler) as server:
        LOGGER.info("Serving watchdog metrics on port %s", port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            LOGGER.info("Shutting down metrics server.")
            server.shutdown()


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export watchdog metrics for monitoring.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to watchdog_closed_positions.csv")
    parser.add_argument("--no-unrealized", action="store_true", help="Skip Coinbase calls for unrealised PnL.")
    parser.add_argument("--export", choices=("table", "json", "prometheus"), default="table", help="Output format.")
    parser.add_argument("--output", type=Path, help="Write metrics to this file instead of stdout.")
    parser.add_argument("--serve-port", type=int, help="Run an HTTP metrics endpoint on the given port.")
    parser.add_argument("--interval", type=float, default=30.0, help="Seconds between refreshes when serving HTTP.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO).")
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(asctime)s - %(levelname)s - %(message)s")


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    try:
        df = _load_closed_positions(args.csv)
    except Exception as exc:
        LOGGER.error("Unable to load closed positions: %s", exc)
        raise SystemExit(1) from exc

    def _snapshot_provider() -> MetricsSnapshot:
        return build_snapshot(df.copy(), include_unrealized=not args.no_unrealized)

    snapshot = _snapshot_provider()

    if args.export == "json":
        payload = snapshot_to_json(snapshot)
    elif args.export == "prometheus":
        payload = snapshot_to_prometheus(snapshot)
    else:
        payload = snapshot_to_table(snapshot)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        LOGGER.info("Metrics written to %s", args.output)
    else:
        sys.stdout.write(payload + ("\n" if not payload.endswith("\n") else ""))

    if args.serve_port:
        def refresher():
            nonlocal df, snapshot
            while True:
                try:
                    df = _load_closed_positions(args.csv)
                    snapshot = build_snapshot(df.copy(), include_unrealized=not args.no_unrealized)
                except Exception as exc:
                    LOGGER.error("Metrics refresh failed: %s", exc)
                time.sleep(max(1.0, args.interval))

        Thread(target=refresher, daemon=True).start()
        _serve_http(lambda: snapshot, args.serve_port)


if __name__ == "__main__":
    main()
